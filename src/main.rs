#![allow(clippy::too_many_arguments, clippy::type_complexity)]

//! `ultravec` research CLI.
//!
//!   ultravec bench  --dataset data/sift_base.fvecs [--query-file data/sift_query.fvecs]
//!                    [--query-max 1000] [--bits 1,2,3,4] [--seed 42]
//!
//! Dataset arguments are paths to Texmex `.fvecs` files. Results are written to
//! `results/` as Markdown and also printed.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use ultravec::{bench, datasets, l2_norm, VectorBackend};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        usage();
        return Ok(());
    }
    let opts = parse_opts(&args[2..])?;
    match args[1].as_str() {
        "bench" => cmd_bench(&opts),
        "ivf" => cmd_ivf(&opts),
        "hnsw" => cmd_hnsw(&opts),
        "diag" => cmd_diag(&opts),
        "recon" => cmd_recon(&opts),
        "stream" => cmd_stream(&opts),
        "coldstart" => cmd_coldstart(&opts),
        "companion" => cmd_companion(&opts),
        "-h" | "--help" | "help" => {
            usage();
            Ok(())
        }
        other => {
            eprintln!("unknown subcommand: {other}\n");
            usage();
            bail!("unknown subcommand");
        }
    }
}

fn usage() {
    eprintln!(
        "ultravec — low-bit vector-search research artifact\n\n\
         USAGE:\n  \
         ultravec bench  --dataset <PATH.fvecs> [--bits 1,2,3,4] [--queries N] [--seed S] [--max N]\n\n\
         --max   cap vectors loaded (0 = all)\n"
    );
}

const FLAG_OPTIONS: &[&str] = &["sota3", "sota", "aniso", "pq", "lean", "graph"];
const VALUE_OPTIONS: &[&str] = &[
    "dataset",
    "max",
    "bits",
    "queries",
    "seed",
    "reps",
    "query-file",
    "query-max",
    "gold-file",
    "metric",
    "shortlist",
    "shortlist-scorer",
    "shortlist-rnorm",
    "nlist",
    "nprobe",
    "codecs",
    "rerank",
    "m",
    "ef-construction",
    "ef",
    "calib",
    "groups",
    "window",
    "sample-db",
    "sample-q",
    "backend",
    "decode-bench",
    "out",
    "per-query-out",
    "fit",
    "budget-bytes",
];

/// Strict dependency-free `--key value` / `--flag` parser.
fn parse_opts(args: &[String]) -> Result<HashMap<String, String>> {
    let mut m = HashMap::new();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        let Some(key) = a.strip_prefix("--") else {
            bail!("unexpected positional argument {a:?}; options must start with --");
        };
        anyhow::ensure!(!m.contains_key(key), "duplicate option --{key}");
        if FLAG_OPTIONS.contains(&key) {
            m.insert(key.to_string(), "true".to_string());
            i += 1;
        } else if VALUE_OPTIONS.contains(&key) {
            let value = args
                .get(i + 1)
                .filter(|value| !value.starts_with("--"))
                .with_context(|| format!("--{key} requires a value"))?;
            validate_option(key, value)?;
            m.insert(key.to_string(), value.clone());
            i += 2;
        } else {
            bail!("unknown option --{key}");
        }
    }
    Ok(m)
}

/// Parse a single `--bits` value, rejecting the comma-list form accepted by the
/// multi-rate `bench` subcommand.
fn single_bits(opts: &HashMap<String, String>, default: u8) -> Result<u8> {
    match opts.get("bits") {
        None => Ok(default),
        Some(raw) => raw
            .trim()
            .parse::<u8>()
            .with_context(|| format!("this subcommand takes a single --bits value, got {raw:?}")),
    }
}

fn validate_option(key: &str, value: &str) -> Result<()> {
    let numeric = [
        "max",
        "queries",
        "seed",
        "reps",
        "query-max",
        "nlist",
        "m",
        "ef-construction",
        "groups",
        "window",
        "sample-db",
        "sample-q",
        "decode-bench",
        "budget-bytes",
    ];
    let numeric_list = ["shortlist", "nprobe", "ef", "calib", "rerank"];
    if numeric.contains(&key) {
        value
            .parse::<usize>()
            .with_context(|| format!("--{key} expects a non-negative integer, got {value:?}"))?;
    } else if numeric_list.contains(&key) {
        anyhow::ensure!(
            !value.is_empty(),
            "--{key} requires a non-empty integer list"
        );
        for part in value.split(',') {
            part.trim().parse::<usize>().with_context(|| {
                format!("--{key} expects a comma-separated integer list, got {value:?}")
            })?;
        }
    } else if key == "bits" {
        for part in value.split(',') {
            let bits = part
                .trim()
                .parse::<u8>()
                .with_context(|| format!("--bits expects integers, got {value:?}"))?;
            anyhow::ensure!((1..=16).contains(&bits), "--bits values must be in 1..=16");
        }
    } else if key == "metric" {
        anyhow::ensure!(
            matches!(value, "ip" | "cosine"),
            "--metric must be ip or cosine"
        );
    } else if key == "backend" {
        anyhow::ensure!(
            matches!(
                value,
                "trellis"
                    | "trellis-lr"
                    | "rabitq"
                    | "pvq"
                    | "baseline"
                    | "blockquant"
                    | "pq"
                    | "ultraquant"
                    | "dehub"
            ),
            "unsupported --backend {value:?}"
        );
    } else {
        anyhow::ensure!(!value.is_empty(), "--{key} may not be empty");
    }
    Ok(())
}

/// FastScan companion (lever A): a 1-bit sign-code shortlist → trellis rerank of the
/// top-C, sweeping C to trace the recall-vs-candidate-count (QPS) curve.
///   ultravec companion --dataset X.fvecs --query-file Q.fvecs [--bits 2] [--shortlist 50,100,...] [--reps 3]
fn cmd_companion(opts: &HashMap<String, String>) -> Result<()> {
    use ultravec::bench::{companion_pareto, render_companion_markdown};
    let ds = load_dataset(opts)?;
    let bits: u8 = single_bits(opts, 2)?;
    let seed: u64 = opts.get("seed").and_then(|s| s.parse().ok()).unwrap_or(42);
    let reps: usize = opts.get("reps").and_then(|s| s.parse().ok()).unwrap_or(3);
    let shortlist_cs: Vec<usize> = opts
        .get("shortlist")
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![50, 100, 200, 500, 1000, 2000]);
    let qf = opts
        .get("query-file")
        .context("--query-file required for companion")?;
    let qmax: usize = opts
        .get("query-max")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let queries = datasets::load_fvecs(Path::new(qf), qmax)?;
    anyhow::ensure!(
        queries.dim == ds.dim,
        "query dim {} != db dim {} (after pow2 pad)",
        queries.dim,
        ds.dim
    );
    eprintln!(
        "companion: {} — {} db, dim {}, {} queries, {}-bit, shortlist {:?}, reps {}, seed {}",
        ds.name,
        ds.vectors.len(),
        ds.dim,
        queries.vectors.len(),
        bits,
        shortlist_cs,
        reps,
        seed
    );
    let report = companion_pareto(
        &ds.name,
        ds.dim,
        &ds.vectors,
        &queries.vectors,
        bits,
        &shortlist_cs,
        seed,
        reps,
    );
    let md = render_companion_markdown(&report);
    let path = results_dir().join(format!("companion-{}-b{}.md", ds.name, bits));
    std::fs::write(&path, &md)?;
    println!("{md}");
    eprintln!("→ wrote {}", path.display());
    Ok(())
}

fn load_dataset(opts: &HashMap<String, String>) -> Result<datasets::Dataset> {
    let spec = opts
        .get("dataset")
        .map(String::as_str)
        .context("--dataset <PATH.fvecs> is required")?;
    let max: usize = opts.get("max").and_then(|s| s.parse().ok()).unwrap_or(0);
    datasets::load_fvecs(Path::new(spec), max)
}

fn results_dir() -> PathBuf {
    // Crate-relative results/ regardless of CWD, unless scratch or diagnostic
    // consumers redirect output with ULTRAVEC_RESULTS_DIR.
    let base = match std::env::var_os("ULTRAVEC_RESULTS_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("results"),
    };
    std::fs::create_dir_all(&base).ok();
    base
}

fn cmd_bench(opts: &HashMap<String, String>) -> Result<()> {
    let ds = load_dataset(opts)?;
    let bits: Vec<u8> = opts
        .get("bits")
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![5]);
    let n_queries: usize = opts
        .get("queries")
        .and_then(|s| s.parse().ok())
        .unwrap_or(150);
    let seed: u64 = opts.get("seed").and_then(|s| s.parse().ok()).unwrap_or(42);
    println!(
        "bench: {} — {} vectors, dim {}, bits {:?}, seed {}",
        ds.name,
        ds.len(),
        ds.dim,
        bits,
        seed
    );

    let set = if opts.contains_key("sota3") {
        bench::VariantSet::Sota3
    } else if opts.contains_key("sota") {
        bench::VariantSet::Sota
    } else if opts.contains_key("aniso") {
        bench::VariantSet::Aniso
    } else if opts.contains_key("pq") {
        bench::VariantSet::Pq
    } else if opts.contains_key("lean") {
        bench::VariantSet::Lean
    } else {
        bench::VariantSet::Full
    };
    let metric_ip = opts.get("metric").map(|s| s == "ip").unwrap_or(false);
    let report = if let Some(gf) = opts.get("gold-file") {
        // Recall against externally supplied relevant-id sets.
        let qf = opts
            .get("query-file")
            .context("--gold-file requires --query-file")?;
        let queries = datasets::load_fvecs(std::path::Path::new(qf), 0)?;
        let gold = datasets::load_ivecs(std::path::Path::new(gf))?;
        println!(
            "  gold-file: {} queries, {} gold rows",
            queries.len(),
            gold.len()
        );
        bench::benchmark_with_gold(
            &ds.name,
            ds.dim,
            &ds.vectors,
            &queries.vectors,
            &gold,
            &bits,
            set,
            seed,
        )
    } else if metric_ip {
        // Inner-product (MIPS) ground truth ranks by raw ⟨q,v⟩.
        if let Some(qf) = opts.get("query-file") {
            let qmax: usize = opts
                .get("query-max")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            let queries = datasets::load_fvecs(std::path::Path::new(qf), qmax)?;
            println!(
                "  IP metric, external queries: {} from {}",
                queries.len(),
                qf
            );
            bench::benchmark_ip(
                &ds.name,
                ds.dim,
                &ds.vectors,
                &queries.vectors,
                &bits,
                set,
                seed,
            )
        } else {
            let (db, queries) = ds.split_queries(n_queries, seed);
            println!("  IP metric, {} held-out queries", queries.len());
            bench::benchmark_ip(&ds.name, ds.dim, &db, &queries, &bits, set, seed)
        }
    } else if let Some(qf) = opts.get("query-file") {
        // Held-out external queries (valid protocol vs official RaBitQ): full db,
        // cosine gt over the whole db.
        let qmax: usize = opts
            .get("query-max")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let queries = datasets::load_fvecs(std::path::Path::new(qf), qmax)?;
        println!("  external queries: {} from {}", queries.len(), qf);
        bench::benchmark_external(
            &ds.name,
            ds.dim,
            &ds.vectors,
            &queries.vectors,
            &bits,
            set,
            seed,
        )
    } else {
        bench::benchmark(&ds, &bits, n_queries, seed, set)
    };
    println!("  evaluated queries: {}", report.n_queries);
    let md = bench::render_markdown(&report);
    let path = results_dir().join(format!("flat-{}.md", report.dataset));
    std::fs::write(&path, &md).with_context(|| format!("write {}", path.display()))?;
    let per_query_path = opts
        .get("per-query-out")
        .map(PathBuf::from)
        .unwrap_or_else(|| results_dir().join(format!("flat-{}-per-query.csv", report.dataset)));
    if let Some(parent) = per_query_path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    std::fs::write(&per_query_path, bench::render_per_query_csv(&report))
        .with_context(|| format!("write {}", per_query_path.display()))?;
    println!("{md}");
    println!("→ wrote {}", path.display());
    println!("→ wrote {}", per_query_path.display());
    Ok(())
}

/// IVF recall-vs-QPS Pareto. Builds an
/// IVF index per codec ONCE and sweeps nprobe, reporting recall@10 (vs exact-cosine
/// ground truth) plus QPS at each operating point. The trellis is measured in both
/// reconstruction-cache and codes-only modes. This reuses the existing codecs and
/// the benchmark's `kmeans` coarse quantizer.
///   ultravec ivf --dataset data/sift/sift_base.fvecs --query-file data/sift/sift_query.fvecs \
///     --bits 2 --nlist 1024 [--max 1000000] [--nprobe 1,2,4,8,16,32,64,128] \
///     [--codecs baseline,rabitq,trellis_recon,trellis_codes,pvq,trellis_shortlist] \
///     [--shortlist 128,256,512] [--shortlist-scorer asym|hamming] [--shortlist-rnorm 0|1] \
///     [--query-max N] [--reps 3] [--seed 42]
fn cmd_ivf(opts: &HashMap<String, String>) -> Result<()> {
    use ultravec::ivf::{ivf_pareto, render_ivf_markdown, IvfCodec};
    let ds = load_dataset(opts)?;
    let bits: u8 = single_bits(opts, 2)?;
    let seed: u64 = opts.get("seed").and_then(|s| s.parse().ok()).unwrap_or(42);
    let reps: usize = opts.get("reps").and_then(|s| s.parse().ok()).unwrap_or(3);
    // nlist default ≈ √N (the textbook IVF rule of thumb).
    let nlist: usize = opts
        .get("nlist")
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| (ds.vectors.len() as f64).sqrt().round() as usize)
        .max(2);
    let nprobes: Vec<usize> = opts
        .get("nprobe")
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 2, 4, 8, 16, 32, 64, 128]);
    // Shortlist sizes for the two-stage arm. One arm per C, so the row is a
    // recall/throughput curve rather than a single sampled point.
    let shortlist_cs: Vec<usize> = opts
        .get("shortlist")
        .map(|s| {
            s.split(',')
                .filter_map(|x| x.trim().parse::<usize>().ok())
                .filter(|&c| c > 0)
                .collect()
        })
        .unwrap_or_else(|| vec![256]);
    // Stage-1 scorer for the shortlist arms. `asym` keeps the query in full precision
    // (⟨q_r, sign(u_r)⟩); `hamming` binarizes the query too. Selected here so the two
    // are an A/B through one driver rather than a process-global.
    let shortlist_asym = opts
        .get("shortlist-scorer")
        .map(|s| match s.trim() {
            "asym" | "asymmetric" => true,
            "hamming" | "ham" | "symmetric" => false,
            other => {
                eprintln!("ivf: unknown --shortlist-scorer '{other}' — using asym");
                true
            }
        })
        .unwrap_or(true);
    let shortlist_rnorm = opts
        .get("shortlist-rnorm")
        .map(|s| s.trim() != "0")
        .unwrap_or(true);
    let codecs: Vec<IvfCodec> = opts
        .get("codecs")
        .map(|s| {
            s.split(',')
                .flat_map(|x| match x.trim() {
                    "baseline" | "turboquant_baseline" => vec![IvfCodec::Baseline],
                    "rabitq" => vec![IvfCodec::Rabitq],
                    "trellis_recon" | "trellis-recon" => vec![IvfCodec::TrellisRecon],
                    "trellis_codes" | "trellis-codes" => vec![IvfCodec::TrellisCodes],
                    "pvq" => vec![IvfCodec::Pvq],
                    // Previously unselectable by name even though it ran by default,
                    // so `--codecs` could not reproduce the default arm set.
                    "trellis_shortlist" | "trellis-shortlist" => shortlist_cs
                        .iter()
                        .map(|&c| IvfCodec::TrellisShortlist {
                            c,
                            asym: shortlist_asym,
                            rnorm: shortlist_rnorm,
                        })
                        .collect(),
                    other => {
                        eprintln!("ivf: unknown codec '{other}' — skipping");
                        vec![]
                    }
                })
                .collect()
        })
        .unwrap_or_else(|| {
            let mut all = vec![
                IvfCodec::Baseline,
                IvfCodec::Rabitq,
                IvfCodec::TrellisRecon,
                IvfCodec::TrellisCodes,
                IvfCodec::Pvq,
            ];
            all.extend(shortlist_cs.iter().map(|&c| IvfCodec::TrellisShortlist {
                c,
                asym: shortlist_asym,
                rnorm: shortlist_rnorm,
            }));
            all
        });
    anyhow::ensure!(!codecs.is_empty(), "no valid codecs selected");
    let qf = opts
        .get("query-file")
        .context("--query-file required for ivf")?;
    let qmax: usize = opts
        .get("query-max")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let queries = datasets::load_fvecs(Path::new(qf), qmax)?;
    anyhow::ensure!(
        queries.dim == ds.dim,
        "query dim {} != db dim {} (after pow2 pad)",
        queries.dim,
        ds.dim
    );
    eprintln!(
        "ivf: {} — {} db, dim {}, {} queries, {}-bit, nlist {}, nprobe {:?}, reps {}, seed {}",
        ds.name,
        ds.vectors.len(),
        ds.dim,
        queries.vectors.len(),
        bits,
        nlist,
        nprobes,
        reps,
        seed
    );
    // `--rerank <depths>` runs the deployed-metric study (recall@QPS WITH exact fp32 rerank) at a
    // single nprobe (first of --nprobe, or 16), sweeping the rerank depth, instead of the pre-rerank
    // Pareto. Writes ivf-rerank.md.
    if let Some(rd) = opts.get("rerank") {
        use ultravec::ivf::{ivf_rerank_pareto, render_rerank_markdown};
        let rdepths: Vec<usize> = rd
            .split(',')
            .filter_map(|x| x.trim().parse().ok())
            .collect();
        let np = nprobes.first().copied().unwrap_or(16);
        anyhow::ensure!(
            !rdepths.is_empty(),
            "--rerank needs depths, e.g. --rerank 0,10,50,100,200"
        );
        let report = ivf_rerank_pareto(
            &ds.name,
            ds.dim,
            &ds.vectors,
            &queries.vectors,
            bits,
            nlist,
            np,
            &rdepths,
            &codecs,
            seed,
            reps,
        );
        let md = render_rerank_markdown(&report);
        let path = results_dir().join("ivf-rerank.md");
        std::fs::write(&path, &md).with_context(|| format!("write {}", path.display()))?;
        println!("{md}");
        println!("→ wrote {}", path.display());
        return Ok(());
    }
    let report = ivf_pareto(
        &ds.name,
        ds.dim,
        &ds.vectors,
        &queries.vectors,
        bits,
        nlist,
        &nprobes,
        &codecs,
        seed,
        reps,
    );
    let md = render_ivf_markdown(&report);
    let path = results_dir().join("ivf-qps-pareto.md");
    std::fs::write(&path, &md).with_context(|| format!("write {}", path.display()))?;
    println!("{md}");
    println!("→ wrote {}", path.display());
    Ok(())
}

/// From-scratch pure-Rust HNSW codec drop-in: build the graph under each codec's reconstruction and
/// report recall@10 vs QPS sweeping the query-time `ef`. Shows the recall edge that survives an IVF also
/// survives the dominant production graph index.
/// `ultravec hnsw --dataset X.fvecs --query-file Q.fvecs [--max N] [--bits 2] [--m 16]
/// [--ef-construction 200] [--ef 16,32,64,128,256] [--codecs fp32,trellis,rabitq,turboquant_baseline]
/// [--seed 42] [--reps 3] [--query-max N]`
fn cmd_hnsw(opts: &HashMap<String, String>) -> Result<()> {
    use ultravec::hnsw::{hnsw_pareto, render_hnsw_markdown, HnswCodec};
    let ds = load_dataset(opts)?;
    let bits: u8 = single_bits(opts, 2)?;
    let seed: u64 = opts.get("seed").and_then(|s| s.parse().ok()).unwrap_or(42);
    let reps: usize = opts.get("reps").and_then(|s| s.parse().ok()).unwrap_or(3);
    let m: usize = opts.get("m").and_then(|s| s.parse().ok()).unwrap_or(16);
    let ef_construction: usize = opts
        .get("ef-construction")
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let efs: Vec<usize> = opts
        .get("ef")
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![16, 32, 64, 128, 256]);
    let codecs: Vec<HnswCodec> = opts
        .get("codecs")
        .map(|s| {
            s.split(',')
                .filter_map(|x| match x.trim() {
                    "fp32" => Some(HnswCodec::Fp32),
                    "baseline" | "turboquant_baseline" => Some(HnswCodec::Baseline),
                    "rabitq" => Some(HnswCodec::Rabitq),
                    "trellis" | "trellis_recon" | "trellis-recon" => Some(HnswCodec::Trellis),
                    "trellis_codes_only" | "trellis-codes-only" | "trellis_codes" => {
                        Some(HnswCodec::TrellisCodesOnly)
                    }
                    other => {
                        eprintln!("hnsw: unknown codec '{other}' — skipping");
                        None
                    }
                })
                .collect()
        })
        .unwrap_or_else(|| {
            vec![
                HnswCodec::Fp32,
                HnswCodec::Trellis,
                HnswCodec::TrellisCodesOnly,
                HnswCodec::Rabitq,
                HnswCodec::Baseline,
            ]
        });
    anyhow::ensure!(!codecs.is_empty(), "no valid codecs selected");
    let qf = opts
        .get("query-file")
        .context("--query-file required for hnsw")?;
    let qmax: usize = opts
        .get("query-max")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let queries = datasets::load_fvecs(Path::new(qf), qmax)?;
    anyhow::ensure!(
        queries.dim == ds.dim,
        "query dim {} != db dim {} (after pow2 pad)",
        queries.dim,
        ds.dim
    );
    eprintln!(
        "hnsw: {} — {} db, dim {}, {} queries, {}-bit, M {}, ef_construction {}, ef {:?}, reps {}, seed {}",
        ds.name,
        ds.vectors.len(),
        ds.dim,
        queries.vectors.len(),
        bits,
        m,
        ef_construction,
        efs,
        reps,
        seed
    );
    let report = hnsw_pareto(
        &ds.name,
        ds.dim,
        &ds.vectors,
        &queries.vectors,
        bits,
        m,
        ef_construction,
        &efs,
        &codecs,
        seed,
        reps,
    );
    let md = render_hnsw_markdown(&report);
    let path = results_dir().join("hnsw-qps-pareto.md");
    std::fs::write(&path, &md).with_context(|| format!("write {}", path.display()))?;
    println!("{md}");
    println!("→ wrote {}", path.display());
    Ok(())
}

/// Cold-start experiment (oblivious vs data-dependent PQ at small calibration).
/// `ultravec coldstart --dataset X.fvecs --query-file Q.fvecs [--max N] [--bits 2]
/// [--calib 64,128,256,512,1000,2000,5000,20000,0] [--seed 42] [--query-max 200]`
/// (calib 0 = full corpus). Honors `ULTRAVEC_TRELLIS_MEM`.
fn cmd_coldstart(opts: &HashMap<String, String>) -> Result<()> {
    let ds = load_dataset(opts)?;
    let bits: u8 = single_bits(opts, 2)?;
    let seed: u64 = opts.get("seed").and_then(|s| s.parse().ok()).unwrap_or(42);
    let calib: Vec<usize> = opts
        .get("calib")
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![64, 128, 256, 512, 1000, 2000, 5000, 20000, 0]);
    let qf = opts
        .get("query-file")
        .context("--query-file required for coldstart")?;
    let qmax: usize = opts
        .get("query-max")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let queries = datasets::load_fvecs(Path::new(qf), qmax)?;
    eprintln!(
        "coldstart: {} — {} db, dim {}, {} queries, {}-bit, calib {:?}",
        ds.name,
        ds.vectors.len(),
        ds.dim,
        queries.vectors.len(),
        bits,
        calib
    );
    let report = bench::cold_start_sweep(
        &ds.name,
        ds.dim,
        &ds.vectors,
        &queries.vectors,
        bits,
        &calib,
        seed,
    );
    let md = bench::render_cold_start_markdown(&report);
    let path = results_dir().join(format!("coldstart-{}.md", report.dataset));
    std::fs::write(&path, &md).with_context(|| format!("write {}", path.display()))?;
    println!("{md}");
    println!("→ wrote {}", path.display());
    Ok(())
}

/// Streaming-drift experiment (recall under churn vs refit cost).
/// `ultravec stream --dataset X.fvecs --query-file Q.fvecs [--max N] [--groups 10]
/// [--bits 2] [--seed 42] [--query-max 200]`. Honors `ULTRAVEC_TRELLIS_MEM`.
fn cmd_stream(opts: &HashMap<String, String>) -> Result<()> {
    let ds = load_dataset(opts)?;
    let bits: u8 = single_bits(opts, 2)?;
    let groups: usize = opts
        .get("groups")
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let seed: u64 = opts.get("seed").and_then(|s| s.parse().ok()).unwrap_or(42);
    // --window W → delete+insert sliding-window churn (live = last W clusters);
    // absent → insert-only growing prefix.
    let window: Option<usize> = opts.get("window").and_then(|s| s.parse().ok());
    let qf = opts
        .get("query-file")
        .context("--query-file required for stream")?;
    let qmax: usize = opts
        .get("query-max")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let queries = datasets::load_fvecs(Path::new(qf), qmax)?;
    eprintln!(
        "stream: {} — {} db, dim {}, {} queries, {} clusters, {}-bit, window {:?}",
        ds.name,
        ds.vectors.len(),
        ds.dim,
        queries.vectors.len(),
        groups,
        bits,
        window
    );
    // --graph → the graph-ANN (NSW) rung: build a proximity graph over each codec's
    // decoded vectors so drift can corrupt graph STRUCTURE, not just query scores.
    if opts.contains_key("graph") {
        let m: usize = opts.get("m").and_then(|s| s.parse().ok()).unwrap_or(16);
        let ef: usize = opts.get("ef").and_then(|s| s.parse().ok()).unwrap_or(64);
        let report = bench::graph_stream_drift(
            &ds.name,
            ds.dim,
            &ds.vectors,
            &queries.vectors,
            bits,
            groups,
            seed,
            m,
            ef,
        );
        let md = bench::render_graph_stream_markdown(&report);
        let path = results_dir().join(format!("graph-stream-{}.md", report.dataset));
        std::fs::write(&path, &md).with_context(|| format!("write {}", path.display()))?;
        println!("{md}");
        println!("→ wrote {}", path.display());
        return Ok(());
    }
    let report = bench::stream_drift(
        &ds.name,
        ds.dim,
        &ds.vectors,
        &queries.vectors,
        bits,
        groups,
        seed,
        window,
    );
    let md = bench::render_stream_markdown(&report);
    let path = results_dir().join(format!("stream-{}.md", report.dataset));
    std::fs::write(&path, &md).with_context(|| format!("write {}", path.display()))?;
    println!("{md}");
    println!("→ wrote {}", path.display());
    Ok(())
}

/// Estimator-variance diagnostic. Builds the fixed-codec comparison backends and
/// logs g=⟨ō,o⟩ / κ / σ_pred / σ_emp / bias / MSE so the M-sweep can test the
/// predicted variance chain. Trellis M comes from `ULTRAVEC_TRELLIS_MEM`.
fn cmd_diag(opts: &HashMap<String, String>) -> Result<()> {
    let ds = load_dataset(opts)?;
    let bits: u8 = single_bits(opts, 2)?;
    let sample_db: usize = opts
        .get("sample-db")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);
    let sample_q: usize = opts
        .get("sample-q")
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let n_queries: usize = opts
        .get("queries")
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let seed: u64 = opts.get("seed").and_then(|s| s.parse().ok()).unwrap_or(42);
    let set = bench::VariantSet::Sota3;
    println!(
        "diag: {} — {} vectors, dim {}, bits {}, seed {}",
        ds.name,
        ds.len(),
        ds.dim,
        bits,
        seed
    );
    let (rows, n_db, n_q) = if let Some(qf) = opts.get("query-file") {
        let qmax: usize = opts
            .get("query-max")
            .and_then(|s| s.parse().ok())
            .unwrap_or(sample_q);
        let q = datasets::load_fvecs(Path::new(qf), qmax)?;
        println!("  external queries: {} from {}", q.len(), qf);
        let r = bench::diag_estimator_variance(
            ds.dim,
            &ds.vectors,
            &q.vectors,
            bits,
            set,
            sample_db,
            sample_q,
            seed,
        );
        (r, ds.vectors.len(), q.vectors.len().min(sample_q))
    } else {
        let (db, queries) = ds.split_queries(n_queries, seed);
        let r = bench::diag_estimator_variance(
            ds.dim, &db, &queries, bits, set, sample_db, sample_q, seed,
        );
        (r, db.len(), queries.len().min(sample_q))
    };
    let md = bench::render_diag_markdown(&ds.name, ds.dim, n_db, n_q, seed, &rows);
    let mem = std::env::var("ULTRAVEC_TRELLIS_MEM").unwrap_or_else(|_| "6".into());
    let path = results_dir().join(format!("diag-{}-m{}-b{}.md", ds.name, mem, bits));
    std::fs::write(&path, &md).with_context(|| format!("write {}", path.display()))?;
    println!("{md}");
    println!("→ wrote {}", path.display());
    Ok(())
}

/// Reconstruct each row with one backend and write full vectors (unit direction
/// times exact norm) as `.fvecs` for codec-level ANN evaluation.
///   ultravec recon --dataset V.fvecs --backend trellis --bits 2 --out Vhat.fvecs
fn cmd_recon(opts: &HashMap<String, String>) -> Result<()> {
    let ds = load_dataset(opts)?;
    let bits: u8 = single_bits(opts, 2)?;
    let name = opts.get("backend").map(String::as_str).unwrap_or("trellis");
    // Decode-only throughput probe (trellis): encode the index once, then time the
    // decode walk independently of the one-time index construction cost.
    // from the one-time Viterbi ENCODE. `recon --backend trellis --decode-bench <rounds>`.
    if let Some(r) = opts.get("decode-bench") {
        let rounds: usize = r.parse().unwrap_or(3);
        let mut q = ultravec::trellis::TrellisQuantizer::new(ds.dim, bits);
        let t0 = std::time::Instant::now();
        q.add_batch(&ds.vectors);
        let enc = t0.elapsed().as_secs_f64();
        let n = ds.vectors.len();
        let t1 = std::time::Instant::now();
        let dn = q.decode_bench(rounds);
        let dec = t1.elapsed().as_secs_f64();
        println!(
            "decode-bench {name} bits={bits} dim={}: {n} vecs  ENCODE {enc:.3}s ({:.0} vec/s)  DECODE-only {dec:.4}s ({:.0} vec/s)  ratio enc/dec={:.0}x",
            ds.dim, n as f64 / enc, dn as f64 / dec, (n as f64 / enc).recip() / (dn as f64 / dec).recip()
        );
        return Ok(());
    }
    let out = opts.get("out").context("--out <path.fvecs> required")?;
    // Centering is a property of the comparison, not of one codec: `bench` applies it
    // to the trellis, RaBitQ, PVQ and E8 together so a head-to-head never crosses
    // conventions. Reconstruction honours the same switch for the same reason --
    // centering one side only would manufacture exactly the asymmetry this flag
    // exists to rule out. Off by default, so every retained result is unchanged.
    let centered = std::env::var("ULTRAVEC_CENTER").ok().as_deref() == Some("1");
    // A mean is corpus state, so the honest question is not whether centering helps
    // but how little data buys it. `ULTRAVEC_CENTER_K=<n>` estimates it from a seeded
    // n-vector draw instead of the whole base, which puts the centered variant on the
    // same cold-start axis as PQ rather than exempting it from one.
    let mean = || {
        let draw: usize = std::env::var("ULTRAVEC_CENTER_K")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        if draw == 0 || draw >= ds.vectors.len() {
            return ultravec::unit_mean(ds.dim, &ds.vectors);
        }
        // The draw is a random variable like PQ's calibration set, so it is seeded
        // per draw rather than fixed: a single sample cannot show whether an
        // eight-vector mean is reliably good or was one lucky eight.
        let mut state = std::env::var("ULTRAVEC_CENTER_SEED")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(42);
        let mut index: Vec<usize> = (0..ds.vectors.len()).collect();
        for position in 0..draw {
            let pick =
                position + (ultravec::splitmix64(&mut state) as usize) % (index.len() - position);
            index.swap(position, pick);
        }
        let sample: Vec<Vec<f32>> = index[..draw]
            .iter()
            .map(|&i| ds.vectors[i].clone())
            .collect();
        ultravec::unit_mean(ds.dim, &sample)
    };
    let backend: Box<dyn VectorBackend> = match name {
        "trellis" => {
            let quantizer = ultravec::trellis::TrellisQuantizer::new(ds.dim, bits);
            Box::new(if centered {
                quantizer.with_mean(mean())
            } else {
                quantizer
            })
        }
        // L1 data-insight: the trellis with a LEARNED rotation (fit on this db) instead of the
        // oblivious Hadamard one. Rotation kind via ULTRAVEC_ROTATION={pca,itq}. Tests whether a
        // data-aware rotation alone closes the warm gap to data-dependent PQ.
        "trellis-lr" => Box::new(
            ultravec::trellis::TrellisQuantizer::new(ds.dim, bits).with_rotation_fit(&ds.vectors)),
        // RaBitQ's published construction quantizes the residual to a centroid; an
        // IVF build therefore always centers, on its list centroid. Reconstruction
        // here is centroid-free by default, which is the right default for the flat
        // comparison (a flat scan has no partition) but is NOT what an upstream
        // single-list build does -- that list has one centroid, the global mean.
        // `ULTRAVEC_CENTER=1` matches it, so the parity check can separate an
        // implementation gap from a centering difference. Default unchanged.
        "rabitq" => {
            let quantizer = ultravec::rabitq::RaBitQ::new(ds.dim, bits);
            Box::new(if centered {
                quantizer.with_mean(mean())
            } else {
                quantizer
            })
        }
        "pvq" => Box::new(ultravec::pvq::PvqQuantizer::new(ds.dim, bits)),
        "baseline" => Box::new(ultravec::baseline::TurboQuantBaseline::new(ds.dim, bits)),
        // Oblivious block-sphere VQ (codebook from the rotation-invariant marginal, oblivious
        // rotation -- NO data fit), so it can serve as the coarse coder in an oblivious
        // block-residual hybrid (PCA-head style, but BlockQuant instead of PQ).
        "blockquant" => Box::new(ultravec::blockquant::BlockQuantizer::new(ds.dim, bits)),
        // PQ is data-dependent: it trains its codebook on the recon input (its k-means
        // sees this db). That is exactly the property the rare-cell test probes — a
        // density-weighted codebook under-serves the rare directions.
        "pq" => Box::new(ultravec::pq::ProductQuantizer::train(
            ds.dim, bits, ultravec::pq::PqLoss::Mse, &ds.vectors)),
        // L2 data-insight: the distribution-matched codec — a Lloyd-Max codebook fit to THIS
        // corpus's pooled (rotated) marginal (DataAware), vs the trellis's fixed Gaussian code.
        // Tests whether a data-matched codebook closes the warm gap to PQ.
        "ultraquant" => Box::new(
            ultravec::ultraquant::UltraQuant::from_corpus(ds.dim, bits, true, &ds.vectors)),
        // PCA-head residual hybrid: quantized PCA head (rank chosen adaptively) + oblivious trellis residual.
        // Fits the basis on `--fit <db.fvecs>` (default: the dataset itself) and applies it to the
        // `--dataset` being reconstructed — so base + query share one DB-fit basis, as the Python
        // this hybrid does. `ULTRAVEC_DEHUB_R=<r>` forces a fixed rank (for fixed-r validation).
        "dehub" => {
            let fit = match opts.get("fit") {
                Some(p) => ultravec::datasets::load_fvecs(std::path::Path::new(p), usize::MAX)
                    .with_context(|| format!("load --fit {p}"))?
                    .vectors,
                None => ds.vectors.clone(),
            };
            match opts.get("budget-bytes").and_then(|s| s.parse::<usize>().ok()) {
                Some(budget) => Box::new(ultravec::dehub::DehubBackend::fit_budget(ds.dim, budget, &fit)),
                None => Box::new(ultravec::dehub::DehubBackend::fit(ds.dim, bits, &fit)),
            }
        }
        other => bail!("unknown backend '{other}' (trellis|rabitq|pvq|baseline|blockquant|pq|ultraquant|dehub)"),
    };
    // Parallel reconstruct: reconstruct_unit is &self and VectorBackend: Send + Sync, so
    // each row decodes independently. Order-preserving collect ⇒ byte-identical to the
    // serial path.
    use rayon::prelude::*;
    let recons: Vec<Vec<f32>> = ds
        .vectors
        .par_iter()
        .map(|v| {
            let norm = l2_norm(v);
            backend
                .reconstruct_unit(v)
                .map(|u| u.iter().map(|r| r * norm).collect::<Vec<f32>>())
        })
        .collect::<Option<Vec<Vec<f32>>>>()
        .with_context(|| format!("backend '{name}' has no reconstruct_unit"))?;
    let mut f = BufWriter::new(File::create(out).with_context(|| format!("create {out}"))?);
    for recon in &recons {
        f.write_all(&(ds.dim as i32).to_le_bytes())?;
        for r in recon {
            f.write_all(&r.to_le_bytes())?;
        }
    }
    f.flush()?;
    println!(
        "recon: {name} {bits}-bit → {} rows ({}) → {out}",
        recons.len(),
        ds.name
    );
    Ok(())
}

#[cfg(test)]
mod cli_tests {
    use super::parse_opts;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn strict_parser_accepts_known_options() {
        let parsed = parse_opts(&args(&[
            "--dataset",
            "data.fvecs",
            "--bits",
            "1,2,4",
            "--sota3",
        ]))
        .unwrap();
        assert_eq!(parsed["bits"], "1,2,4");
        assert_eq!(parsed["sota3"], "true");
    }

    #[test]
    fn strict_parser_accepts_fit_path_and_reconstruction_backends() {
        let parsed = parse_opts(&args(&[
            "--backend",
            "dehub",
            "--fit",
            "data/base.fvecs",
            "--dataset",
            "data/query.fvecs",
        ]))
        .unwrap();
        assert_eq!(parsed["fit"], "data/base.fvecs");

        for backend in [
            "trellis",
            "trellis-lr",
            "rabitq",
            "pvq",
            "baseline",
            "blockquant",
            "pq",
            "ultraquant",
            "dehub",
        ] {
            assert!(parse_opts(&args(&["--backend", backend])).is_ok());
        }
    }

    #[test]
    fn strict_parser_rejects_unknown_missing_and_invalid_options() {
        assert!(parse_opts(&args(&["--unknown", "x"])).is_err());
        assert!(parse_opts(&args(&["--dataset"])).is_err());
        assert!(parse_opts(&args(&["orphan"])).is_err());
        assert!(parse_opts(&args(&["--bits", "0"])).is_err());
        assert!(parse_opts(&args(&["--seed", "not-a-number"])).is_err());
    }
}
