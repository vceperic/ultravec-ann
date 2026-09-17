//! Flat codec benchmark. For a dataset, build exact-cosine ground truth, then
//! measure each quantizer's approximate recall@k (the headline), query latency,
//! and resident bytes. Comparisons use the TurboQuant reference at matched bits.
//!
//! **Pre-rerank recall is the primary metric.** An exact-f32 rerank pass restores
//! ~1.0 recall for *every* approximate backend, masking the quantizer's own
//! quality, so the benchmark reports the raw quantizer.
use rayon::prelude::*;

use std::time::Instant;

use crate::{
    baseline::TurboQuantBaseline,
    blockquant::BlockQuantizer,
    coset_trellis::CosetTrellis,
    datasets::Dataset,
    e8::E8Quantizer,
    eden::EdenQuantizer,
    pq::{PqLoss, ProductQuantizer},
    pvq::PvqQuantizer,
    rabitq::RaBitQ,
    residual::TurboQuantResidual,
    trellis::TrellisQuantizer,
    ultraquant::{Target, UltraQuant},
    ItemId, VectorBackend,
};

/// Which variant set to benchmark.
#[derive(Clone, Copy, PartialEq)]
pub enum VariantSet {
    /// Marginal-shape variants plus the reference and residual codec (small corpora).
    Full,
    /// Baseline + QJL residual only (fast at scale).
    Lean,
    /// Reference plus product-quantization variants (bits must divide 8).
    Pq,
    /// Baseline + training-free anisotropic λ-sweep + PQ ceiling.
    Aniso,
    /// TurboQuant, RaBitQ, the anisotropic variant, and PQ at the requested rate,
    /// plus one-bit RaBitQ as a low-rate reference.
    Sota,
    /// Fixed-codec comparison: TurboQuant, RaBitQ, trellis, PVQ, EDEN,
    /// BlockQuant, and E8 at its supported rates.
    Sota3,
}

/// Recall@k values reported by every benchmark.
pub const KS: [usize; 3] = [1, 10, 100];

pub struct BenchRow {
    pub quantizer: String,
    pub bits: u8,
    pub recall: [f64; 3], // aligned with KS
    /// Per-query Recall@k values, retained for paired uncertainty estimates and
    /// audit-friendly CSV output.
    pub per_query_recall: Vec<[f64; 3]>,
    pub query_us_p50: f64,
    pub code_bytes: usize,
    pub resident_bytes: usize,
    /// Reconstruction MSE of the unit direction (rank-distortion study); None if
    /// the backend can't reconstruct.
    pub mse: Option<f64>,
}

pub struct PairedRecallDelta {
    pub comparator: String,
    pub bits: u8,
    /// UltraVec minus comparator, in percentage points.
    pub delta_pp: f64,
    pub ci_low_pp: f64,
    pub ci_high_pp: f64,
}

pub struct BenchReport {
    pub dataset: String,
    pub dim: usize,
    pub n_db: usize,
    pub n_queries: usize,
    pub seed: u64,
    pub rows: Vec<BenchRow>,
    /// (variant, bits, Δpp@10 vs baseline) for every non-baseline row.
    pub deltas: Vec<(String, u8, f64)>,
    /// Paired query-bootstrap comparisons of UltraVec against every other codec.
    pub paired_r10: Vec<PairedRecallDelta>,
    pub has_three_point_gain: bool,
}

fn with_mips_norm_sidecar(
    memory: &crate::MemoryBreakdown,
    n_db: usize,
    norm_already_stored: bool,
) -> (usize, usize) {
    // TurboQuant's serialized record already contains the exact source norm;
    // directional-only codecs need a separate f32 norm for MIPS. Never charge
    // the same scalar twice merely because this benchmark keeps a generic lookup
    // vector outside the backend.
    let norm_bytes = if norm_already_stored {
        0
    } else {
        n_db * std::mem::size_of::<f32>()
    };
    (
        memory.code_bytes + norm_bytes,
        memory.total_resident_bytes() + norm_bytes,
    )
}

/// Exact cosine top-`k` ids for one query over `db`.
/// Descending by similarity, ascending by id on a tie.
///
/// The previous ordering compared similarity alone under an unstable sort, so the
/// order of tied candidates was an artifact of the sort's internal state. Making the
/// order total is what allows the selection below to be partial rather than a full
/// sort without changing which ids come back.
#[inline]
fn by_sim_then_id(a: &(ItemId, f32), b: &(ItemId, f32)) -> std::cmp::Ordering {
    b.1.total_cmp(&a.1).then(a.0.cmp(&b.0))
}

/// Exact cosine top-`k` ids for one query over `db`, given the database norms.
///
/// `db_norms` is passed in rather than recomputed: `cosine` takes the norm of BOTH
/// arguments, so computing gold for Q queries over N vectors recomputed every
/// database norm Q times -- on GIST that is a thousand recomputations of a
/// 1024-dimensional norm for each of a hundred thousand vectors, and it dominated
/// the ground truth far more than the codecs it exists to score.
fn exact_topk_with(query: &[f32], db: &[Vec<f32>], db_norms: &[f32], k: usize) -> Vec<ItemId> {
    let qn = crate::l2_norm(query);
    if qn < f32::EPSILON {
        return Vec::new();
    }
    let mut scored: Vec<(ItemId, f32)> = db
        .iter()
        .zip(db_norms)
        .enumerate()
        .map(|(i, (v, &nb))| {
            let sim = if nb < f32::EPSILON {
                0.0
            } else {
                let dot: f32 = query.iter().zip(v).map(|(x, y)| x * y).sum();
                dot / (qn * nb)
            };
            (i as ItemId, sim)
        })
        .collect();
    select_top_k(&mut scored, k)
}

/// Keep the `k` best under [`by_sim_then_id`] without sorting the whole array.
fn select_top_k(scored: &mut Vec<(ItemId, f32)>, k: usize) -> Vec<ItemId> {
    let k = k.min(scored.len());
    if k < scored.len() {
        scored.select_nth_unstable_by(k, by_sim_then_id);
        scored.truncate(k);
    }
    scored.sort_unstable_by(by_sim_then_id);
    scored.iter().map(|(id, _)| *id).collect()
}

fn exact_topk(query: &[f32], db: &[Vec<f32>], k: usize) -> Vec<ItemId> {
    let norms: Vec<f32> = db.iter().map(|v| crate::l2_norm(v)).collect();
    exact_topk_with(query, db, &norms, k)
}

/// Per-query Recall@k of `got` (approximate) against `gold` (exact).
fn recall_per_query(gold: &[Vec<ItemId>], got: &[Vec<ItemId>]) -> Vec<[f64; 3]> {
    gold.iter()
        .zip(got)
        .map(|(g, a)| {
            let mut out = [0.0; 3];
            for (ki, &k) in KS.iter().enumerate() {
                let gset: std::collections::HashSet<ItemId> = g.iter().take(k).copied().collect();
                let aset: std::collections::HashSet<ItemId> = a.iter().take(k).copied().collect();
                out[ki] = if gset.is_empty() {
                    0.0
                } else {
                    gset.intersection(&aset).count() as f64 / gset.len() as f64
                };
            }
            out
        })
        .collect()
}

fn mean_recall(per_query: &[[f64; 3]]) -> [f64; 3] {
    let mut out = [0.0; 3];
    if per_query.is_empty() {
        return out;
    }
    for values in per_query {
        for (sum, value) in out.iter_mut().zip(values) {
            *sum += value;
        }
    }
    for value in &mut out {
        *value /= per_query.len() as f64;
    }
    out
}

/// Recall@k of `got` (approximate) against `gold` (exact), for each k in [`KS`].
fn recall_against(gold: &[Vec<ItemId>], got: &[Vec<ItemId>]) -> [f64; 3] {
    mean_recall(&recall_per_query(gold, got))
}

const BOOTSTRAP_SAMPLES: usize = 10_000;

fn paired_bootstrap_r10(
    ultravec: &[[f64; 3]],
    comparator: &[[f64; 3]],
    mut state: u64,
) -> (f64, f64, f64) {
    assert_eq!(ultravec.len(), comparator.len());
    if ultravec.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let differences: Vec<f64> = ultravec
        .iter()
        .zip(comparator)
        .map(|(u, c)| (u[1] - c[1]) * 100.0)
        .collect();
    let delta = differences.iter().sum::<f64>() / differences.len() as f64;
    let mut bootstrap = Vec::with_capacity(BOOTSTRAP_SAMPLES);
    for _ in 0..BOOTSTRAP_SAMPLES {
        let mut sum = 0.0;
        for _ in 0..differences.len() {
            let index = crate::splitmix64(&mut state) as usize % differences.len();
            sum += differences[index];
        }
        bootstrap.push(sum / differences.len() as f64);
    }
    bootstrap.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let low = bootstrap[((BOOTSTRAP_SAMPLES - 1) as f64 * 0.025).round() as usize];
    let high = bootstrap[((BOOTSTRAP_SAMPLES - 1) as f64 * 0.975).round() as usize];
    (delta, low, high)
}

fn stable_label_seed(label: &str) -> u64 {
    label.bytes().fold(0xcbf2_9ce4_8422_2325, |state, byte| {
        (state ^ byte as u64).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn append_paired_r10(output: &mut Vec<PairedRecallDelta>, rows: &[BenchRow], bits: u8, seed: u64) {
    let Some(ultravec) = rows.iter().find(|row| row.quantizer == "trellis") else {
        return;
    };
    for comparator in rows.iter().filter(|row| row.quantizer != "trellis") {
        let bootstrap_seed =
            seed ^ ((bits as u64) << 56) ^ stable_label_seed(&comparator.quantizer);
        let (delta_pp, ci_low_pp, ci_high_pp) = paired_bootstrap_r10(
            &ultravec.per_query_recall,
            &comparator.per_query_recall,
            bootstrap_seed,
        );
        output.push(PairedRecallDelta {
            comparator: comparator.quantizer.clone(),
            bits,
            delta_pp,
            ci_low_pp,
            ci_high_pp,
        });
    }
}

/// Run one backend over the query set, returning (per-query topk-ids, p50 query µs).
fn run_backend(
    backend: &(dyn VectorBackend + Sync),
    queries: &[Vec<f32>],
    max_k: usize,
) -> (Vec<Vec<ItemId>>, f64) {
    use rayon::prelude::*;
    // Parallelize the per-query scan (the single-threaded bottleneck at 100k x N_q).
    // Recall is identical (each query is scored independently); the query-µs below
    // is wall-time under load, not a single-core latency.
    let pairs: Vec<(Vec<ItemId>, f64)> = queries
        .par_iter()
        .map(|q| {
            let t = Instant::now();
            let res = backend.search(q, max_k);
            let us = t.elapsed().as_secs_f64() * 1e6;
            (res.into_iter().map(|(id, _)| id).collect(), us)
        })
        .collect();
    let mut times: Vec<f64> = pairs.iter().map(|p| p.1).collect();
    let got: Vec<Vec<ItemId>> = pairs.into_iter().map(|p| p.0).collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = times.get(times.len() / 2).copied().unwrap_or(0.0);
    (got, p50)
}

/// KMeans (Lloyd's, seeded, fixed iters) over the unit-normalized db vectors →
/// `nlist` coarse centroids for IVF-style per-cluster centering. Assignment is by
/// cosine (= dot on unit vectors); centroids are the per-cluster means (not
/// re-normalized — they're the cluster centroid, matching RaBitQ's `c`).
///
/// `pub(crate)` so the IVF index ([`crate::ivf`]) reuses the SAME coarse quantizer
/// the per-cluster-centering benchmark uses — one k-means, two consumers (DRY).
pub(crate) fn kmeans(dim: usize, db: &[Vec<f32>], nlist: usize, seed: u64) -> Vec<Vec<f32>> {
    let units: Vec<Vec<f32>> = db
        .iter()
        .map(|v| {
            let n = crate::l2_norm(v).max(f32::EPSILON);
            v.iter().map(|x| x / n).collect()
        })
        .collect();
    // Seeded init: distinct random vectors as initial centroids.
    let mut st = seed;
    let mut centroids: Vec<Vec<f32>> = (0..nlist)
        .map(|_| units[(crate::splitmix64(&mut st) as usize) % units.len()].clone())
        .collect();
    for _ in 0..5 {
        // Assignment (the O(N·nlist·dim) hotspot) in parallel — each unit's nearest
        // centroid is an independent argmax (first-wins tie via strict `>`, so the
        // result is deterministic and order-independent). The accumulation below then
        // sums in the ORIGINAL unit order, keeping the centroid means byte-identical
        // to the prior serial path (only the assignment is parallelized, not the
        // float reduction). Lets a 1M×1024 IVF coarse build use all cores.
        use rayon::prelude::*;
        let assign: Vec<usize> = units
            .par_iter()
            .map(|u| {
                let mut best = 0usize;
                let mut bd = f32::NEG_INFINITY;
                for (j, c) in centroids.iter().enumerate() {
                    let d: f32 = u.iter().zip(c).map(|(a, b)| a * b).sum();
                    if d > bd {
                        bd = d;
                        best = j;
                    }
                }
                best
            })
            .collect();
        let mut sums = vec![vec![0.0f32; dim]; nlist];
        let mut counts = vec![0usize; nlist];
        for (u, &best) in units.iter().zip(&assign) {
            counts[best] += 1;
            for (s, x) in sums[best].iter_mut().zip(u) {
                *s += x;
            }
        }
        for (j, (s, &cnt)) in sums.iter().zip(&counts).enumerate() {
            if cnt > 0 {
                centroids[j] = s.iter().map(|x| x / cnt as f32).collect();
            }
        }
    }
    centroids
}

/// The quantizer variants to benchmark at a given bit-width.
///
/// Two groups:
///   * **rotation ON** — same pipeline as the baseline. The post-rotation
///     marginal is approximately Gaussian, providing the matched-rotation comparison.
///   * **rotation OFF** (`*_raw`) — the rotation-skip hypothesis. On a raw
///     heavy-tailed corpus the matched codebook *could* beat the baseline while
///     saving the O(d·log d) rotation. `ultra_gaussian_raw` isolates the cost of
///     dropping rotation alone (Gaussian codebook, no rotation).
fn variants(
    dim: usize,
    bits: u8,
    db: &[Vec<f32>],
    set: VariantSet,
) -> Vec<(String, Box<dyn VectorBackend + Sync>)> {
    let baseline = || Box::new(TurboQuantBaseline::new(dim, bits)) as Box<dyn VectorBackend + Sync>;
    // RaBitQ-style centering, applied to BOTH rabitq and trellis so the
    // head-to-head stays fair. ULTRAVEC_IVF_NLIST=N → N KMeans centroids (IVF-style
    // per-cluster centering, the regime RaBitQ ships in); ULTRAVEC_CENTER=1 → single
    // global mean (N=1). Tests whether the trellis edge survives finer centering.
    let nlist: usize = std::env::var("ULTRAVEC_IVF_NLIST")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let center = std::env::var("ULTRAVEC_CENTER").ok().as_deref() == Some("1");
    let centroids: Vec<Vec<f32>> = if nlist >= 2 {
        kmeans(dim, db, nlist, 42)
    } else if nlist == 1 || center {
        vec![crate::unit_mean(dim, db)]
    } else {
        Vec::new()
    };
    let use_c = !centroids.is_empty();
    // `with_rotation_fit(db)` swaps in a learned rotation (PCA/ITQ) when
    // ULTRAVEC_ROTATION requests it; the TurboQuant reference retains its
    // defined oblivious rotation.
    let rabitq = |b: u8| -> Box<dyn VectorBackend + Sync> {
        let q = RaBitQ::new(dim, b).with_rotation_fit(db);
        Box::new(if use_c {
            q.with_centroids(centroids.clone())
        } else {
            q
        })
    };
    let trellis = || -> Box<dyn VectorBackend + Sync> {
        let q = TrellisQuantizer::new(dim, bits).with_rotation_fit(db);
        Box::new(if use_c {
            q.with_centroids(centroids.clone())
        } else {
            q
        })
    };
    // PVQ decode-free spherical code, with the same rotation and centering.
    let pvq = || -> Box<dyn VectorBackend + Sync> {
        let q = PvqQuantizer::new(dim, bits).with_rotation_fit(db);
        Box::new(if use_c {
            q.with_centroids(centroids.clone())
        } else {
            q
        })
    };
    // E8 (Gosset) lattice VQ, with the same rotation and centering.
    // Included when dim%8==0 and bits∈{1,2,3}. Four bits would need a 2^32-point
    // codebook (137 GB), so the ceiling is the enumeration, not the lattice.
    let e8 = || -> Box<dyn VectorBackend + Sync> {
        let q = E8Quantizer::new(dim, bits).with_rotation_fit(db);
        Box::new(if use_c {
            q.with_centroids(centroids.clone())
        } else {
            q
        })
    };
    // EDEN reference (oblivious N(0,1) Lloyd-Max plus optimal scale).
    let eden = || -> Box<dyn VectorBackend + Sync> {
        let q = EdenQuantizer::new(dim, bits).with_rotation_fit(db);
        Box::new(if use_c {
            q.with_centroids(centroids.clone())
        } else {
            q
        })
    };
    // BlockQuant reference (oblivious block-sphere quantization).
    let blockquant = || -> Box<dyn VectorBackend + Sync> {
        let q = BlockQuantizer::new(dim, bits).with_rotation_fit(db);
        Box::new(if use_c {
            q.with_centroids(centroids.clone())
        } else {
            q
        })
    };
    // Bit-matched reference vs product quantization (MSE and anisotropic).
    if set == VariantSet::Pq {
        let aniso = ProductQuantizer::train(dim, bits, PqLoss::Anisotropic { eta: 4.0 }, db);
        let aniso_label = aniso.label().to_string();
        let mse = ProductQuantizer::train(dim, bits, PqLoss::Mse, db);
        let mse_label = mse.label().to_string();
        return vec![
            ("turboquant_baseline".into(), baseline()),
            (mse_label, Box::new(mse)),
            (aniso_label, Box::new(aniso)),
        ];
    }
    // Complete fixed-codec field. E8 only has implementations at one and two
    // bits; its higher-rate cells are intentionally absent rather than replaced
    // by an unmatched configuration.
    if set == VariantSet::Sota3 {
        if std::env::var("ULTRAVEC_BENCH_ONLY").as_deref() == Ok("trellis") {
            return vec![("trellis".into(), trellis())];
        }
        // `turboquant_baseline` is TurboQuant's MSE-optimal reconstruction arm. On a
        // maximum-inner-product benchmark the method's own inner-product construction
        // belongs in the field too, so the sketched residual arm runs alongside it and
        // both are reported; they are bit-matched by construction (see residual.rs),
        // so this costs no rate parity. It needs at least two bits -- the base spends
        // `b-1` and the sketch one -- so, like E8, its unsupported cells are absent
        // rather than filled with an unmatched configuration.
        let mut v: Vec<(String, Box<dyn VectorBackend + Sync>)> = vec![
            ("turboquant_baseline".into(), baseline()),
            ("rabitq".into(), rabitq(bits)),
            ("trellis".into(), trellis()),
            ("pvq".into(), pvq()),
            ("eden".into(), eden()),
            ("blockquant".into(), blockquant()),
        ];
        if bits >= 2 {
            v.insert(
                1,
                (
                    "turboquant_residual".into(),
                    Box::new(TurboQuantResidual::new(dim, bits)),
                ),
            );
        }
        if dim.is_multiple_of(8) && (1..=3).contains(&bits) {
            v.push(("e8".into(), e8()));
            // Research arm, opt-in: E8 cosets on the trellis branches. Off by default
            // so the retained flat bundles reproduce byte-identically.
            if std::env::var("ULTRAVEC_COSET_TRELLIS").as_deref() == Ok("1") {
                let q = CosetTrellis::new(dim, bits);
                let q = if use_c {
                    q.with_centroids(centroids.clone())
                } else {
                    q
                };
                v.push(("coset_trellis".into(), Box::new(q)));
            }
        }
        return v;
    }
    // Broader mixed comparison with RaBitQ at matched bits.
    if set == VariantSet::Sota {
        let mut v: Vec<(String, Box<dyn VectorBackend + Sync>)> = vec![
            ("turboquant_baseline".into(), baseline()),
            ("rabitq".into(), rabitq(bits)),
            ("trellis".into(), trellis()),
            (
                "ultra_aniso_l4".into(),
                Box::new(UltraQuant::new(
                    dim,
                    bits,
                    Target::Anisotropic { lambda: 4.0 },
                    true,
                )),
            ),
            ("rabitq_1bit_ref".into(), rabitq(1)),
        ];
        // PQ needs an integer sub_dim = 8/bits, and dim divisible by it.
        if bits != 0 && 8 % bits == 0 && dim.is_multiple_of(8 / bits as usize) {
            let pq = ProductQuantizer::train(dim, bits, PqLoss::Mse, db);
            v.push((format!("{}_dep", pq.label()), Box::new(pq)));
        }
        return v;
    }
    // Training-free anisotropic scalar codebook λ-sweep (all
    // bit-matched to the baseline — same rotation, same bits, oblivious), plus
    // the data-dependent PQ as the recall ceiling.
    if set == VariantSet::Aniso {
        let mut v: Vec<(String, Box<dyn VectorBackend + Sync>)> =
            vec![("turboquant_baseline".into(), baseline())];
        for lambda in [0.5f64, 1.0, 2.0, 4.0, 8.0, 16.0] {
            v.push((
                format!("ultra_aniso_l{:.1}", lambda),
                Box::new(UltraQuant::new(
                    dim,
                    bits,
                    Target::Anisotropic { lambda },
                    true,
                )),
            ));
        }
        if dim.is_multiple_of(2) {
            let pq = ProductQuantizer::train(dim, bits, PqLoss::Mse, db);
            v.push((format!("{}_ceiling", pq.label()), Box::new(pq)));
        }
        return v;
    }
    // TurboQuant MSE and residual arms, both bit-matched at `bits`.
    let mut v: Vec<(String, Box<dyn VectorBackend + Sync>)> = vec![
        ("turboquant_baseline".into(), baseline()),
        (
            "turboquant_residual".into(),
            Box::new(TurboQuantResidual::new(dim, bits)),
        ),
    ];
    if set == VariantSet::Lean {
        return v;
    }
    // Marginal-shape variants used by the small-corpus comparison.
    let da_rot = UltraQuant::from_corpus(dim, bits, true, db);
    let da_rot_label = format!("{}_rot", da_rot.target().label());
    let da_raw = UltraQuant::from_corpus(dim, bits, false, db);
    let da_raw_label = format!("{}_raw", da_raw.target().label());
    v.extend::<Vec<(String, Box<dyn VectorBackend + Sync>)>>(vec![
        (
            "ultra_t3_rot".into(),
            Box::new(UltraQuant::new(dim, bits, Target::StudentT(3.0), true)),
        ),
        (
            "ultra_t5_rot".into(),
            Box::new(UltraQuant::new(dim, bits, Target::StudentT(5.0), true)),
        ),
        (da_rot_label, Box::new(da_rot)),
        (
            "ultra_gaussian_raw".into(),
            Box::new(UltraQuant::new(dim, bits, Target::Gaussian, false)),
        ),
        (
            "ultra_t5_raw".into(),
            Box::new(UltraQuant::new(dim, bits, Target::StudentT(5.0), false)),
        ),
        (da_raw_label, Box::new(da_raw)),
    ]);
    v
}

/// Benchmark `dataset` at the given bit-widths. Computes exact ground truth once,
/// then evaluates every selected variant and records whether any gains at least
/// three Recall@10 percentage points over the TurboQuant reference.
pub fn benchmark(
    dataset: &Dataset,
    bits_list: &[u8],
    n_queries: usize,
    seed: u64,
    set: VariantSet,
) -> BenchReport {
    let (db, queries) = dataset.split_queries(n_queries, seed);
    run_core(
        &dataset.name,
        dataset.dim,
        &db,
        &queries,
        bits_list,
        set,
        seed,
    )
}

/// External-query benchmark: `db` is the full database, `queries` are provided
/// (e.g. the held-out `sift_query.fvecs`), cosine ground truth computed over the
/// full db. This is the valid protocol for comparing against the official RaBitQ
/// — held-out queries, not queries split from base.
pub fn benchmark_external(
    name: &str,
    dim: usize,
    db: &[Vec<f32>],
    queries: &[Vec<f32>],
    bits_list: &[u8],
    set: VariantSet,
    seed: u64,
) -> BenchReport {
    run_core(name, dim, db, queries, bits_list, set, seed)
}

fn run_core(
    name: &str,
    dim: usize,
    db: &[Vec<f32>],
    queries: &[Vec<f32>],
    bits_list: &[u8],
    set: VariantSet,
    seed: u64,
) -> BenchReport {
    let max_k = *KS.iter().max().unwrap();
    // Ground truth is exact and independent of every codec, so it parallelises
    // cleanly; it was the sequential term in a harness whose codecs all fan out.
    let db_norms: Vec<f32> = db.par_iter().map(|v| crate::l2_norm(v)).collect();
    let gold: Vec<Vec<ItemId>> = queries
        .par_iter()
        .map(|q| exact_topk_with(q, db, &db_norms, max_k))
        .collect();
    let mut rows = Vec::new();
    let mut deltas = Vec::new();
    let mut paired_r10 = Vec::new();
    let mut has_three_point_gain = false;
    for &bits in bits_list {
        let first_row = rows.len();
        let mut baseline_r10 = 0.0;
        for (vname, mut backend) in variants(dim, bits, db, set) {
            backend.add_batch(db);
            let (got, p50) = run_backend(&*backend, queries, max_k);
            let per_query_recall = recall_per_query(&gold, &got);
            let recall = mean_recall(&per_query_recall);
            // Whole loaded base, not a prefix. `.fvecs` files are not shuffled, so a
            // 500-vector prefix was a non-random sample of a statistic the write-up
            // quotes as a corpus property. `reconstruction_mse` is already parallel
            // and this runs once per (codec, rate) cell.
            let mse = crate::reconstruction_mse(&*backend, db);
            if vname == "turboquant_baseline" {
                baseline_r10 = recall[1];
            } else {
                let delta_pp = (recall[1] - baseline_r10) * 100.0;
                deltas.push((vname.clone(), bits, delta_pp));
                if delta_pp >= 3.0 {
                    has_three_point_gain = true;
                }
            }
            let memory = backend.memory_breakdown();
            rows.push(BenchRow {
                quantizer: vname,
                bits,
                recall,
                per_query_recall,
                query_us_p50: p50,
                code_bytes: memory.code_bytes,
                resident_bytes: memory.total_resident_bytes(),
                mse,
            });
        }
        append_paired_r10(&mut paired_r10, &rows[first_row..], bits, seed);
    }
    BenchReport {
        dataset: name.to_string(),
        dim,
        n_db: db.len(),
        n_queries: queries.len(),
        seed,
        rows,
        deltas,
        paired_r10,
        has_three_point_gain,
    }
}

/// Render a `BenchReport` as a markdown results file.
pub fn render_markdown(r: &BenchReport) -> String {
    let mut s = String::new();
    s.push_str(&format!("# Flat codec benchmark — {}\n\n", r.dataset));
    s.push_str(&format!(
        "- dim **{}**, db **{}**, queries **{}**, seed **{}**\n",
        r.dim, r.n_db, r.n_queries, r.seed
    ));
    s.push_str("- recall is **pre-rerank** (the quantizer's own quality; an exact-f32 rerank would restore ~1.0 for all)\n\n");
    if r.dataset.ends_with("-ip") {
        s.push_str(
            "- MIPS code and resident bytes include the exact database norm: already present in TurboQuant's record, otherwise a 4-byte sidecar.\n\n",
        );
    }
    s.push_str(
        "| quantizer | bits | R@1 | R@10 | R@100 | recon MSE | query µs (p50) | code B/vec | resident B/vec |\n",
    );
    s.push_str("|---|---|---|---|---|---|---|---|---|\n");
    for row in &r.rows {
        let code_per_vec = row.code_bytes.checked_div(r.n_db).unwrap_or(0);
        let resident_per_vec = row.resident_bytes.checked_div(r.n_db).unwrap_or(0);
        let mse = row
            .mse
            .map(|m| format!("{:.5}", m))
            .unwrap_or_else(|| "—".into());
        s.push_str(&format!(
            "| {} | {} | {:.3} | {:.3} | {:.3} | {} | {:.1} | {} | {} |\n",
            row.quantizer,
            row.bits,
            row.recall[0],
            row.recall[1],
            row.recall[2],
            mse,
            row.query_us_p50,
            code_per_vec,
            resident_per_vec
        ));
    }
    s.push_str("\n## Δ recall@10 vs TurboQuant baseline (percentage points)\n\n");
    s.push_str("| variant | bits | Δpp@10 |\n|---|---|---|\n");
    for (v, b, d) in &r.deltas {
        s.push_str(&format!("| {} | {} | {:+.2} |\n", v, b, d));
    }
    s.push_str(&format!(
        "\n## Paired query-bootstrap Recall@10 comparisons\n\n\
         UltraVec minus comparator in percentage points; 95% percentile interval \
         from {BOOTSTRAP_SAMPLES} paired resamples. \"lead\"/\"behind\" requires the \
         interval to exclude zero.\n\n"
    ));
    s.push_str("| comparator | bits | Δpp@10 | 95% CI | outcome |\n");
    s.push_str("|---|---|---|---|---|\n");
    for comparison in &r.paired_r10 {
        let outcome = if comparison.ci_low_pp > 0.0 {
            "lead"
        } else if comparison.ci_high_pp < 0.0 {
            "behind"
        } else {
            "tied"
        };
        s.push_str(&format!(
            "| {} | {} | {:+.3} | [{:+.3}, {:+.3}] | {} |\n",
            comparison.comparator,
            comparison.bits,
            comparison.delta_pp,
            comparison.ci_low_pp,
            comparison.ci_high_pp,
            outcome
        ));
    }
    s.push_str(&format!(
        "\n**At least one ≥3pp recall@10 gain over the TurboQuant reference ({}):** {}.\n",
        r.dataset,
        if r.has_three_point_gain { "yes" } else { "no" }
    ));
    s
}

/// Per-query recall values in a stable, analysis-friendly format.
pub fn render_per_query_csv(r: &BenchReport) -> String {
    let mut output =
        String::from("dataset,bits,query,quantizer,recall_at_1,recall_at_10,recall_at_100\n");
    for row in &r.rows {
        for (query, recall) in row.per_query_recall.iter().enumerate() {
            output.push_str(&format!(
                "{},{},{},{},{:.6},{:.6},{:.6}\n",
                r.dataset, row.bits, query, row.quantizer, recall[0], recall[1], recall[2]
            ));
        }
    }
    output
}

// ── Estimator-variance diagnostic ───────────────────────────────────────────
// Measures every term in the variance chain the theory predicts governs the
// trellis-over-RaBitQ recall margin:
//   M ↑ → g=⟨ō,o⟩ ↑ → σ_est = κ(g)/√(D−1) ↓ → fewer top-k inversions → recall ↑
// `g` is the cosine of the unit reconstruction `ō` to the truth `o` — the single
// quantity the *encoder* controls; everything else (D, codebook) is held fixed
// across the head-to-head. We log the predicted per-pair std (from g, via the
// RaBitQ Thm-3.2 bound) AND the empirical std (estimate − exact cosine over
// sampled pairs) so the bound itself is validated on these vectors.

pub struct DiagRow {
    pub backend: String,
    pub bits: u8,
    pub mem: u8,         // trellis state bits M (0 for non-trellis)
    pub mean_g: f64,     // mean ⟨ō,o⟩
    pub mean_kappa: f64, // mean √((1−g²)/g²)
    pub sigma_pred: f64, // mean_kappa / √(D−1) — predicted per-pair estimator std
    pub sigma_emp: f64,  // empirical std of (estimate − exact cosine)
    pub bias_emp: f64,   // mean (estimate − exact) — should be ≈0 (unbiased)
    pub mse: f64,        // reconstruction MSE of the unit direction
    /// False when the backend exposes no reconstruction, so `mean_g`,
    /// `mean_kappa`, `sigma_pred` and `mse` are undefined rather than zero.
    /// An asymmetric estimator such as TurboQuant's sketched residual arm has no
    /// reconstruction to measure direction fidelity against; printing 0.0000 for it
    /// would feed the cross-codec analysis a fabricated observation.
    pub has_recon: bool,
}

/// Build each `set` backend over `db`, then measure g / κ / σ_pred / σ_emp / bias
/// / MSE. `sample_db` db vectors feed the g/MSE pass; `sample_q` queries × the
/// same db sample feed the empirical-error pass. Trellis M is read from the env
/// (so one call per M reuses the existing sweep knob, no recompile).
pub fn diag_estimator_variance(
    dim: usize,
    db: &[Vec<f32>],
    queries: &[Vec<f32>],
    bits: u8,
    set: VariantSet,
    sample_db: usize,
    sample_q: usize,
    seed: u64,
) -> Vec<DiagRow> {
    let mem = std::env::var("ULTRAVEC_TRELLIS_MEM")
        .ok()
        .and_then(|s| s.parse::<u8>().ok())
        .unwrap_or(6);
    let nq = sample_q.min(queries.len());
    let sdb = sample_db.min(db.len());
    // Deterministic db subsample (Fisher-Yates prefix, seeded — R5).
    let mut st = seed;
    let mut order: Vec<usize> = (0..db.len()).collect();
    for i in (1..db.len()).rev() {
        let j = (crate::next_f64(&mut st) * (i as f64 + 1.0)) as usize;
        order.swap(i, j.min(i));
    }
    let db_sample: Vec<usize> = order[..sdb].to_vec();

    let mut rows = Vec::new();
    for (vname, mut backend) in variants(dim, bits, db, set) {
        backend.add_batch(db);
        // g, κ, MSE over the sampled db vectors.
        let (mut sg, mut sk, mut smse, mut ng) = (0.0f64, 0.0f64, 0.0f64, 0u64);
        for &idx in &db_sample {
            let o = &db[idx];
            if let Some(obar) = backend.reconstruct_unit(o) {
                let on = (crate::l2_norm(o).max(f32::EPSILON)) as f64;
                let (mut g, mut mse) = (0.0f64, 0.0f64);
                for (xi, ri) in o.iter().zip(&obar) {
                    let ou = *xi as f64 / on;
                    g += ou * *ri as f64;
                    let d = ou - *ri as f64;
                    mse += d * d;
                }
                let gc = g.clamp(1e-4, 1.0);
                sg += gc;
                sk += (1.0 / (gc * gc) - 1.0).max(0.0).sqrt();
                smse += mse;
                ng += 1;
            }
        }
        let (mean_g, mean_kappa, mse) = if ng > 0 {
            (sg / ng as f64, sk / ng as f64, smse / ng as f64)
        } else {
            (0.0, 0.0, 0.0)
        };
        let sigma_pred = mean_kappa / (dim as f64 - 1.0).max(1.0).sqrt();
        // Empirical per-pair estimator error: backend score minus exact cosine,
        // over sampled (query, db-sample) pairs.
        let (mut se, mut se2, mut npair) = (0.0f64, 0.0f64, 0u64);
        for q in queries.iter().take(nq) {
            let scored = backend.search(q, db.len());
            let mut by_id = vec![f32::NAN; db.len()];
            for (id, s) in scored {
                if (id as usize) < by_id.len() {
                    by_id[id as usize] = s;
                }
            }
            for &idx in &db_sample {
                let e = by_id[idx];
                if e.is_finite() {
                    let err = (e - crate::cosine(q, &db[idx])) as f64;
                    se += err;
                    se2 += err * err;
                    npair += 1;
                }
            }
        }
        let bias_emp = if npair > 0 { se / npair as f64 } else { 0.0 };
        let var = if npair > 0 {
            (se2 / npair as f64 - bias_emp * bias_emp).max(0.0)
        } else {
            0.0
        };
        let sigma_emp = var.sqrt();
        let row_mem = if vname.starts_with("trellis") { mem } else { 0 };
        rows.push(DiagRow {
            backend: vname,
            bits,
            mem: row_mem,
            mean_g,
            mean_kappa,
            sigma_pred,
            sigma_emp,
            bias_emp,
            mse,
            has_recon: ng > 0,
        });
    }
    rows
}

pub fn render_diag_markdown(
    name: &str,
    dim: usize,
    n_db: usize,
    n_q: usize,
    seed: u64,
    rows: &[DiagRow],
) -> String {
    let mut s = String::new();
    s.push_str(&format!("# Estimator-variance diagnostic — {}\n\n", name));
    s.push_str(&format!(
        "- dim **{}**, db **{}**, queries **{}**, seed **{}**\n",
        dim, n_db, n_q, seed
    ));
    s.push_str("- `g=⟨ō,o⟩` recon-to-truth cosine; `κ=√((1−g²)/g²)`; ");
    s.push_str("`σ_pred=κ/√(D−1)` (RaBitQ Thm-3.2 per-pair std); `σ_emp`=std(estimate−exact)\n\n");
    s.push_str("| backend | bits | M | mean g | κ | σ_pred | σ_emp | bias | recon MSE |\n");
    s.push_str("|---|---|---|---|---|---|---|---|---|\n");
    for r in rows {
        // Reconstruction-derived columns read "—" when the backend has none, so a
        // downstream parser skips the row instead of ingesting zeros as data.
        let (g, kappa, pred, mse) = if r.has_recon {
            (
                format!("{:.4}", r.mean_g),
                format!("{:.4}", r.mean_kappa),
                format!("{:.5}", r.sigma_pred),
                format!("{:.5}", r.mse),
            )
        } else {
            ("—".into(), "—".into(), "—".into(), "—".into())
        };
        s.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {:.5} | {:+.5} | {} |\n",
            r.backend, r.bits, r.mem, g, kappa, pred, r.sigma_emp, r.bias_emp, mse
        ));
    }
    s
}

// ── Inner-product (MIPS) ground-truth mode ────────────────────────────────────
// Maximum-inner-product search ranks by inner product ⟨q,v⟩, not cosine.
// The trellis/RaBitQ quantizers compress the *direction* (unit vector); the exact
// per-vector norm ‖v‖ (4 bytes) folds the magnitude back in — `cos·‖v‖` ranks
// identically to ⟨q,v⟩ for a fixed query. So a direction-quantizer + an exact norm
// is a MIPS index, with the backends reused UNCHANGED. Tests whether the
// trellis>RaBitQ ordering survives the IP regime. Reportable on real,
// non-unit-norm data such as raw SIFT descriptors.

/// Exact top-`k` by raw inner product ⟨q,v⟩ (MIPS ground truth — no normalization).
fn exact_topk_ip(query: &[f32], db: &[Vec<f32>], k: usize) -> Vec<ItemId> {
    let mut scored: Vec<(ItemId, f32)> = db
        .iter()
        .enumerate()
        .map(|(i, v)| {
            (
                i as ItemId,
                query.iter().zip(v).map(|(a, b)| a * b).sum::<f32>(),
            )
        })
        .collect();
    select_top_k(&mut scored, k)
}

pub fn benchmark_ip(
    name: &str,
    dim: usize,
    db: &[Vec<f32>],
    queries: &[Vec<f32>],
    bits_list: &[u8],
    set: VariantSet,
    seed: u64,
) -> BenchReport {
    use rayon::prelude::*;

    let max_k = *KS.iter().max().unwrap();
    let gold: Vec<Vec<ItemId>> = queries
        .par_iter()
        .map(|q| exact_topk_ip(q, db, max_k))
        .collect();
    let norms: Vec<f32> = db.iter().map(|v| crate::l2_norm(v)).collect();
    let mut rows = Vec::new();
    let mut deltas = Vec::new();
    let mut paired_r10 = Vec::new();
    let mut has_three_point_gain = false;
    for &bits in bits_list {
        let first_row = rows.len();
        let mut baseline_r10 = 0.0;
        for (vname, mut backend) in variants(dim, bits, db, set) {
            backend.add_batch(db);
            let pairs: Vec<(Vec<ItemId>, f64)> = queries
                .par_iter()
                .map(|q| {
                    let t = Instant::now();
                    let mut scored = backend.search(q, db.len());
                    for score in &mut scored {
                        // cos → ⟨q,v⟩-rank
                        score.1 *= norms.get(score.0 as usize).copied().unwrap_or(1.0);
                    }
                    scored.sort_unstable_by(|a, b| {
                        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let elapsed_us = t.elapsed().as_secs_f64() * 1e6;
                    let ids = scored.into_iter().take(max_k).map(|(id, _)| id).collect();
                    (ids, elapsed_us)
                })
                .collect();
            let (got, mut times): (Vec<Vec<ItemId>>, Vec<f64>) = pairs.into_iter().unzip();
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p50 = times.get(times.len() / 2).copied().unwrap_or(0.0);
            let per_query_recall = recall_per_query(&gold, &got);
            let recall = mean_recall(&per_query_recall);
            // Whole loaded base, not a prefix. `.fvecs` files are not shuffled, so a
            // 500-vector prefix was a non-random sample of a statistic the write-up
            // quotes as a corpus property. `reconstruction_mse` is already parallel
            // and this runs once per (codec, rate) cell.
            let mse = crate::reconstruction_mse(&*backend, db);
            if vname == "turboquant_baseline" {
                baseline_r10 = recall[1];
            } else {
                let d = (recall[1] - baseline_r10) * 100.0;
                deltas.push((vname.clone(), bits, d));
                if d >= 3.0 {
                    has_three_point_gain = true;
                }
            }
            let memory = backend.memory_breakdown();
            let (code_bytes, resident_bytes) =
                with_mips_norm_sidecar(&memory, db.len(), vname == "turboquant_baseline");
            rows.push(BenchRow {
                quantizer: vname,
                bits,
                recall,
                per_query_recall,
                query_us_p50: p50,
                code_bytes,
                resident_bytes,
                mse,
            });
        }
        append_paired_r10(&mut paired_r10, &rows[first_row..], bits, seed);
    }
    BenchReport {
        dataset: format!("{}-ip", name),
        dim,
        n_db: db.len(),
        n_queries: queries.len(),
        seed,
        rows,
        deltas,
        paired_r10,
        has_three_point_gain,
    }
}

/// Recall against externally supplied per-query relevant-id sets rather than
/// recomputed nearest-neighbor ground truth. Backends still score vector
/// candidates by cosine; this function only changes the relevance labels used
/// for evaluation.
pub fn benchmark_with_gold(
    name: &str,
    dim: usize,
    db: &[Vec<f32>],
    queries: &[Vec<f32>],
    gold: &[Vec<ItemId>],
    bits_list: &[u8],
    set: VariantSet,
    seed: u64,
) -> BenchReport {
    let max_k = *KS.iter().max().unwrap();
    let mut rows = Vec::new();
    let mut deltas = Vec::new();
    let mut paired_r10 = Vec::new();
    let mut has_three_point_gain = false;
    for &bits in bits_list {
        let first_row = rows.len();
        let mut baseline_r10 = 0.0;
        for (vname, mut backend) in variants(dim, bits, db, set) {
            backend.add_batch(db);
            let (got, p50) = run_backend(&*backend, queries, max_k);
            let per_query_recall = recall_per_query(gold, &got);
            let recall = mean_recall(&per_query_recall);
            // Whole loaded base, not a prefix. `.fvecs` files are not shuffled, so a
            // 500-vector prefix was a non-random sample of a statistic the write-up
            // quotes as a corpus property. `reconstruction_mse` is already parallel
            // and this runs once per (codec, rate) cell.
            let mse = crate::reconstruction_mse(&*backend, db);
            if vname == "turboquant_baseline" {
                baseline_r10 = recall[1];
            } else {
                let d = (recall[1] - baseline_r10) * 100.0;
                deltas.push((vname.clone(), bits, d));
                if d >= 3.0 {
                    has_three_point_gain = true;
                }
            }
            let memory = backend.memory_breakdown();
            rows.push(BenchRow {
                quantizer: vname,
                bits,
                recall,
                per_query_recall,
                query_us_p50: p50,
                code_bytes: memory.code_bytes,
                resident_bytes: memory.total_resident_bytes(),
                mse,
            });
        }
        append_paired_r10(&mut paired_r10, &rows[first_row..], bits, seed);
    }
    BenchReport {
        dataset: format!("{}-gold", name),
        dim,
        n_db: db.len(),
        n_queries: queries.len(),
        seed,
        rows,
        deltas,
        paired_r10,
        has_three_point_gain,
    }
}

// ── Streaming drift experiment ───────────────────────────────────────────────
//
// Following the CoDEQ (arXiv 2512.18335) taxonomy, a data-oblivious quantizer
// requires no centroid or codebook refit on insertion, whereas a data-dependent
// codec either retains its initial codebook or performs an O(N) rebuild. We stream
// a distribution-drifting insertion order (Big-ANN "clustered": cluster the corpus,
// feed one cluster at a time) and, at each checkpoint, measure pre-rerank recall@10
// over the *live* set against freshly recomputed exact gold.
//
// Codecs map onto CoDEQ's taxonomy:
//   - trellis / rabitq: data-oblivious and centroid-free; refit cost = 0.
//   - frozen_pq: codebook trained on the first cluster and then held fixed.
//   - rebuild_pq: codebook retrained on the full live set at every checkpoint.

pub struct StreamCheckpoint {
    pub frac: f64,
    pub n_live: usize,
    /// (codec, [recall@1,@10,@100]) over the live set at this checkpoint.
    pub recall: Vec<(String, [f64; 3])>,
    /// (codec, vectors-touched-by-refit at this checkpoint): 0 for oblivious +
    /// frozen, n_live for rebuild.
    pub refit_touched: Vec<(String, usize)>,
}

pub struct StreamReport {
    pub dataset: String,
    pub dim: usize,
    pub n_db: usize,
    pub n_queries: usize,
    pub n_groups: usize,
    pub bits: u8,
    pub seed: u64,
    pub checkpoints: Vec<StreamCheckpoint>,
}

/// Order db indices by cluster (kmeans into `n_groups`), so streaming them in
/// cluster order produces the Big-ANN "clustered" distribution drift.
fn cluster_order(dim: usize, db: &[Vec<f32>], n_groups: usize, seed: u64) -> Vec<Vec<usize>> {
    let centroids = kmeans(dim, db, n_groups, seed);
    let units: Vec<Vec<f32>> = db
        .iter()
        .map(|v| {
            let n = crate::l2_norm(v).max(f32::EPSILON);
            v.iter().map(|x| x / n).collect()
        })
        .collect();
    let mut groups = vec![Vec::new(); n_groups];
    for (i, u) in units.iter().enumerate() {
        let mut best = 0usize;
        let mut bd = f32::NEG_INFINITY;
        for (j, c) in centroids.iter().enumerate() {
            let d: f32 = u.iter().zip(c).map(|(a, b)| a * b).sum();
            if d > bd {
                bd = d;
                best = j;
            }
        }
        groups[best].push(i);
    }
    groups
}

/// Score one already-built backend over `queries` against `gold` (live-set ids).
fn recall_of(backend: &dyn VectorBackend, queries: &[Vec<f32>], gold: &[Vec<ItemId>]) -> [f64; 3] {
    let max_k = *KS.iter().max().unwrap();
    let (got, _) = run_backend(backend, queries, max_k);
    recall_against(gold, &got)
}

/// The streaming drift experiment. Streams `db` in cluster order over `n_groups`
/// checkpoints. `window`:
///   - `None` → **insert-only**: live set is the growing prefix.
///   - `Some(W)` → **delete+insert sliding window**: live set is the last W
///     clusters — the oldest cluster is evicted as each new one arrives, true churn.
/// At each checkpoint, gold is recomputed exactly over the live set and every codec
/// is scored on it. Codecs are rebuilt over the live set each checkpoint (the
/// measured cost is the *codebook-retraining* touch, not the re-encode): oblivious
/// (trellis/rabitq) and frozen_pq retrain NOTHING (refit 0 — their code is data-free
/// resp. frozen on cluster 0), rebuild_pq retrains on the live set (O(n_live)).
/// `frozen_pq`'s codebook is always trained once on cluster 0. `ULTRAVEC_TRELLIS_MEM`
/// honored. Builds codecs centroid-FREE so the oblivious zero-refit claim is structural.
pub fn stream_drift(
    name: &str,
    dim: usize,
    db: &[Vec<f32>],
    queries: &[Vec<f32>],
    bits: u8,
    n_groups: usize,
    seed: u64,
    window: Option<usize>,
) -> StreamReport {
    let groups = cluster_order(dim, db, n_groups, seed);
    let max_k = *KS.iter().max().unwrap();
    let do_pq = dim.is_multiple_of(4);
    // frozen_pq codebook: trained ONCE on the first cluster, never updated.
    let first: Vec<Vec<f32>> = groups[0].iter().map(|&i| db[i].clone()).collect();

    let mut checkpoints = Vec::new();

    for gi in 0..groups.len() {
        // Live cluster index range: [lo, gi]. Insert-only ⇒ lo=0; window ⇒ last W.
        let lo = match window {
            Some(w) => (gi + 1).saturating_sub(w),
            None => 0,
        };
        let live: Vec<Vec<f32>> = (lo..=gi)
            .flat_map(|g| groups[g].iter().map(|&i| db[i].clone()))
            .collect();

        // Exact gold over the live set, recomputed at this checkpoint.
        let gold: Vec<Vec<ItemId>> = queries
            .iter()
            .map(|q| exact_topk(q, &live, max_k))
            .collect();

        // Build each codec over the live set. Oblivious codecs re-encode (data-free
        // code → no retrain); frozen_pq re-encodes with its cluster-0 codebook;
        // rebuild_pq retrains. ItemId = index into `live`, so gold ids line up.
        let mut trellis = crate::trellis::TrellisQuantizer::new(dim, bits);
        let mut rabitq = crate::rabitq::RaBitQ::new(dim, bits);
        let mut frozen_pq = do_pq.then(|| {
            crate::pq::ProductQuantizer::with_codebook_from(dim, 2, crate::pq::PqLoss::Mse, &first)
        });
        for (id, v) in live.iter().enumerate() {
            trellis.add(id as ItemId, v);
            rabitq.add(id as ItemId, v);
            if let Some(pq) = frozen_pq.as_mut() {
                pq.push(id as ItemId, v);
            }
        }

        let mut recall = Vec::new();
        let mut refit = Vec::new();
        recall.push(("trellis".to_string(), recall_of(&trellis, queries, &gold)));
        refit.push(("trellis".to_string(), 0usize));
        recall.push(("rabitq".to_string(), recall_of(&rabitq, queries, &gold)));
        refit.push(("rabitq".to_string(), 0usize));
        if let Some(pq) = frozen_pq.as_ref() {
            recall.push(("frozen_pq".to_string(), recall_of(pq, queries, &gold)));
            refit.push(("frozen_pq".to_string(), 0usize));
        }
        if do_pq {
            let rebuilt = crate::pq::ProductQuantizer::train(dim, 2, crate::pq::PqLoss::Mse, &live);
            recall.push((
                "rebuild_pq".to_string(),
                recall_of(&rebuilt, queries, &gold),
            ));
            refit.push(("rebuild_pq".to_string(), live.len())); // O(n_live) retrain
        }

        checkpoints.push(StreamCheckpoint {
            frac: (gi + 1) as f64 / n_groups as f64,
            n_live: live.len(),
            recall,
            refit_touched: refit,
        });
    }

    StreamReport {
        dataset: name.to_string(),
        dim,
        n_db: db.len(),
        n_queries: queries.len(),
        n_groups,
        bits,
        seed,
        checkpoints,
    }
}

// ── Graph-ANN streaming experiment ───────────────────────────────────────────
//
// Does drift corrupt the graph STRUCTURE, not just query scores? Each codec gets an
// incrementally-built NSW graph whose nodes are the codec's *decoded* unit vectors —
// so edges are chosen by codec-fidelity distances at insert time. An oblivious codec
// uses a fixed data-free transform; a fitted codec uses its initial codebook. We
// report both the
// graph-search recall and the flat-exhaustive recall over the SAME decoded vectors:
// the gap is the graph-structure penalty, and its growth under drift is the figure.

pub struct GraphCheckpoint {
    pub frac: f64,
    pub n_live: usize,
    /// (codec, graph_recall@10, flat_recall@10) over the live set.
    pub rows: Vec<(String, f64, f64)>,
}

pub struct GraphStreamReport {
    pub dataset: String,
    pub dim: usize,
    pub n_queries: usize,
    pub n_groups: usize,
    pub bits: u8,
    pub seed: u64,
    pub m: usize,
    pub ef: usize,
    pub checkpoints: Vec<GraphCheckpoint>,
}

/// recall@10 of a set of retrieved ids vs the exact gold (top-10).
fn r10(gold: &[Vec<ItemId>], got: &[Vec<ItemId>]) -> f64 {
    let mut hit = 0usize;
    let mut tot = 0usize;
    for (g, a) in gold.iter().zip(got) {
        let gs: std::collections::HashSet<ItemId> = g.iter().take(10).copied().collect();
        hit += a.iter().take(10).filter(|i| gs.contains(i)).count();
        tot += gs.len().min(10);
    }
    if tot == 0 {
        0.0
    } else {
        hit as f64 / tot as f64
    }
}

/// Build a decoder closure for one codec name over `(dim, bits, train_set)`. Returns
/// `reconstruct_unit`-style decoded vectors. Oblivious codecs ignore `train`; frozen
/// trains its codebook on it; "exact" is the un-quantized ceiling.
fn decode_all(
    codec: &str,
    dim: usize,
    bits: u8,
    train: &[Vec<f32>],
    items: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    use crate::VectorBackend;
    let back: Option<Box<dyn VectorBackend + Sync>> = match codec {
        "trellis" => Some(Box::new(crate::trellis::TrellisQuantizer::new(dim, bits))),
        "rabitq" => Some(Box::new(crate::rabitq::RaBitQ::new(dim, bits))),
        "frozen_pq" => Some(Box::new(crate::pq::ProductQuantizer::with_codebook_from(
            dim,
            2,
            crate::pq::PqLoss::Mse,
            train,
        ))),
        "rebuild_pq" => Some(Box::new(crate::pq::ProductQuantizer::train(
            dim,
            2,
            crate::pq::PqLoss::Mse,
            train,
        ))),
        _ => None, // "exact"
    };
    match back {
        Some(b) => items
            .iter()
            .map(|v| b.reconstruct_unit(v).unwrap())
            .collect(),
        None => items
            .iter()
            .map(|v| {
                let n = crate::l2_norm(v).max(f32::EPSILON);
                v.iter().map(|x| x / n).collect()
            })
            .collect(),
    }
}

/// Graph-ANN streaming drift (insert-only). For each codec, decode the live set with
/// that codec, build an NSW over the decoded vectors, and at each checkpoint score
/// graph-search recall@10 AND flat-exhaustive recall@10 over the *decoded* set, both
/// against exact (un-quantized) gold. `frozen_pq` trains its codebook on cluster 0;
/// `rebuild_pq`/`exact` are re-decoded each checkpoint (so the graph is rebuilt — the
/// O(n_live) upper bound). Oblivious codecs decode identically regardless of order, so
/// their graph is the incremental one.
pub fn graph_stream_drift(
    name: &str,
    dim: usize,
    db: &[Vec<f32>],
    queries: &[Vec<f32>],
    bits: u8,
    n_groups: usize,
    seed: u64,
    m: usize,
    ef: usize,
) -> GraphStreamReport {
    let groups = cluster_order(dim, db, n_groups, seed);
    let do_pq = dim.is_multiple_of(4);
    let first: Vec<Vec<f32>> = groups[0].iter().map(|&i| db[i].clone()).collect();
    let codecs: Vec<&str> = {
        let mut c = vec!["trellis", "rabitq"];
        if do_pq {
            c.push("frozen_pq");
            c.push("rebuild_pq");
        }
        c.push("exact");
        c
    };
    // Per-codec persistent graph + how many live items it has already inserted.
    let mut graphs: Vec<crate::graph::NswGraph> = codecs
        .iter()
        .map(|_| crate::graph::NswGraph::new(m, ef))
        .collect();
    let mut inserted = vec![0usize; codecs.len()];

    let mut live: Vec<Vec<f32>> = Vec::new();
    let mut checkpoints = Vec::new();

    for gi in 0..groups.len() {
        for &idx in &groups[gi] {
            live.push(db[idx].clone());
        }
        let live_norms: Vec<f32> = live.par_iter().map(|v| crate::l2_norm(v)).collect();
        let gold: Vec<Vec<ItemId>> = queries
            .par_iter()
            .map(|q| exact_topk_with(q, &live, &live_norms, 10))
            .collect();

        let mut rows = Vec::new();
        for (ci, &codec) in codecs.iter().enumerate() {
            // Codecs whose decoding is order/refit-DEPENDENT must rebuild the graph
            // from the full live decode each checkpoint (frozen drifts but its codes
            // are append-stable, so it can grow incrementally too; rebuild_pq + exact
            // re-decode). Oblivious + frozen: grow incrementally.
            let rebuild = matches!(codec, "rebuild_pq");
            let decoded_full =
                decode_all(codec, dim, bits, first_or_live(codec, &first, &live), &live);
            if rebuild {
                let mut g = crate::graph::NswGraph::new(m, ef);
                for v in &decoded_full {
                    g.insert(v.clone());
                }
                graphs[ci] = g;
                inserted[ci] = live.len();
            } else {
                // append the newly-decoded tail incrementally
                for v in decoded_full.iter().skip(inserted[ci]) {
                    graphs[ci].insert(v.clone());
                }
                inserted[ci] = live.len();
            }
            // graph recall
            let got_graph: Vec<Vec<ItemId>> = queries
                .iter()
                .map(|q| {
                    let qn = crate::l2_norm(q).max(f32::EPSILON);
                    let qu: Vec<f32> = q.iter().map(|x| x / qn).collect();
                    graphs[ci]
                        .search(&qu, 10, ef)
                        .into_iter()
                        .map(|i| i as ItemId)
                        .collect()
                })
                .collect();
            let g_recall = r10(&gold, &got_graph);
            // flat recall over the SAME decoded set (exhaustive cosine)
            let got_flat: Vec<Vec<ItemId>> = queries
                .iter()
                .map(|q| exact_topk(q, &decoded_full, 10))
                .collect();
            let f_recall = r10(&gold, &got_flat);
            rows.push((codec.to_string(), g_recall, f_recall));
        }
        checkpoints.push(GraphCheckpoint {
            frac: (gi + 1) as f64 / n_groups as f64,
            n_live: live.len(),
            rows,
        });
    }

    GraphStreamReport {
        dataset: name.to_string(),
        dim,
        n_queries: queries.len(),
        n_groups,
        bits,
        seed,
        m,
        ef,
        checkpoints,
    }
}

/// frozen_pq trains on cluster 0; everyone else's "train" is the live set.
fn first_or_live<'a>(codec: &str, first: &'a [Vec<f32>], live: &'a [Vec<f32>]) -> &'a [Vec<f32>] {
    if codec == "frozen_pq" {
        first
    } else {
        live
    }
}

/// Markdown for a [`GraphStreamReport`]: graph-recall@10 + the graph-vs-flat penalty.
pub fn render_graph_stream_markdown(r: &GraphStreamReport) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "# Graph-ANN streaming under distribution drift — {}\n\n",
        r.dataset
    ));
    s.push_str(&format!(
        "- dim **{}**, queries **{}**, {} clusters (drift order), {}-bit, NSW M={} ef={}, seed {}\n",
        r.dim, r.n_queries, r.n_groups, r.bits, r.m, r.ef, r.seed
    ));
    s.push_str("- graph recall@10 (NSW search) vs flat recall@10 (exhaustive over the SAME decoded vectors); both vs exact un-quantized gold. The gap = graph-structure penalty.\n\n");
    let codecs: Vec<String> = r
        .checkpoints
        .first()
        .map(|c| c.rows.iter().map(|(n, _, _)| n.clone()).collect())
        .unwrap_or_default();

    s.push_str("## graph recall@10 vs stream fraction\n\n| frac | n_live |");
    for c in &codecs {
        s.push_str(&format!(" {} |", c));
    }
    s.push_str("\n|---|---|");
    for _ in &codecs {
        s.push_str("---|");
    }
    s.push('\n');
    for cp in &r.checkpoints {
        s.push_str(&format!("| {:.2} | {} |", cp.frac, cp.n_live));
        for c in &codecs {
            let v = cp
                .rows
                .iter()
                .find(|(n, _, _)| n == c)
                .map(|(_, g, _)| *g)
                .unwrap_or(0.0);
            s.push_str(&format!(" {:.3} |", v));
        }
        s.push('\n');
    }

    s.push_str(
        "\n## flat recall@10 (exhaustive over decoded — fair per-codec)\n\n| frac | n_live |",
    );
    for c in &codecs {
        s.push_str(&format!(" {} |", c));
    }
    s.push_str("\n|---|---|");
    for _ in &codecs {
        s.push_str("---|");
    }
    s.push('\n');
    for cp in &r.checkpoints {
        s.push_str(&format!("| {:.2} | {} |", cp.frac, cp.n_live));
        for c in &codecs {
            let v = cp
                .rows
                .iter()
                .find(|(n, _, _)| n == c)
                .map(|(_, _, f)| *f)
                .unwrap_or(0.0);
            s.push_str(&format!(" {:.3} |", v));
        }
        s.push('\n');
    }

    s.push_str("\n## graph−flat penalty (recall lost to graph structure)\n\n| frac |");
    for c in &codecs {
        s.push_str(&format!(" {} |", c));
    }
    s.push_str("\n|---|");
    for _ in &codecs {
        s.push_str("---|");
    }
    s.push('\n');
    for cp in &r.checkpoints {
        s.push_str(&format!("| {:.2} |", cp.frac));
        for c in &codecs {
            let (g, f) = cp
                .rows
                .iter()
                .find(|(n, _, _)| n == c)
                .map(|(_, g, f)| (*g, *f))
                .unwrap_or((0.0, 0.0));
            s.push_str(&format!(" {:+.3} |", g - f));
        }
        s.push('\n');
    }
    s
}

// ── Cold-start experiment ────────────────────────────────────────────────────
//
// The structural oblivious win: a data-dependent codec (PQ) needs calibration data to
// build its codebook; below 256 samples per subspace, k-means cannot initialize all
// 256 centroids. The oblivious codecs require no calibration. We sweep PQ's
// calibration-set size K using a seeded representative subsample and score every
// codec by reconstruction-cosine. Oblivious recall is K-independent.

pub struct ColdStartReport {
    pub dataset: String,
    pub dim: usize,
    pub n_db: usize,
    pub n_queries: usize,
    pub bits: u8,
    pub seed: u64,
    pub trellis_r10: f64, // oblivious, calibration-independent
    pub rabitq_r10: f64,
    /// (calibration_size, pq_r10) — 0 ⇒ full corpus (warm).
    pub pq: Vec<(usize, f64)>,
}

/// Seeded representative random subsample of `db` of size `k` (Fisher-Yates prefix).
fn subsample(db: &[Vec<f32>], k: usize, seed: u64) -> Vec<Vec<f32>> {
    let n = db.len();
    let k = k.min(n);
    let mut order: Vec<usize> = (0..n).collect();
    let mut st = seed;
    for i in (1..n).rev() {
        let j = (crate::next_f64(&mut st) * (i as f64 + 1.0)) as usize;
        order.swap(i, j.min(i));
    }
    order[..k].iter().map(|&i| db[i].clone()).collect()
}

/// Cold-start sweep: PQ trained on K representative calibration vectors, encoding the
/// FULL corpus, scored fairly (reconstruction-cosine) vs exact gold; across calib
/// sizes. Oblivious trellis/rabitq computed once (calibration-free).
pub fn cold_start_sweep(
    name: &str,
    dim: usize,
    db: &[Vec<f32>],
    queries: &[Vec<f32>],
    bits: u8,
    calib_sizes: &[usize],
    seed: u64,
) -> ColdStartReport {
    let gold_norms: Vec<f32> = db.par_iter().map(|v| crate::l2_norm(v)).collect();
    let gold: Vec<Vec<ItemId>> = queries
        .par_iter()
        .map(|q| exact_topk_with(q, db, &gold_norms, 10))
        .collect();
    // Fair recall = exhaustive cosine over each codec's reconstruction.
    let recall_of_decoded = |decoded: &[Vec<f32>]| -> f64 {
        let got: Vec<Vec<ItemId>> = queries.iter().map(|q| exact_topk(q, decoded, 10)).collect();
        r10(&gold, &got)
    };
    // Oblivious codecs — calibration-free, computed once.
    let trellis_r10 = recall_of_decoded(&decode_all("trellis", dim, bits, &[], db));
    let rabitq_r10 = recall_of_decoded(&decode_all("rabitq", dim, bits, &[], db));
    // PQ at each calibration size (0 ⇒ full corpus).
    let do_pq = dim.is_multiple_of(4);
    let mut pq = Vec::new();
    if do_pq {
        for &k in calib_sizes {
            let calib = if k == 0 {
                db.to_vec()
            } else {
                subsample(db, k, seed)
            };
            // For K<256 there are fewer calibration points than centroids per
            // subspace, so the with-replacement initialization necessarily repeats.
            let decoded = decode_all("rebuild_pq", dim, bits, &calib, db);
            pq.push((k, recall_of_decoded(&decoded)));
        }
    }
    ColdStartReport {
        dataset: name.to_string(),
        dim,
        n_db: db.len(),
        n_queries: queries.len(),
        bits,
        seed,
        trellis_r10,
        rabitq_r10,
        pq,
    }
}

/// Markdown for a [`ColdStartReport`]: PQ recall vs calibration size, with the
/// calibration-free oblivious lines and the crossover.
pub fn render_cold_start_markdown(r: &ColdStartReport) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "# Cold-start calibration sweep — {}\n\n",
        r.dataset
    ));
    s.push_str(&format!(
        "- dim **{}**, db **{}**, queries **{}**, {}-bit, seed {}; fair reconstruction-cosine recall@10 over the full corpus\n",
        r.dim, r.n_db, r.n_queries, r.bits, r.seed
    ));
    s.push_str(&format!(
        "- **oblivious (calibration-free): trellis {:.3} · rabitq {:.3}** — flat, independent of calibration size\n\n",
        r.trellis_r10, r.rabitq_r10
    ));
    s.push_str("## PQ recall@10 vs calibration-set size K\n\n");
    s.push_str("| K (PQ calib) | pq R@10 | vs trellis | note |\n|---|---|---|---|\n");
    for &(k, v) in &r.pq {
        let klabel = if k == 0 {
            "full".to_string()
        } else {
            k.to_string()
        };
        let delta = v - r.trellis_r10;
        let note = if k != 0 && k < 256 {
            "K<256: fewer calibration vectors than centroids/subspace"
        } else if delta >= 0.0 {
            "PQ ahead (warm)"
        } else {
            "oblivious ahead (cold)"
        };
        s.push_str(&format!(
            "| {} | {:.3} | {:+.3} | {} |\n",
            klabel, v, delta, note
        ));
    }
    s
}

/// Markdown for a [`StreamReport`]: recall@10-vs-stream-fraction (the hero curve) +
/// the refit-cost column, then per-codec recall@{1,10,100} at the final checkpoint.
pub fn render_stream_markdown(r: &StreamReport) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "# Streaming drift: recall under churn vs refit cost — {}\n\n",
        r.dataset
    ));
    s.push_str(&format!(
        "- dim **{}**, db **{}**, queries **{}**, {} clusters (drift order), {}-bit, seed {}\n",
        r.dim, r.n_db, r.n_queries, r.n_groups, r.bits, r.seed
    ));
    s.push_str(
        "- pre-rerank recall@10 over the **live** set; exact gold recomputed each checkpoint\n",
    );
    s.push_str("- trellis/rabitq: data-oblivious, **centroid-free** (refit cost 0). frozen_pq: codebook frozen on cluster 0. rebuild_pq: codebook retrained on the full live set each checkpoint (O(n_live)).\n\n");

    // Codec column order from the first checkpoint.
    let codecs: Vec<String> = r
        .checkpoints
        .first()
        .map(|c| c.recall.iter().map(|(n, _)| n.clone()).collect())
        .unwrap_or_default();

    s.push_str("## recall@10 vs stream fraction (the hero curve)\n\n");
    s.push_str("| frac | n_live |");
    for c in &codecs {
        s.push_str(&format!(" {} |", c));
    }
    s.push('\n');
    s.push_str("|---|---|");
    for _ in &codecs {
        s.push_str("---|");
    }
    s.push('\n');
    for cp in &r.checkpoints {
        s.push_str(&format!("| {:.2} | {} |", cp.frac, cp.n_live));
        for c in &codecs {
            let v = cp
                .recall
                .iter()
                .find(|(n, _)| n == c)
                .map(|(_, r)| r[1])
                .unwrap_or(0.0);
            s.push_str(&format!(" {:.3} |", v));
        }
        s.push('\n');
    }

    s.push_str("\n## refit cost (vectors touched per checkpoint)\n\n");
    s.push_str("| frac |");
    for c in &codecs {
        s.push_str(&format!(" {} |", c));
    }
    s.push('\n');
    s.push_str("|---|");
    for _ in &codecs {
        s.push_str("---|");
    }
    s.push('\n');
    for cp in &r.checkpoints {
        s.push_str(&format!("| {:.2} |", cp.frac));
        for c in &codecs {
            let t = cp
                .refit_touched
                .iter()
                .find(|(n, _)| n == c)
                .map(|(_, t)| *t)
                .unwrap_or(0);
            s.push_str(&format!(" {} |", t));
        }
        s.push('\n');
    }

    if let Some(last) = r.checkpoints.last() {
        s.push_str("\n## final checkpoint — recall@{1,10,100}\n\n");
        s.push_str("| codec | R@1 | R@10 | R@100 |\n|---|---|---|---|\n");
        for (n, rec) in &last.recall {
            s.push_str(&format!(
                "| {} | {:.3} | {:.3} | {:.3} |\n",
                n, rec[0], rec[1], rec[2]
            ));
        }
    }
    s
}

// ---- FastScan companion Pareto (lever A) ------------------------------------
pub struct CompanionPoint {
    pub label: String,
    pub shortlist_c: usize,
    pub recall: [f64; 3],
    pub cand_per_q: usize,
    pub qps: f64,
    pub bytes_per_vec: usize,
}

pub struct CompanionReport {
    pub dataset: String,
    pub dim: usize,
    pub n_db: usize,
    pub n_queries: usize,
    pub bits: u8,
    pub seed: u64,
    pub points: Vec<CompanionPoint>,
}

/// Best-of-`reps` rayon QPS for scoring `queries` with `f` (best-of implies warmup).
fn timed_qps<F>(queries: &[Vec<f32>], reps: usize, f: F) -> f64
where
    F: Fn(&[f32]) -> Vec<ItemId> + Sync,
{
    use rayon::prelude::*;
    let mut best = 0.0f64;
    for _ in 0..reps.max(1) {
        let t = Instant::now();
        let _got: Vec<Vec<ItemId>> = queries.par_iter().map(|q| f(q)).collect();
        let qps = queries.len() as f64 / t.elapsed().as_secs_f64().max(1e-9);
        if qps > best {
            best = qps;
        }
    }
    best
}

/// FastScan companion recall-vs-shortlist-size sweep (the mid-band QPS lever). Builds a
/// codes-only trellis + 1-bit sign shortlist ONCE; for each C reports pre-rerank
/// recall@{1,10,100}, the candidate count C (= trellis-decode count, the QPS proxy),
/// measured QPS, and the full byte budget (incl. the +⌈D/8⌉B sign code). Reference rows:
/// C=N (full codes-only scan), the RaBitQ-1bit incumbent, and a byte-matched plain
/// trellis (full scan) so the byte-matched comparison is explicit. Centering
/// OFF (the uncentered query sign vs residual data sign misalign).
pub fn companion_pareto(
    name: &str,
    dim: usize,
    db: &[Vec<f32>],
    queries: &[Vec<f32>],
    bits: u8,
    shortlist_cs: &[usize],
    seed: u64,
    reps: usize,
) -> CompanionReport {
    use rayon::prelude::*;
    let n = db.len();
    let gold: Vec<Vec<ItemId>> = queries.par_iter().map(|q| exact_topk(q, db, 100)).collect();

    let mut comp = TrellisQuantizer::new(dim, bits)
        .with_codes_only(true)
        .with_shortlist(1);
    comp.add_batch(db);
    let comp_bytes = comp.mem_bytes() / n.max(1);

    let mut points = Vec::new();
    let mut cs: Vec<usize> = shortlist_cs.to_vec();
    cs.push(n);
    cs.sort_unstable();
    cs.dedup();
    for &c in &cs {
        let got: Vec<Vec<ItemId>> = queries
            .par_iter()
            .map(|q| {
                comp.search_shortlist(q, 100, c)
                    .into_iter()
                    .map(|(id, _)| id)
                    .collect()
            })
            .collect();
        let recall = recall_against(&gold, &got);
        let qps = timed_qps(queries, reps, |q| {
            comp.search_shortlist(q, 100, c)
                .into_iter()
                .map(|(id, _)| id)
                .collect()
        });
        points.push(CompanionPoint {
            label: if c >= n {
                "C=N (full codes-only scan)".into()
            } else {
                format!("C={c}")
            },
            shortlist_c: c,
            recall,
            cand_per_q: c.min(n),
            qps,
            bytes_per_vec: comp_bytes,
        });
    }

    // Reference: RaBitQ 1-bit incumbent (the cheap mid-band winner in tab:qps).
    {
        let mut rb = RaBitQ::new(dim, 1);
        rb.add_batch(db);
        let (got, _) = run_backend(&rb, queries, 100);
        let recall = recall_against(&gold, &got);
        let qps = timed_qps(queries, reps, |q| {
            rb.search(q, 100).into_iter().map(|(id, _)| id).collect()
        });
        points.push(CompanionPoint {
            label: "rabitq-1bit (incumbent)".into(),
            shortlist_c: 0,
            recall,
            cand_per_q: n,
            qps,
            bytes_per_vec: rb.mem_bytes() / n.max(1),
        });
    }

    // Reference: byte-matched plain trellis codes-only (bits+1 adds approximately
    // ⌈D/8⌉ bytes, matching the sign-code sidecar), with a full scan.
    {
        let bm = bits + 1;
        let mut tr = TrellisQuantizer::new(dim, bm).with_codes_only(true);
        tr.add_batch(db);
        let (got, _) = run_backend(&tr, queries, 100);
        let recall = recall_against(&gold, &got);
        let qps = timed_qps(queries, reps, |q| {
            tr.search(q, 100).into_iter().map(|(id, _)| id).collect()
        });
        points.push(CompanionPoint {
            label: format!("trellis-codes @{bm}b (bytematch, full scan)"),
            shortlist_c: 0,
            recall,
            cand_per_q: n,
            qps,
            bytes_per_vec: tr.mem_bytes() / n.max(1),
        });
    }

    CompanionReport {
        dataset: name.to_string(),
        dim,
        n_db: n,
        n_queries: queries.len(),
        bits,
        seed,
        points,
    }
}

pub fn render_companion_markdown(r: &CompanionReport) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    let _ = writeln!(s, "# FastScan companion Pareto — {}", r.dataset);
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "- dim **{}**, db **{}**, queries **{}**, base bits **{}**, seed **{}**",
        r.dim, r.n_db, r.n_queries, r.bits, r.seed
    );
    let _ = writeln!(s, "- pre-rerank recall vs exact-cosine gold; `cand/q` = trellis decodes per query (the QPS proxy, C vs N={}); QPS = best-of-reps rayon throughput.", r.n_db);
    let _ = writeln!(
        s,
        "- 1-bit sign-code shortlist → trellis-rescaled rerank of the top-C. Centering OFF."
    );
    let _ = writeln!(s);
    let _ = writeln!(s, "| config | R@1 | R@10 | R@100 | cand/q | QPS | B/vec |");
    let _ = writeln!(s, "|---|---|---|---|---|---|---|");
    for p in &r.points {
        let _ = writeln!(
            s,
            "| {} | {:.3} | {:.3} | {:.3} | {} | {:.0} | {} |",
            p.label, p.recall[0], p.recall[1], p.recall[2], p.cand_per_q, p.qps, p.bytes_per_vec
        );
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::next_f64;

    #[test]
    fn recall_metric_basic() {
        let gold = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let got = vec![vec![1, 2, 9], vec![4, 5, 6]];
        let per_query = recall_per_query(&gold, &got);
        let r = recall_against(&gold, &got);
        // k=1: both top-1 correct → 1.0. (R@1 index 0.)
        assert!((r[0] - 1.0).abs() < 1e-9);
        assert_eq!(mean_recall(&per_query), r);
        assert!((per_query[0][1] - 2.0 / 3.0).abs() < 1e-9);
        assert!((per_query[1][1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn paired_bootstrap_is_deterministic_and_preserves_fixed_difference() {
        let ultravec = vec![[1.0, 0.8, 0.9]; 20];
        let comparator = vec![[1.0, 0.7, 0.9]; 20];
        let first = paired_bootstrap_r10(&ultravec, &comparator, 42);
        let second = paired_bootstrap_r10(&ultravec, &comparator, 42);
        assert_eq!(first, second);
        assert!((first.0 - 10.0).abs() < 1e-9);
        assert!((first.1 - 10.0).abs() < 1e-9);
        assert!((first.2 - 10.0).abs() < 1e-9);
    }

    #[test]
    fn fixed_codec_parallel_batches_preserve_serial_results() {
        let vectors: Vec<Vec<f32>> = (0..32)
            .map(|row| {
                (0..8)
                    .map(|column| ((row * 17 + column * 13) % 97) as f32 / 97.0)
                    .collect()
            })
            .collect();
        let query = &vectors[7];
        let pairs: Vec<(Box<dyn VectorBackend + Sync>, Box<dyn VectorBackend + Sync>)> = vec![
            (
                Box::new(TurboQuantBaseline::new(8, 1)),
                Box::new(TurboQuantBaseline::new(8, 1)),
            ),
            (Box::new(RaBitQ::new(8, 1)), Box::new(RaBitQ::new(8, 1))),
            (
                Box::new(TrellisQuantizer::new(8, 1)),
                Box::new(TrellisQuantizer::new(8, 1)),
            ),
            (
                Box::new(PvqQuantizer::new(8, 1)),
                Box::new(PvqQuantizer::new(8, 1)),
            ),
            (
                Box::new(EdenQuantizer::new(8, 1)),
                Box::new(EdenQuantizer::new(8, 1)),
            ),
            (
                Box::new(BlockQuantizer::new(8, 1)),
                Box::new(BlockQuantizer::new(8, 1)),
            ),
            (
                Box::new(E8Quantizer::new(8, 1)),
                Box::new(E8Quantizer::new(8, 1)),
            ),
        ];

        for (mut serial, mut parallel) in pairs {
            for (id, vector) in vectors.iter().enumerate() {
                serial.add(id as ItemId, vector);
            }
            parallel.add_batch(&vectors);
            assert_eq!(serial.search(query, 10), parallel.search(query, 10));
        }
    }

    #[test]
    fn mips_memory_includes_exact_norm_sidecar() {
        let memory = crate::MemoryBreakdown {
            code_bytes: 10,
            model_bytes: 20,
            index_bytes: 30,
            cache_bytes: 40,
        };
        assert_eq!(with_mips_norm_sidecar(&memory, 3, false), (22, 112));
        assert_eq!(with_mips_norm_sidecar(&memory, 3, true), (10, 100));
    }

    #[test]
    fn benchmark_runs_end_to_end_on_synthetic() {
        // Small smoke: synthetic Gaussian unit vectors, just confirm the harness
        // produces sane recall numbers (baseline should be > 0).
        let dim = 64;
        let mut state = 5u64;
        let normal = |s: &mut u64| -> f32 {
            let u1 = next_f64(s).max(1e-12);
            let u2 = next_f64(s);
            ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
        };
        let vectors: Vec<Vec<f32>> = (0..300)
            .map(|_| (0..dim).map(|_| normal(&mut state)).collect())
            .collect();
        let ds = Dataset {
            name: "synth".into(),
            dim,
            vectors,
        };
        let r = benchmark(&ds, &[5], 30, 42, VariantSet::Full);
        assert!(!r.rows.is_empty());
        let base = r
            .rows
            .iter()
            .find(|x| x.quantizer == "turboquant_baseline")
            .unwrap();
        assert!(
            base.recall[1] > 0.3,
            "baseline R@10 {} implausibly low",
            base.recall[1]
        );
        // Markdown renders without panicking.
        assert!(render_markdown(&r).contains("≥3pp recall@10 gain"));
    }
}
