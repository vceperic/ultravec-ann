//! IVF (inverted-file) index for database-systems integration. The flat-scan
//! benchmarks ([`crate::bench`]) measure *recall per candidate*; this layer
//! measures *recall vs QPS inside an ANN index*: whether the
//! trellis's higher recall-per-candidate (so fewer candidates scanned for a target
//! recall) win the recall-vs-QPS Pareto despite its per-candidate scoring cost?
//!
//! Structure (textbook IVF / IVFADC coarse stage):
//!   - **Coarse quantizer**: `nlist` k-means centroids over the unit-normalized base
//!     vectors — the SAME [`crate::bench::kmeans`] the per-cluster-centering bench
//!     uses (one coarse quantizer, two consumers).
//!   - **Posting lists**: each base vector is assigned to its nearest centroid
//!     (by cosine = dot on unit vectors); each list is its OWN codec backend holding
//!     only that cluster's vectors at their global ids. This reuses each codec's
//!     `VectorBackend::{add,search}` UNCHANGED — no per-candidate scoring API, no
//!     touching `baseline.rs` — and candidate restriction is automatic: only probed
//!     clusters are scored.
//!   - **Query**: rank all `nlist` centroids by exact cosine (the small coarse set),
//!     take the `nprobe` nearest, score those clusters' backends, merge → top-k.
//!
//! QPS is wall-clock over the query set at a stated thread count. We parallelize
//! ACROSS queries (each query scores its probed clusters serially) so the per-query
//! latency is realistic and the aggregate QPS uses all cores — exactly how a server
//! fans out independent queries.
//!
//! The trellis is measured in both deployment modes via its `codes_only` knob:
//!   - **recon** (mode a): decode-free dot vs the fp32 recon cache — fast, fp32 RAM.
//!   - **codes** (mode b): per-candidate O(D) Viterbi-state decode — RAM-compressed.
//! Both are reported and labelled with their storage regime.

use std::time::Instant;

use rayon::prelude::*;

use crate::{
    baseline::TurboQuantBaseline, cosine, l2_norm, pvq::PvqQuantizer, rabitq::RaBitQ,
    trellis::TrellisQuantizer, ItemId, MemoryBreakdown, VectorBackend,
};

/// Which codec scores the posting lists and, for the trellis, which storage mode.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum IvfCodec {
    /// TurboQuant reference (N(0,1), codes-only—its only mode).
    Baseline,
    /// RaBitQ / Extended-RaBitQ (codes-only — its only mode).
    Rabitq,
    /// Trellis, recon-cache mode (a): decode-free dot, fp32 RAM (not compressed).
    TrellisRecon,
    /// Trellis, codes-only mode (b): per-candidate decode, RAM-compressed.
    TrellisCodes,
    /// PVQ decode-free spherical code (codes-only — its only mode).
    Pvq,
    /// Trellis, codes-only, with the 1-bit companion shortlist enabled: a
    /// SIMD-popcount prefilter over sign codes selects `C` candidates per list and
    /// only those are decoded. Same two-stage shape as the FastScan line of work,
    /// and the same packer and bit order as canonical RaBitQ 1-bit -- it was
    /// implemented and tested in this crate but wired into no experiment, so the
    /// codec was evaluated without the technique its closest comparator's family is
    /// known for. The variant carries `C` so the shortlist size is a property of the
    /// arm rather than of the launching shell -- an ambient knob here selected the
    /// prefilter on arms that were meant to be exhaustive, and made a whole IVF run
    /// report three identical trellis curves.
    TrellisShortlist { c: usize, asym: bool, rnorm: bool },
}

impl IvfCodec {
    pub fn label(self) -> String {
        match self {
            IvfCodec::Baseline => "turboquant_baseline".to_string(),
            IvfCodec::Rabitq => "rabitq".to_string(),
            IvfCodec::TrellisRecon => "trellis_recon".to_string(),
            IvfCodec::TrellisCodes => "trellis_codes".to_string(),
            IvfCodec::Pvq => "pvq".to_string(),
            // The size is in the label because a shortlist row means nothing without
            // it: C is the entire recall/throughput trade this arm exists to show.
            IvfCodec::TrellisShortlist { c, .. } => format!("trellis_shortlist_c{c}"),
        }
    }
    /// Storage and scoring regime represented by this curve.
    pub fn regime(self) -> &'static str {
        match self {
            IvfCodec::TrellisRecon => "fp32 recon cache (NOT RAM-compressed)",
            IvfCodec::TrellisShortlist { .. } => "codes-only + 1-bit shortlist (RAM-compressed)",
            _ => "codes-only (RAM-compressed)",
        }
    }
    /// Construct an empty codec backend for one posting list.
    ///
    /// `centroid` is that list's coarse centroid. With `ULTRAVEC_IVF_CENTER=1` each
    /// codec that supports it quantizes the residual to this centroid and keeps the
    /// `<q,c>` term exact -- the step RaBitQ's own construction is published with,
    /// and which this harness previously gave to no codec at all even though the
    /// centroids were already computed for the coarse stage.
    ///
    /// Worth noting for the calibration story: inside an IVF index the centroids are
    /// fitted state the index already holds, so centering on them adds no assumption
    /// that the index has not already made. That is not true of the flat comparison,
    /// where centering means introducing corpus state that was not otherwise there.
    ///
    /// The frozen TurboQuant control has no centering path and does not get one --
    /// R1 keeps `baseline.rs` a verbatim copy -- so it stays uncentered and the
    /// asymmetry is disclosed rather than hidden.
    fn make(self, dim: usize, bits: u8, centroid: &[f32]) -> Box<dyn VectorBackend + Sync> {
        let center = std::env::var("ULTRAVEC_IVF_CENTER").as_deref() == Ok("1");
        let list_centroids = || vec![centroid.to_vec()];
        match self {
            IvfCodec::Baseline => Box::new(TurboQuantBaseline::new(dim, bits)),
            IvfCodec::Rabitq => {
                let q = RaBitQ::new(dim, bits);
                Box::new(if center {
                    q.with_centroids(list_centroids())
                } else {
                    q
                })
            }
            IvfCodec::TrellisRecon | IvfCodec::TrellisCodes | IvfCodec::TrellisShortlist { .. } => {
                let mut q = TrellisQuantizer::new(dim, bits)
                    .with_codes_only(!matches!(self, IvfCodec::TrellisRecon));
                // Set the shortlist explicitly on both arms rather than letting the
                // constructor's ambient `ULTRAVEC_TRELLIS_SHORTLIST` decide. Reading
                // it there means exporting the variable to select this arm silently
                // enables the prefilter on the exhaustive arms too, which makes every
                // trellis row the same experiment -- an error that produced a
                // full-scale IVF run where all three trellis curves were identical.
                q = q.with_shortlist(match self {
                    IvfCodec::TrellisShortlist { c, .. } => c,
                    _ => 0,
                });
                // Set the stage-1 scorer explicitly for the same reason the shortlist
                // size is set explicitly: `ULTRAVEC_TRELLIS_SHORTLIST_ASYM` is read in
                // the constructor and reaches every arm, so leaving it ambient would
                // let one exported variable silently redefine the exhaustive rows too.
                q = q.with_shortlist_asym(matches!(
                    self,
                    IvfCodec::TrellisShortlist { asym: true, .. }
                ));
                q = q.with_shortlist_rnorm(!matches!(
                    self,
                    IvfCodec::TrellisShortlist { rnorm: false, .. }
                ));
                Box::new(if center {
                    q.with_centroids(list_centroids())
                } else {
                    q
                })
            }
            IvfCodec::Pvq => {
                let q = PvqQuantizer::new(dim, bits);
                Box::new(if center {
                    q.with_centroids(list_centroids())
                } else {
                    q
                })
            }
        }
    }
}

/// An IVF index over one codec at one bit-width: coarse centroids + per-cluster
/// codec backends (the posting lists, holding global ids).
pub struct IvfIndex {
    centroids: Vec<Vec<f32>>, // nlist coarse centroids (unit-space means)
    posts: Vec<Box<dyn VectorBackend + Sync>>, // one codec backend per centroid
    // Kept for debug/inspection of a built index (not read on the hot path).
    #[allow(dead_code)]
    dim: usize,
    #[allow(dead_code)]
    codec: IvfCodec,
    #[allow(dead_code)]
    bits: u8,
}

impl IvfIndex {
    /// Build: k-means → assign → encode each posting list with `codec`. `centroids`
    /// is passed in so all codecs at a given `(nlist, seed)` share ONE coarse
    /// quantizer (the coarse stage is codec-independent; only scoring differs).
    pub fn build(
        dim: usize,
        db: &[Vec<f32>],
        centroids: &[Vec<f32>],
        assign: &[u32],
        codec: IvfCodec,
        bits: u8,
    ) -> Self {
        let nlist = centroids.len();
        // Bucket global ids by assigned centroid.
        let mut buckets: Vec<Vec<ItemId>> = vec![Vec::new(); nlist];
        for (i, &c) in assign.iter().enumerate() {
            buckets[c as usize].push(i as ItemId);
        }
        // Encode each posting list in parallel (the trellis Viterbi dominates build).
        let posts: Vec<Box<dyn VectorBackend + Sync>> = buckets
            .par_iter()
            .enumerate()
            .map(|(list, ids)| {
                let mut b = codec.make(dim, bits, &centroids[list]);
                b.reserve(ids.len());
                for &id in ids {
                    b.add(id, &db[id as usize]);
                }
                b
            })
            .collect();
        Self {
            dim,
            centroids: centroids.to_vec(),
            posts,
            codec,
            bits,
        }
    }

    /// Rank the `nlist` centroids by exact cosine to `query`, return the `nprobe`
    /// nearest centroid indices (the coarse stage — exact, over the small set).
    fn probe(&self, query: &[f32], nprobe: usize) -> Vec<usize> {
        let mut scored: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(j, c)| (j, cosine(query, c)))
            .collect();
        let np = nprobe.min(scored.len());
        let pivot = np.saturating_sub(1).min(scored.len() - 1);
        scored.select_nth_unstable_by(pivot, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(np);
        scored.into_iter().map(|(j, _)| j).collect()
    }

    /// Search: probe `nprobe` clusters, score those posting-list backends, merge →
    /// top-`k`. Each probed list returns its own top-`k`; merging the per-list
    /// top-`k` is exact for the global top-`k` (a global top-k member is top-k in its
    /// own list). Single-query, single-threaded (the QPS harness fans queries out).
    pub fn search(&self, query: &[f32], k: usize, nprobe: usize) -> Vec<(ItemId, f32)> {
        let lists = self.probe(query, nprobe);
        let mut merged: Vec<(ItemId, f32)> = Vec::new();
        for j in lists {
            // Each posting backend scores only ITS vectors — candidate restriction.
            merged.extend(self.posts[j].search(query, k));
        }
        merged.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        merged.truncate(k);
        merged
    }

    /// Search WITH an exact-fp32 rerank stage — the deployed ANN protocol. Retrieve the top-`rdepth`
    /// candidates by the codec's approximate score (probe `nprobe` lists), then re-score those
    /// `rdepth` candidates by EXACT cosine against the fp32 `db` and return the true top-`k`. The
    /// index RAM stays compressed (codes only); the fp32 rerank reads a separate tier (the
    /// Faiss-IVFPQ+refine pattern). A higher-fidelity codec's better approximate ranking puts more of
    /// the true top-k inside the `rdepth` candidate set, so it reaches a target recall at a SMALLER
    /// `rdepth` — fewer exact dots — hence higher QPS at fixed recall.
    pub fn search_with_rerank(
        &self,
        query: &[f32],
        k: usize,
        nprobe: usize,
        rdepth: usize,
        db: &[Vec<f32>],
    ) -> Vec<(ItemId, f32)> {
        let r = rdepth.max(k);
        let lists = self.probe(query, nprobe);
        let mut cand: Vec<(ItemId, f32)> = Vec::new();
        for j in lists {
            cand.extend(self.posts[j].search(query, r)); // each list's approx top-r
        }
        // global top-r by approximate score (a global top-r member is top-r in its own list).
        cand.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        cand.truncate(r);
        // exact-fp32 rerank of the r candidates.
        let mut reranked: Vec<(ItemId, f32)> = cand
            .iter()
            .map(|&(id, _)| (id, cosine(query, &db[id as usize])))
            .collect();
        reranked
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        reranked.truncate(k);
        reranked
    }

    /// Resident bytes across all posting lists (the codec's reported compressed
    /// footprint; for trellis-recon this EXCLUDES the fp32 recon cache, matching
    /// `mem_bytes`'s contract — see the regime label).
    pub fn mem_bytes(&self) -> usize {
        self.posts.iter().map(|p| p.mem_bytes()).sum()
    }

    pub fn memory_breakdown(&self) -> MemoryBreakdown {
        let mut memory = MemoryBreakdown {
            model_bytes: self.centroids.capacity() * std::mem::size_of::<Vec<f32>>()
                + self
                    .centroids
                    .iter()
                    .map(|centroid| centroid.capacity() * std::mem::size_of::<f32>())
                    .sum::<usize>(),
            index_bytes: self.posts.capacity()
                * std::mem::size_of::<Box<dyn VectorBackend + Sync>>(),
            ..MemoryBreakdown::default()
        };
        for post in &self.posts {
            let post_memory = post.memory_breakdown();
            memory.code_bytes = memory.code_bytes.saturating_add(post_memory.code_bytes);
            memory.model_bytes = memory.model_bytes.saturating_add(post_memory.model_bytes);
            memory.index_bytes = memory.index_bytes.saturating_add(post_memory.index_bytes);
            memory.cache_bytes = memory.cache_bytes.saturating_add(post_memory.cache_bytes);
        }
        memory
    }
}

// ── recall@k + ground truth (consistent with the flat-scan bench: cosine NN) ──

/// Exact cosine top-`k` ids for one query over `db` (the IVF ground truth — same
/// protocol as [`crate::bench`], so recall is "fraction of the exact-cosine top-k
/// the IVF returns", directly comparable to the flat-scan numbers). Computed
/// in-crate over the same database indexed by IVF, rather than a Texmex `.ivecs` file,
/// which is L2-over-the-full-1M and would mismatch a cosine/`--max` run.
pub(crate) fn exact_topk(query: &[f32], db: &[Vec<f32>], k: usize) -> Vec<ItemId> {
    let mut scored: Vec<(ItemId, f32)> = db
        .iter()
        .enumerate()
        .map(|(i, v)| (i as ItemId, cosine(query, v)))
        .collect();
    let kk = k.min(scored.len());
    if kk == 0 {
        return Vec::new();
    }
    scored.select_nth_unstable_by(kk - 1, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(kk);
    scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().map(|(id, _)| id).collect()
}

/// recall@`k` of `got` (the IVF's per-query top-k ids) vs `gold` (exact cosine top-k).
pub(crate) fn recall_at(gold: &[Vec<ItemId>], got: &[Vec<ItemId>], k: usize) -> f64 {
    let mut hit = 0usize;
    let mut total = 0usize;
    for (g, a) in gold.iter().zip(got) {
        let gset: std::collections::HashSet<ItemId> = g.iter().take(k).copied().collect();
        hit += a.iter().take(k).filter(|i| gset.contains(i)).count();
        total += gset.len();
    }
    if total == 0 {
        0.0
    } else {
        hit as f64 / total as f64
    }
}

/// One (codec, nprobe) operating point on the Pareto.
pub struct ParetoPoint {
    pub codec: String,
    pub regime: String,
    pub nprobe: usize,
    pub recall10: f64,
    pub qps: f64,
    pub mean_us: f64, // mean wall-clock per query (1/qps × threads-ish; reported raw)
    pub bytes_per_vec: usize,
    pub total_resident_bytes_per_vec: usize,
}

pub struct IvfReport {
    pub dataset: String,
    pub dim: usize,
    pub n_db: usize,
    pub n_queries: usize,
    pub nlist: usize,
    pub bits: u8,
    pub seed: u64,
    pub threads: usize,
    pub nprobes: Vec<usize>,
    pub points: Vec<ParetoPoint>,
}

/// Build the IVF once per codec and sweep `nprobes`, measuring recall@10 (vs exact
/// cosine GT) and QPS (wall-clock over the query set, `threads` threads — queries
/// fanned out by rayon) at each nprobe, for every codec in `codecs`.
///
/// QPS protocol: for each (codec, nprobe) we run ALL queries (`reps` passes for a
/// stable timer), parallelized across queries; QPS = (n_queries × reps) / wall. A
/// warmup pass primes caches before timing. recall is computed once (nprobe fixes
/// the candidate set; reps don't change it).
#[allow(clippy::too_many_arguments)]
pub fn ivf_pareto(
    name: &str,
    dim: usize,
    db: &[Vec<f32>],
    queries: &[Vec<f32>],
    bits: u8,
    nlist: usize,
    nprobes: &[usize],
    codecs: &[IvfCodec],
    seed: u64,
    reps: usize,
) -> IvfReport {
    let threads = rayon::current_num_threads();
    // Exact cosine ground truth over the same database, computed in parallel.
    eprintln!(
        "ivf: computing exact-cosine ground-truth (k=10) for {} queries over {} db…",
        queries.len(),
        db.len()
    );
    let gold: Vec<Vec<ItemId>> = queries.par_iter().map(|q| exact_topk(q, db, 10)).collect();

    // ONE coarse quantizer for all codecs: k-means → per-vector assignment.
    eprintln!("ivf: k-means coarse quantizer, nlist={nlist} (seed {seed})…");
    let centroids = crate::bench::kmeans(dim, db, nlist, seed);
    let assign: Vec<u32> = db
        .par_iter()
        .map(|v| {
            let n = l2_norm(v).max(f32::EPSILON);
            let u: Vec<f32> = v.iter().map(|x| x / n).collect();
            let mut best = 0u32;
            let mut bd = f32::NEG_INFINITY;
            for (j, c) in centroids.iter().enumerate() {
                let d: f32 = u.iter().zip(c).map(|(a, b)| a * b).sum();
                if d > bd {
                    bd = d;
                    best = j as u32;
                }
            }
            best
        })
        .collect();
    // Posting-list size stats (an IVF health check — wildly skewed lists hurt QPS).
    let mut sizes = vec![0usize; nlist];
    for &c in &assign {
        sizes[c as usize] += 1;
    }
    let nonempty = sizes.iter().filter(|&&s| s > 0).count();
    let maxsz = sizes.iter().copied().max().unwrap_or(0);
    eprintln!(
        "ivf: {nonempty}/{nlist} non-empty lists, mean {:.0}, max {maxsz}",
        db.len() as f64 / nonempty.max(1) as f64
    );

    let mut points = Vec::new();
    for &codec in codecs {
        eprintln!("ivf: building {} index…", codec.label());
        let t0 = Instant::now();
        let index = IvfIndex::build(dim, db, &centroids, &assign, codec, bits);
        let build_s = t0.elapsed().as_secs_f64();
        let bpv = if db.is_empty() {
            0
        } else {
            index.mem_bytes() / db.len()
        };
        let total_bpv = if db.is_empty() {
            0
        } else {
            index.memory_breakdown().total_resident_bytes() / db.len()
        };
        eprintln!(
            "ivf: built {} in {build_s:.1}s ({} bytes/vec, {})",
            codec.label(),
            bpv,
            codec.regime()
        );
        for &nprobe in nprobes {
            // recall (once — candidate set fixed by nprobe).
            let got: Vec<Vec<ItemId>> = queries
                .par_iter()
                .map(|q| {
                    index
                        .search(q, 10, nprobe)
                        .into_iter()
                        .map(|(id, _)| id)
                        .collect()
                })
                .collect();
            let recall10 = recall_at(&gold, &got, 10);
            // QPS: warmup, then `reps` timed passes, queries fanned across threads.
            let _ = queries
                .par_iter()
                .map(|q| index.search(q, 10, nprobe).len())
                .sum::<usize>();
            let (wall, passes) = crate::timed_window(reps, || {
                let _ = queries
                    .par_iter()
                    .map(|q| index.search(q, 10, nprobe).len())
                    .sum::<usize>();
            });
            let total_q = (queries.len() * passes) as f64;
            let qps = total_q / wall;
            let mean_us = wall / total_q * 1e6;
            eprintln!(
                "ivf:   {} nprobe={nprobe:<4} R@10={recall10:.4} QPS={qps:>9.0}",
                codec.label()
            );
            points.push(ParetoPoint {
                codec: codec.label().to_string(),
                regime: codec.regime().to_string(),
                nprobe,
                recall10,
                qps,
                mean_us,
                bytes_per_vec: bpv,
                total_resident_bytes_per_vec: total_bpv,
            });
        }
    }

    IvfReport {
        dataset: name.to_string(),
        dim,
        n_db: db.len(),
        n_queries: queries.len(),
        nlist,
        bits,
        seed,
        threads,
        nprobes: nprobes.to_vec(),
        points,
    }
}

// ── recall@QPS WITH an exact-fp32 rerank stage (the deployed ANN metric) ──

/// One (codec, nprobe, rerank-depth) operating point WITH exact rerank.
pub struct RerankPoint {
    pub codec: String,
    pub regime: String,
    pub nprobe: usize,
    pub rdepth: usize,
    pub recall10: f64,
    pub qps: f64,
    pub bytes_per_vec: usize,
    pub total_resident_bytes_per_vec: usize,
}

pub struct RerankReport {
    pub dataset: String,
    pub dim: usize,
    pub n_db: usize,
    pub n_queries: usize,
    pub nlist: usize,
    pub bits: u8,
    pub threads: usize,
    pub points: Vec<RerankPoint>,
}

/// recall@10-vs-QPS WITH exact rerank: per codec, at a fixed nprobe, sweep the rerank depth `rdepths`
/// (0 = no rerank). The deployed-metric answer: codes-only RAM stays compressed; rerank reads fp32 from a
/// separate tier. The fidelity codec reaches a target recall at a smaller rdepth -> higher QPS.
#[allow(clippy::too_many_arguments)]
pub fn ivf_rerank_pareto(
    name: &str,
    dim: usize,
    db: &[Vec<f32>],
    queries: &[Vec<f32>],
    bits: u8,
    nlist: usize,
    nprobe: usize,
    rdepths: &[usize],
    codecs: &[IvfCodec],
    seed: u64,
    reps: usize,
) -> RerankReport {
    let threads = rayon::current_num_threads();
    eprintln!(
        "ivf-rerank: exact-cosine GT for {} queries over {} db…",
        queries.len(),
        db.len()
    );
    let gold: Vec<Vec<ItemId>> = queries.par_iter().map(|q| exact_topk(q, db, 10)).collect();
    let centroids = crate::bench::kmeans(dim, db, nlist, seed);
    let assign: Vec<u32> = db
        .par_iter()
        .map(|v| {
            let n = l2_norm(v).max(f32::EPSILON);
            let u: Vec<f32> = v.iter().map(|x| x / n).collect();
            let mut best = 0u32;
            let mut bd = f32::NEG_INFINITY;
            for (j, c) in centroids.iter().enumerate() {
                let d: f32 = u.iter().zip(c).map(|(a, b)| a * b).sum();
                if d > bd {
                    bd = d;
                    best = j as u32;
                }
            }
            best
        })
        .collect();
    let mut points = Vec::new();
    for &codec in codecs {
        let index = IvfIndex::build(dim, db, &centroids, &assign, codec, bits);
        let bpv = if db.is_empty() {
            0
        } else {
            index.mem_bytes() / db.len()
        };
        let total_bpv = if db.is_empty() {
            0
        } else {
            index.memory_breakdown().total_resident_bytes() / db.len()
        };
        eprintln!(
            "ivf-rerank: {} ({} B/vec, {})",
            codec.label(),
            bpv,
            codec.regime()
        );
        for &rd in rdepths {
            let search = |q: &[f32]| -> Vec<ItemId> {
                if rd == 0 {
                    index
                        .search(q, 10, nprobe)
                        .into_iter()
                        .map(|(id, _)| id)
                        .collect()
                } else {
                    index
                        .search_with_rerank(q, 10, nprobe, rd, db)
                        .into_iter()
                        .map(|(id, _)| id)
                        .collect()
                }
            };
            let got: Vec<Vec<ItemId>> = queries.par_iter().map(|q| search(q)).collect();
            let recall10 = recall_at(&gold, &got, 10);
            let _ = queries.par_iter().map(|q| search(q).len()).sum::<usize>(); // warmup
            let t = Instant::now();
            for _ in 0..reps {
                let _ = queries.par_iter().map(|q| search(q).len()).sum::<usize>();
            }
            let qps = (queries.len() * reps) as f64 / t.elapsed().as_secs_f64();
            eprintln!(
                "ivf-rerank:   {} nprobe={nprobe} rdepth={rd:<4} R@10={recall10:.4} QPS={qps:>9.0}",
                codec.label()
            );
            points.push(RerankPoint {
                codec: codec.label().to_string(),
                regime: codec.regime().to_string(),
                nprobe,
                rdepth: rd,
                recall10,
                qps,
                bytes_per_vec: bpv,
                total_resident_bytes_per_vec: total_bpv
                    + if rd == 0 {
                        0
                    } else {
                        dim * std::mem::size_of::<f32>()
                    },
            });
        }
    }
    RerankReport {
        dataset: name.to_string(),
        dim,
        n_db: db.len(),
        n_queries: queries.len(),
        nlist,
        bits,
        threads,
        points,
    }
}

pub fn render_rerank_markdown(r: &RerankReport) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "# IVF recall@10 vs QPS WITH exact rerank — {}\n\n",
        r.dataset
    ));
    s.push_str(&format!(
        "- dim **{}**, db **{}**, queries **{}**, nlist **{}**, {}-bit, **{} threads**. The deployed ANN \
         protocol: IVF retrieves the top-`rdepth` candidates by the compressed codec, then re-scores them \
         by **exact fp32 cosine** (a separate memory tier) and returns the true top-10.\n",
        r.dim, r.n_db, r.n_queries, r.nlist, r.bits, r.threads
    ));
    s.push_str("- `code B/vec` is the serialized codec footprint. `resident B/vec` includes index/model/cache memory and, when `rdepth>0`, the fp32 exact-vector tier.\n\n");
    s.push_str("| codec | regime | nprobe | rdepth | recall@10 | QPS | code B/vec | resident B/vec |\n|---|---|---|---|---|---|---|---|\n");
    for p in &r.points {
        s.push_str(&format!(
            "| {} | {} | {} | {} | {:.4} | {:.0} | {} | {} |\n",
            p.codec,
            p.regime,
            p.nprobe,
            p.rdepth,
            p.recall10,
            p.qps,
            p.bytes_per_vec,
            p.total_resident_bytes_per_vec
        ));
    }
    // QPS at a fixed recall target (the deployed Pareto operating point).
    s.push_str("\n## QPS at a fixed recall target (smaller rdepth -> higher QPS; the fidelity codec needs fewer)\n\n");
    for tgt in [0.90f64, 0.95, 0.99] {
        s.push_str(&format!("- **recall@10 >= {tgt:.2}:** "));
        let mut seen = Vec::new();
        for p in &r.points {
            if !seen.contains(&p.codec) {
                seen.push(p.codec.clone());
            }
        }
        let mut parts = Vec::new();
        for c in &seen {
            let best = r
                .points
                .iter()
                .filter(|p| &p.codec == c && p.recall10 >= tgt)
                .max_by(|a, b| {
                    a.qps
                        .partial_cmp(&b.qps)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            match best {
                Some(p) => parts.push(format!("{} {:.0} QPS @ rdepth {}", c, p.qps, p.rdepth)),
                None => parts.push(format!("{} (target unreached)", c)),
            }
        }
        s.push_str(&parts.join(" · "));
        s.push('\n');
    }
    s
}

/// Render the recall-vs-QPS Pareto as markdown: a per-(codec, nprobe) table, plus a
/// per-codec "QPS @ a recall target" summary (the operating points a systems paper
/// quotes), with both trellis deployment modes represented explicitly.
pub fn render_ivf_markdown(r: &IvfReport) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "# IVF recall@10 vs QPS Pareto — {}\n\n",
        r.dataset
    ));
    s.push_str(&format!(
        "- dim **{}**, db **{}**, queries **{}**, nlist **{}**, {}-bit, seed **{}**, **{} threads** (rayon, queries fanned out)\n",
        r.dim, r.n_db, r.n_queries, r.nlist, r.bits, r.seed, r.threads
    ));
    s.push_str("- recall@10 is measured against exact cosine top-10 ground truth over the same database; QPS is wall-clock over the full query set.\n");
    s.push_str("- trellis is reported in both deployment modes: `trellis_recon` uses a decode-free fp32 reconstruction cache; `trellis_codes` performs per-candidate Viterbi-state decoding from the compact representation.\n\n");

    // Main table: one row per (codec, nprobe).
    s.push_str("## recall@10 vs QPS, per codec × nprobe\n\n");
    s.push_str(
        "| codec | regime | nprobe | recall@10 | QPS | mean µs/q | code B/vec | resident B/vec |\n",
    );
    s.push_str("|---|---|---|---|---|---|---|---|\n");
    for p in &r.points {
        s.push_str(&format!(
            "| {} | {} | {} | {:.4} | {:.0} | {:.1} | {} | {} |\n",
            p.codec,
            p.regime,
            p.nprobe,
            p.recall10,
            p.qps,
            p.mean_us,
            p.bytes_per_vec,
            p.total_resident_bytes_per_vec
        ));
    }

    // Per-codec QPS at recall thresholds (the Pareto-operating-point view).
    let codecs: Vec<String> = {
        let mut seen = Vec::new();
        for p in &r.points {
            if !seen.contains(&p.codec) {
                seen.push(p.codec.clone());
            }
        }
        seen
    };
    // Recall targets: the fixed systems-paper thresholds that are actually
    // reachable here, plus data-derived ones (fractions of the max achieved recall)
    // so the "QPS @ target" view is informative even at a low-bit recall ceiling
    // (at 2-bit on SIFT the quantizer tops out well below 0.99 — that's intrinsic,
    // not an IVF artifact; full-probe IVF recall == flat-scan recall, see tests).
    let max_recall = r.points.iter().map(|p| p.recall10).fold(0.0f64, f64::max);
    let mut targets: Vec<f64> = [0.80f64, 0.90, 0.95, 0.99]
        .into_iter()
        .filter(|&t| t <= max_recall)
        .collect();
    for frac in [0.90f64, 0.95, 0.99] {
        let t = (max_recall * frac * 1000.0).round() / 1000.0;
        if t > 0.0 && !targets.iter().any(|&x| (x - t).abs() < 1e-6) {
            targets.push(t);
        }
    }
    targets.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s.push_str("\n## QPS at a recall@10 target (highest QPS among nprobes that reach it)\n\n");
    s.push_str(&format!(
        "Targets include the standard systems-paper thresholds reached here plus data-derived ones (fractions of the max achieved recall@10 = {:.3}).\n\n",
        max_recall
    ));
    s.push_str("| codec |");
    for t in &targets {
        s.push_str(&format!(" R@10≥{:.2} |", t));
    }
    s.push_str("\n|---|");
    for _ in &targets {
        s.push_str("---|");
    }
    s.push('\n');
    for c in &codecs {
        s.push_str(&format!("| {} |", c));
        for &t in &targets {
            // highest QPS among this codec's points that meet the recall target.
            let best = r
                .points
                .iter()
                .filter(|p| &p.codec == c && p.recall10 >= t)
                .map(|p| p.qps)
                .fold(f64::NEG_INFINITY, f64::max);
            if best.is_finite() {
                s.push_str(&format!(" {:.0} |", best));
            } else {
                s.push_str(" — |");
            }
        }
        s.push('\n');
    }

    // Pareto frontier across ALL codecs (which codec wins at each recall target).
    s.push_str("\n## Pareto winner per recall@10 target (max QPS across all codecs)\n\n");
    s.push_str("| R@10 target | winning codec | QPS | nprobe |\n|---|---|---|---|\n");
    for &t in &targets {
        let win = r.points.iter().filter(|p| p.recall10 >= t).max_by(|a, b| {
            a.qps
                .partial_cmp(&b.qps)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        match win {
            Some(p) => s.push_str(&format!(
                "| ≥{:.2} | {} ({}) | {:.0} | {} |\n",
                t, p.codec, p.regime, p.qps, p.nprobe
            )),
            None => s.push_str(&format!("| ≥{:.2} | (none reached) | — | — |\n", t)),
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::next_f64;

    fn rand_unit(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                let mut v: Vec<f32> = (0..dim)
                    .map(|_| {
                        let u1 = next_f64(&mut state).max(1e-12);
                        let u2 = next_f64(&mut state);
                        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
                    })
                    .collect();
                let nn = l2_norm(&v).max(f32::EPSILON);
                v.iter_mut().for_each(|x| *x /= nn);
                v
            })
            .collect()
    }

    #[test]
    fn ivf_full_probe_recovers_flat_recall() {
        // With nprobe == nlist the IVF scans every candidate, so its recall must
        // equal the codec's flat-scan recall (no coarse-stage loss). High-bit
        // trellis on clean Gaussian data ⇒ near-1.0; the structural check is that
        // exhaustive-probe IVF loses nothing vs flat.
        let dim = 128;
        let db = rand_unit(400, dim, 31);
        let queries = rand_unit(40, dim, 37);
        let nlist = 16;
        let centroids = crate::bench::kmeans(dim, &db, nlist, 42);
        let assign: Vec<u32> = db
            .iter()
            .map(|v| {
                let n = l2_norm(v).max(f32::EPSILON);
                let u: Vec<f32> = v.iter().map(|x| x / n).collect();
                let mut best = 0u32;
                let mut bd = f32::NEG_INFINITY;
                for (j, c) in centroids.iter().enumerate() {
                    let d: f32 = u.iter().zip(c).map(|(a, b)| a * b).sum();
                    if d > bd {
                        bd = d;
                        best = j as u32;
                    }
                }
                best
            })
            .collect();
        let gold: Vec<Vec<ItemId>> = queries.iter().map(|q| exact_topk(q, &db, 10)).collect();

        let index = IvfIndex::build(dim, &db, &centroids, &assign, IvfCodec::TrellisRecon, 8);
        // full probe (nprobe = nlist) — exhaustive, no coarse loss.
        let got_full: Vec<Vec<ItemId>> = queries
            .iter()
            .map(|q| {
                index
                    .search(q, 10, nlist)
                    .into_iter()
                    .map(|(id, _)| id)
                    .collect()
            })
            .collect();
        let r_full = recall_at(&gold, &got_full, 10);
        assert!(
            r_full > 0.95,
            "full-probe trellis IVF R@10 {r_full} (should ≈ flat, ~1.0)"
        );

        // nprobe=1 must be <= full probe (fewer candidates ⇒ no more recall).
        let got1: Vec<Vec<ItemId>> = queries
            .iter()
            .map(|q| {
                index
                    .search(q, 10, 1)
                    .into_iter()
                    .map(|(id, _)| id)
                    .collect()
            })
            .collect();
        let r1 = recall_at(&gold, &got1, 10);
        assert!(
            r1 <= r_full + 1e-9,
            "nprobe=1 recall {r1} > full-probe {r_full}"
        );
    }

    #[test]
    fn ivf_codes_and_recon_trellis_agree_at_full_probe() {
        // The two trellis §6 modes index the SAME codes; at full probe they must
        // return the same recall (the mode only changes WHERE x̄ is decoded).
        let dim = 64;
        let db = rand_unit(300, dim, 41);
        let queries = rand_unit(30, dim, 43);
        let nlist = 8;
        let report = ivf_pareto(
            "synth",
            dim,
            &db,
            &queries,
            4,
            nlist,
            &[nlist], // full probe
            &[IvfCodec::TrellisRecon, IvfCodec::TrellisCodes],
            42,
            1,
        );
        let recon = report
            .points
            .iter()
            .find(|p| p.codec == "trellis_recon")
            .unwrap();
        let codes = report
            .points
            .iter()
            .find(|p| p.codec == "trellis_codes")
            .unwrap();
        assert!(
            (recon.recall10 - codes.recall10).abs() < 1e-9,
            "trellis recon {} vs codes {} recall differ",
            recon.recall10,
            codes.recall10
        );
        // Markdown renders without panicking.
        assert!(render_ivf_markdown(&report).contains("Pareto"));
    }
}
