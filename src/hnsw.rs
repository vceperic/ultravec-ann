//! Pure-Rust, BLAS-free HNSW (Malkov & Yashunin 2018) — to show the oblivious codec is a drop-in for
//! the dominant production graph index, not only IVF. The recall edge that survives an IVF (`ivf.rs`)
//! should survive a graph walk too.
//!
//! The graph is rebuilt from each codec's reconstructed vectors, and search distances use the same
//! resident reconstructions. This measures end-to-end navigation quality for a controlled in-crate
//! HNSW implementation; it is not a codes-only production index. Distances are cosine = dot on unit vectors (we store reconstructed *unit*
//! directions); HNSW minimizes distance, so we maximize similarity and negate where a min-heap is wanted.
//!
//! Reconstruction is resident (an fp32 RAM accelerator, as in the IVF reconstruction-cache mode); the
//! codes are the reported compressed bytes. A codes-only variant would decode per distance eval.

use crate::{l2_norm, ItemId, MemoryBreakdown, VectorBackend};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Min/max heap element keyed by similarity (f32). `Ord` by similarity; NaN-safe via total_cmp.
#[derive(Clone, Copy)]
struct Cand {
    sim: f32,
    id: u32,
}
impl PartialEq for Cand {
    fn eq(&self, o: &Self) -> bool {
        self.sim == o.sim && self.id == o.id
    }
}
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Cand {
    fn cmp(&self, o: &Self) -> Ordering {
        self.sim.total_cmp(&o.sim).then(self.id.cmp(&o.id))
    }
}

/// Epoch-stamped visited set: O(1) `clear` (bump the epoch; only the 2³²-wrap does an O(n) reset), so a
/// single buffer is reused across every `search_layer` call. Without this the `vec![false; n]` per call
/// makes graph build O(n²) — fine at 20k, fatal at 1M. One `Visited` per thread (build is sequential;
/// query search reuses one per rayon worker via `map_init`).
/// Reusable scratch space for [`Hnsw::search_with`].
///
/// Keeping this caller-owned avoids allocating and clearing an `O(n)` visited
/// array for every query.
pub struct Visited {
    mark: Vec<u32>,
    epoch: u32,
}
impl Visited {
    pub fn new(n: usize) -> Self {
        Self {
            mark: vec![0u32; n],
            epoch: 0,
        }
    }
    #[inline]
    fn clear(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.mark.iter_mut().for_each(|m| *m = 0);
            self.epoch = 1;
        }
    }
    /// Mark `i` visited; return whether it was *already* visited.
    #[inline]
    fn test_set(&mut self, i: usize) -> bool {
        if self.mark[i] == self.epoch {
            true
        } else {
            self.mark[i] = self.epoch;
            false
        }
    }
}

pub struct Hnsw {
    dim: usize,
    /// Reconstructed UNIT directions, row-major `n × dim` — distances are dot products against these.
    recon: Vec<f32>,
    n: usize,
    /// Per-node neighbor lists, per layer: `links[node]` is a Vec over layers (0 = base), each a Vec<u32>.
    links: Vec<Vec<Vec<u32>>>,
    levels: Vec<u8>,
    entry: u32,
    max_level: usize,
    m: usize,  // neighbors per node (upper layers)
    m0: usize, // neighbors at layer 0 (= 2*m, the standard)
    ef_construction: usize,
    bytes_per_vec: usize, // the codec's reported compressed footprint (codes), for the memory column
    // True when the codec's stored record IS the resident reconstruction, as for
    // fp32. Without it the memory column charged such a codec twice: once as its
    // notional `dim*4` code record and again as the navigation cache holding the
    // very same array, which inflated the fp32 row and inverted the comparison
    // against the compressed codecs.
    codes_are_recon: bool,
    /// When present the fp32 array has been dropped and search scores from codes.
    /// Build still uses the reconstruction, so the graph is bit-for-bit the same one
    /// the fp32-navigation rows describe and only the scan differs.
    scorer: Option<Box<dyn crate::CandidateScorer>>,
}

impl Hnsw {
    #[inline]
    fn vec(&self, i: u32) -> &[f32] {
        let i = i as usize;
        &self.recon[i * self.dim..(i + 1) * self.dim]
    }
    #[inline]
    fn dot(&self, q: &[f32], i: u32) -> f32 {
        let v = self.vec(i);
        let mut s = 0.0f32;
        for j in 0..self.dim {
            s += q[j] * v[j];
        }
        s
    }

    /// Build: reconstruct every db vector via `codec` (the graph + search both use the reconstruction, so
    /// the navigation quality IS the codec's), then insert in order with the standard HNSW heuristic.
    /// `m` neighbors (upper), `2m` at layer 0, `ef_construction` candidate breadth, seeded levels.
    pub fn build(
        dim: usize,
        db: &[Vec<f32>],
        codec: &dyn VectorBackend,
        bytes_per_vec: usize,
        codes_are_recon: bool,
        m: usize,
        ef_construction: usize,
        seed: u64,
    ) -> Self {
        use rayon::prelude::*;
        let n = db.len();
        assert!(dim > 0, "HNSW dimension must be positive");
        assert!(m >= 2, "HNSW M must be at least 2");
        assert!(
            db.iter().all(|v| v.len() == dim),
            "all HNSW vectors must match dim"
        );
        // Reconstruct every vector to a unit direction (parallel — reconstruct_unit is &self + Sync).
        let recon: Vec<f32> = db
            .par_iter()
            .flat_map_iter(|v| {
                codec
                    .reconstruct_unit(v)
                    .unwrap_or_else(|| {
                        let nn = l2_norm(v).max(f32::EPSILON);
                        v.iter().map(|x| x / nn).collect()
                    })
                    .into_iter()
            })
            .collect();
        let m0 = 2 * m;
        let ml = 1.0f64 / (m as f64).ln();
        let mut st = seed.max(1);
        let mut levels = vec![0u8; n];
        let mut max_level = 0usize;
        for lv in levels.iter_mut() {
            // level = floor(-ln(U) * ml)
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            let u = ((st >> 11) as f64 / (1u64 << 53) as f64).clamp(1e-12, 1.0);
            let l = (-(u.ln()) * ml).floor() as usize;
            *lv = l.min(31) as u8;
            max_level = max_level.max(l);
        }
        let mut h = Hnsw {
            scorer: None,
            dim,
            recon,
            n,
            links: vec![Vec::new(); n],
            levels,
            entry: 0,
            max_level: 0,
            m,
            m0,
            ef_construction,
            bytes_per_vec,
            codes_are_recon,
        };
        for node in 0..n {
            let lv = h.levels[node] as usize;
            h.links[node] = (0..=lv).map(|_| Vec::new()).collect();
        }
        // Insert node 0 as the initial entry, then the rest. One reused visited buffer (build is sequential).
        h.entry = 0;
        if n == 0 {
            return h;
        }
        h.max_level = h.levels[0] as usize;
        let mut vis = Visited::new(n);
        for node in 1..n {
            h.insert(node as u32, &mut vis);
        }
        let _ = max_level;
        h
    }

    /// Greedy 1-best descent from `ep` through one layer using the layer's neighbor lists.
    fn greedy_descend(&self, score: &mut dyn FnMut(u32) -> f32, ep: u32, layer: usize) -> u32 {
        let mut cur = ep;
        let mut cur_sim = score(cur);
        loop {
            let mut improved = false;
            if layer < self.links[cur as usize].len() {
                for &nb in &self.links[cur as usize][layer] {
                    let s = score(nb);
                    if s > cur_sim {
                        cur_sim = s;
                        cur = nb;
                        improved = true;
                    }
                }
            }
            if !improved {
                return cur;
            }
        }
    }

    /// ef-search at `layer` from entry points `eps`: returns up to `ef` nearest (by similarity), the
    /// standard HNSW SEARCH-LAYER (a visited set + a candidate max-heap + a result min-heap of size ef).
    fn search_layer(
        &self,
        score: &mut dyn FnMut(u32) -> f32,
        eps: &[u32],
        layer: usize,
        ef: usize,
        vis: &mut Visited,
    ) -> Vec<Cand> {
        vis.clear();
        let mut cand: BinaryHeap<Cand> = BinaryHeap::new(); // max-heap on sim (best to expand)
        let mut result: BinaryHeap<std::cmp::Reverse<Cand>> = BinaryHeap::new(); // min-heap (worst on top)
        for &e in eps {
            let s = score(e);
            vis.test_set(e as usize);
            cand.push(Cand { sim: s, id: e });
            result.push(std::cmp::Reverse(Cand { sim: s, id: e }));
        }
        while let Some(c) = cand.pop() {
            // worst in result
            let worst = result.peek().map(|r| r.0.sim).unwrap_or(f32::NEG_INFINITY);
            if c.sim < worst && result.len() >= ef {
                break;
            }
            if layer < self.links[c.id as usize].len() {
                for &nb in &self.links[c.id as usize][layer] {
                    if !vis.test_set(nb as usize) {
                        let s = score(nb);
                        let worst = result.peek().map(|r| r.0.sim).unwrap_or(f32::NEG_INFINITY);
                        if s > worst || result.len() < ef {
                            cand.push(Cand { sim: s, id: nb });
                            result.push(std::cmp::Reverse(Cand { sim: s, id: nb }));
                            if result.len() > ef {
                                result.pop();
                            }
                        }
                    }
                }
            }
        }
        result.into_iter().map(|r| r.0).collect()
    }

    /// Select up to `m` neighbors from `cands` (simple top-m by similarity — robust and standard enough
    /// for the codec-comparison; the diversity heuristic is an orthogonal refinement).
    fn select_neighbors(&self, mut cands: Vec<Cand>, m: usize) -> Vec<u32> {
        cands.sort_unstable_by(|a, b| b.sim.total_cmp(&a.sim));
        cands.truncate(m);
        cands.into_iter().map(|c| c.id).collect()
    }

    fn insert(&mut self, node: u32, vis: &mut Visited) {
        let q: Vec<f32> = self.vec(node).to_vec();
        let lv = self.levels[node as usize] as usize;
        let mut ep = self.entry;
        // descend from the top down to lv+1 with greedy 1-best
        let top = self.max_level;
        let mut l = top;
        while l > lv {
            ep = {
                let mut score = |i: u32| self.dot(&q, i);
                self.greedy_descend(&mut score, ep, l)
            };
            if l == 0 {
                break;
            }
            l -= 1;
        }
        // from min(lv, top) down to 0: ef-search, select neighbors, link bidirectionally
        let start = lv.min(top);
        let mut eps = vec![ep];
        for layer in (0..=start).rev() {
            let found = {
                let mut score = |i: u32| self.dot(&q, i);
                self.search_layer(&mut score, &eps, layer, self.ef_construction, vis)
            };
            let mmax = if layer == 0 { self.m0 } else { self.m };
            let neigh = self.select_neighbors(found.clone(), self.m);
            // node -> neigh
            self.links[node as usize][layer] = neigh.clone();
            // neigh -> node (with pruning to mmax)
            for &nb in &neigh {
                let nb_layers = self.links[nb as usize].len();
                if layer < nb_layers {
                    self.links[nb as usize][layer].push(node);
                    if self.links[nb as usize][layer].len() > mmax {
                        let nbv: Vec<f32> = self.vec(nb).to_vec();
                        let cands: Vec<Cand> = self.links[nb as usize][layer]
                            .iter()
                            .map(|&x| Cand {
                                sim: self.dot(&nbv, x),
                                id: x,
                            })
                            .collect();
                        self.links[nb as usize][layer] = self.select_neighbors(cands, mmax);
                    }
                }
            }
            // next layer's entry points = this layer's found set
            eps = found.iter().map(|c| c.id).collect();
            if eps.is_empty() {
                eps = vec![ep];
            }
        }
        if lv > self.max_level {
            self.max_level = lv;
            self.entry = node;
        }
    }

    /// Search: greedy descent from the entry through the upper layers (ef=1), then ef-search at layer 0,
    /// return the top-`k` ids by (codec-reconstructed) cosine to `query`.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(ItemId, f32)> {
        let mut vis = Visited::new(self.n);
        self.search_with(query, k, ef, &mut vis)
    }

    /// Same as [`search`] but with a caller-owned [`Visited`] scratch — reuse one per rayon worker (via
    /// `map_init`) so the hot query path doesn't re-zero an `n`-length buffer per query (which at 1M would
    /// dominate the per-query cost and corrupt the codec-relative QPS).
    pub fn search_with(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        vis: &mut Visited,
    ) -> Vec<(ItemId, f32)> {
        if self.n == 0 || k == 0 {
            return Vec::new();
        }
        assert_eq!(query.len(), self.dim, "HNSW query dimension mismatch");
        assert_eq!(
            vis.mark.len(),
            self.n,
            "HNSW scratch was created for a different index size"
        );
        // The codec normalizes internally, so the codes-only path gets the raw query;
        // the fp32 path needs the unit vector its resident array is stored in.
        let nn = l2_norm(query).max(f32::EPSILON);
        let q: Vec<f32> = query.iter().map(|x| x / nn).collect();
        let mut prepared = self.scorer.as_ref().map(|s| s.prepare(query));
        let mut found = {
            let mut score: Box<dyn FnMut(u32) -> f32> = match prepared.as_mut() {
                Some(p) => Box::new(move |i: u32| p.score_at(i as usize)),
                None => Box::new(|i: u32| self.dot(&q, i)),
            };
            let mut ep = self.entry;
            let mut l = self.max_level;
            while l > 0 {
                ep = self.greedy_descend(&mut *score, ep, l);
                l -= 1;
            }
            self.search_layer(&mut *score, &[ep], 0, ef.max(k), vis)
        };
        found.sort_unstable_by(|a, b| b.sim.total_cmp(&a.sim));
        found.truncate(k);
        found.into_iter().map(|c| (c.id as ItemId, c.sim)).collect()
    }

    /// Drop the fp32 navigation array and score from the codec's codes instead.
    ///
    /// Called AFTER `build`, deliberately: construction reconstructs every vector and
    /// inserts against those reconstructions, so the graph this returns is the exact
    /// graph the fp32-navigation rows were measured on. Only the scan changes, which is
    /// what makes the recall comparison between the two a statement about scoring
    /// rather than about topology.
    pub fn into_codes_only(mut self, scorer: Box<dyn crate::CandidateScorer>) -> Self {
        assert_eq!(
            scorer.scored_len(),
            self.n,
            "codes-only scorer holds {} vectors but the graph has {}",
            scorer.scored_len(),
            self.n
        );
        self.recon = Vec::new();
        self.recon.shrink_to_fit();
        self.scorer = Some(scorer);
        self
    }

    pub fn bytes_per_vec(&self) -> usize {
        self.bytes_per_vec
    }

    pub fn memory_breakdown(&self) -> MemoryBreakdown {
        let list_headers = self
            .links
            .iter()
            .map(|layers| {
                layers.capacity() * std::mem::size_of::<Vec<u32>>()
                    + layers
                        .iter()
                        .map(|neighbors| neighbors.capacity() * std::mem::size_of::<u32>())
                        .sum::<usize>()
            })
            .sum::<usize>();
        MemoryBreakdown {
            // Charged only when the codes are a distinct store from `recon`.
            code_bytes: if self.codes_are_recon {
                0
            } else {
                self.bytes_per_vec.saturating_mul(self.n)
            },
            // A codes-only graph still pays the codec's rotation, code table and
            // centroids; reporting only the per-vector codes would understate it.
            model_bytes: self.scorer.as_ref().map_or(0, |s| s.scorer_model_bytes()),
            index_bytes: self.links.capacity() * std::mem::size_of::<Vec<Vec<u32>>>()
                + list_headers
                + self.levels.capacity() * std::mem::size_of::<u8>(),
            cache_bytes: self.recon.capacity() * std::mem::size_of::<f32>(),
        }
    }
}

// ── Codec drop-in study: build the same graph under each codec's reconstruction ──

/// An fp32 backend whose `reconstruct_unit` returns the exact unit direction — the loss-free upper
/// bound for graph navigation (shared by the Pareto driver and the correctness test).
pub(crate) struct Fp32Recon;
impl VectorBackend for Fp32Recon {
    fn dimensions(&self) -> usize {
        0
    }
    fn len(&self) -> usize {
        0
    }
    fn add(&mut self, _: ItemId, _: &[f32]) {}
    fn search(&self, _: &[f32], _: usize) -> Vec<(ItemId, f32)> {
        vec![]
    }
    fn mem_bytes(&self) -> usize {
        0
    }
    fn reconstruct_unit(&self, x: &[f32]) -> Option<Vec<f32>> {
        let n = l2_norm(x).max(f32::EPSILON);
        Some(x.iter().map(|v| v / n).collect())
    }
}

/// The codecs the HNSW drop-in is benchmarked under. The graph is rebuilt per codec on that codec's
/// reconstruction; navigation quality IS the codec's reconstruction quality.
#[derive(Clone, Copy)]
pub enum HnswCodec {
    /// Exact fp32 directions — the loss-free navigation upper bound (`dim·4` bytes/vec, uncompressed).
    Fp32,
    /// Frozen TurboQuant control (N(0,1), oblivious).
    Baseline,
    /// RaBitQ / Extended-RaBitQ (oblivious residual code).
    Rabitq,
    /// Trellis / TCQ (oblivious Gaussian-matched code) — the proposed codec.
    Trellis,
    /// Trellis navigating from its CODES: the same graph, built from reconstructions,
    /// but with the fp32 array dropped before search. The only configuration here that
    /// is actually RAM-compressed at query time.
    TrellisCodesOnly,
}

impl HnswCodec {
    /// Whether this codec's stored record *is* the resident reconstruction.
    ///
    /// Only fp32 is: this implementation keeps no separate code store for it, so
    /// its `dim*4` "codes" and the navigation cache are the same array. The
    /// compressed codecs genuinely pay both.
    pub fn codes_are_recon(self) -> bool {
        matches!(self, HnswCodec::Fp32)
    }
}

impl HnswCodec {
    pub fn label(self) -> &'static str {
        match self {
            HnswCodec::Fp32 => "fp32",
            HnswCodec::Baseline => "turboquant_baseline",
            HnswCodec::Rabitq => "rabitq",
            HnswCodec::Trellis => "trellis",
            HnswCodec::TrellisCodesOnly => "trellis_codes_only",
        }
    }
    /// Backend used to reconstruct each db vector when building/searching the graph.
    fn recon_backend(self, dim: usize, bits: u8) -> Box<dyn VectorBackend + Sync> {
        use crate::baseline::TurboQuantBaseline;
        use crate::rabitq::RaBitQ;
        use crate::trellis::TrellisQuantizer;
        match self {
            HnswCodec::Fp32 => Box::new(Fp32Recon),
            HnswCodec::Baseline => Box::new(TurboQuantBaseline::new(dim, bits)),
            HnswCodec::Rabitq => Box::new(RaBitQ::new(dim, bits)),
            HnswCodec::Trellis | HnswCodec::TrellisCodesOnly => {
                Box::new(TrellisQuantizer::new(dim, bits))
            }
        }
    }

    /// Populated codec used to score candidates once the fp32 array is dropped.
    /// `None` for every arm that navigates from reconstructions.
    fn codes_only_scorer(
        self,
        dim: usize,
        bits: u8,
        db: &[Vec<f32>],
    ) -> Option<Box<dyn crate::CandidateScorer>> {
        use crate::trellis::TrellisQuantizer;
        match self {
            HnswCodec::TrellisCodesOnly => {
                let mut q = TrellisQuantizer::new(dim, bits).with_codes_only(true);
                q.reserve(db.len());
                q.add_batch(db);
                Some(Box::new(q))
            }
            _ => None,
        }
    }
    /// Compressed codes footprint per vector (what's actually stored). fp32 = `dim·4` (uncompressed); the
    /// others encode a `sample` into a codes backend and read `mem_bytes()/len()` — identical accounting
    /// to the IVF study's codes-only column. The resident fp32 reconstruction is a decode accelerator (a
    /// codes-only variant would decode per distance eval), exactly the IVF recon-cache duality.
    fn bytes_per_vec(self, dim: usize, bits: u8, sample: &[Vec<f32>]) -> usize {
        use crate::baseline::TurboQuantBaseline;
        use crate::rabitq::RaBitQ;
        use crate::trellis::TrellisQuantizer;
        let mut b: Box<dyn VectorBackend + Sync> = match self {
            HnswCodec::Fp32 => return dim * 4,
            HnswCodec::Baseline => Box::new(TurboQuantBaseline::new(dim, bits)),
            HnswCodec::Rabitq => Box::new(RaBitQ::new(dim, bits)),
            HnswCodec::Trellis | HnswCodec::TrellisCodesOnly => {
                Box::new(TrellisQuantizer::new(dim, bits).with_codes_only(true))
            }
        };
        for (i, v) in sample.iter().enumerate() {
            b.add(i as ItemId, v);
        }
        if b.is_empty() {
            0
        } else {
            b.mem_bytes() / b.len()
        }
    }
}

/// One (codec, ef) operating point on the HNSW recall-vs-QPS Pareto.
pub struct HnswPoint {
    pub codec: String,
    pub ef: usize,
    pub recall10: f64,
    pub qps: f64,
    pub bytes_per_vec: usize,
    pub total_resident_bytes_per_vec: usize,
    pub build_s: f64,
}

pub struct HnswReport {
    pub dataset: String,
    pub dim: usize,
    pub n_db: usize,
    pub n_queries: usize,
    pub m: usize,
    pub ef_construction: usize,
    pub bits: u8,
    pub threads: usize,
    pub points: Vec<HnswPoint>,
}

/// recall@10-vs-QPS over a from-scratch HNSW, per codec, sweeping the query-time `ef`. The graph
/// topology (`m`, `ef_construction`) is fixed; each codec reconstructs the vectors the walk navigates on,
/// so a higher-fidelity codec reaches a target recall at a smaller `ef` → higher QPS at its footprint.
#[allow(clippy::too_many_arguments)]
pub fn hnsw_pareto(
    name: &str,
    dim: usize,
    db: &[Vec<f32>],
    queries: &[Vec<f32>],
    bits: u8,
    m: usize,
    ef_construction: usize,
    efs: &[usize],
    codecs: &[HnswCodec],
    seed: u64,
    reps: usize,
) -> HnswReport {
    use rayon::prelude::*;
    use std::time::Instant;
    let threads = rayon::current_num_threads();
    eprintln!(
        "hnsw: exact-cosine GT for {} queries over {} db…",
        queries.len(),
        db.len()
    );
    let gold: Vec<Vec<ItemId>> = queries
        .par_iter()
        .map(|q| crate::ivf::exact_topk(q, db, 10))
        .collect();
    // Byte-accounting sample (re-encoding all of 1M just for the bytes column is wasteful).
    let sample = &db[..db.len().min(4096)];
    let mut points = Vec::new();
    for &codec in codecs {
        let bpv = codec.bytes_per_vec(dim, bits, sample);
        let backend = codec.recon_backend(dim, bits);
        let t0 = Instant::now();
        let h = Hnsw::build(
            dim,
            db,
            backend.as_ref(),
            bpv,
            codec.codes_are_recon(),
            m,
            ef_construction,
            seed,
        );
        // Drop the fp32 array AFTER construction, so the graph measured here is the
        // one the reconstruction-navigation row was measured on.
        let h = match codec.codes_only_scorer(dim, bits, db) {
            Some(scorer) => h.into_codes_only(scorer),
            None => h,
        };
        let build_s = t0.elapsed().as_secs_f64();
        let total_bpv = if db.is_empty() {
            0
        } else {
            h.memory_breakdown().total_resident_bytes() / db.len()
        };
        eprintln!(
            "hnsw: {} built in {:.1}s ({} B/vec) — sweeping ef {:?}",
            codec.label(),
            build_s,
            bpv,
            efs
        );
        let n = db.len();
        for &ef in efs {
            let got: Vec<Vec<ItemId>> = queries
                .par_iter()
                .map_init(
                    || Visited::new(n),
                    |vis, q| {
                        h.search_with(q, 10, ef, vis)
                            .into_iter()
                            .map(|(id, _)| id)
                            .collect()
                    },
                )
                .collect();
            let recall10 = crate::ivf::recall_at(&gold, &got, 10);
            // warmup
            let _ = queries
                .par_iter()
                .map_init(
                    || Visited::new(n),
                    |vis, q| h.search_with(q, 10, ef, vis).len(),
                )
                .sum::<usize>();
            let (wall, passes) = crate::timed_window(reps, || {
                let _ = queries
                    .par_iter()
                    .map_init(
                        || Visited::new(n),
                        |vis, q| h.search_with(q, 10, ef, vis).len(),
                    )
                    .sum::<usize>();
            });
            let qps = (queries.len() * passes) as f64 / wall;
            eprintln!(
                "hnsw:   {} ef={ef:<4} R@10={recall10:.4} QPS={qps:>9.0}",
                codec.label()
            );
            points.push(HnswPoint {
                codec: codec.label().to_string(),
                ef,
                recall10,
                qps,
                bytes_per_vec: bpv,
                total_resident_bytes_per_vec: total_bpv,
                build_s,
            });
        }
    }
    HnswReport {
        dataset: name.to_string(),
        dim,
        n_db: db.len(),
        n_queries: queries.len(),
        m,
        ef_construction,
        bits,
        threads,
        points,
    }
}

pub fn render_hnsw_markdown(r: &HnswReport) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "# HNSW recall@10 vs QPS Pareto (codec drop-in) — {}\n\n",
        r.dataset
    ));
    s.push_str(&format!(
        "- dim **{}**, db **{}**, queries **{}**, graph **M={}**, ef_construction **{}**, {}-bit, **{} threads**.\n",
        r.dim, r.n_db, r.n_queries, r.m, r.ef_construction, r.bits, r.threads
    ));
    s.push_str(
        "- Pure-Rust, BLAS-free HNSW (Malkov & Yashunin 2018). A graph is rebuilt from each codec's \
         reconstructed vectors, and the same reconstructions are resident during search. This is a \
         controlled navigation study, not a codes-only production index. recall@10 is vs **exact cosine \
         top-10** over the same database.\n",
    );
    s.push_str(
        "- `code B/vec` is the serialized codec footprint. `resident B/vec` additionally includes the \
         graph and the mandatory fp32 reconstruction cache used by this implementation. Recall--QPS \
         comparisons must use the resident column.\n\n",
    );
    s.push_str(
        "| codec | ef | recall@10 | QPS | code B/vec | resident B/vec | build s |\n|---|---|---|---|---|---|---|\n",
    );
    for p in &r.points {
        s.push_str(&format!(
            "| {} | {} | {:.4} | {:.0} | {} | {} | {:.1} |\n",
            p.codec,
            p.ef,
            p.recall10,
            p.qps,
            p.bytes_per_vec,
            p.total_resident_bytes_per_vec,
            p.build_s
        ));
    }
    // QPS at a fixed recall target — the operating point a systems paper quotes.
    s.push_str("\n## QPS at a fixed recall target (the fidelity codec reaches it at smaller ef → higher QPS)\n\n");
    let mut seen = Vec::new();
    for p in &r.points {
        if !seen.contains(&p.codec) {
            seen.push(p.codec.clone());
        }
    }
    for tgt in [0.90f64, 0.95, 0.99] {
        s.push_str(&format!("- **recall@10 >= {tgt:.2}:** "));
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
                Some(p) => parts.push(format!(
                    "{} {:.0} QPS @ ef {} ({} B/vec)",
                    c, p.qps, p.ef, p.bytes_per_vec
                )),
                None => parts.push(format!("{} (target unreached)", c)),
            }
        }
        s.push_str(&parts.join(" · "));
        s.push('\n');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{cosine, next_f64};

    #[test]
    fn codes_only_graph_drops_the_fp32_array_without_losing_the_graph() {
        use crate::trellis::TrellisQuantizer;
        let (dim, n) = (64, 400);
        let db = rand_unit(n, dim, 11);
        let queries = rand_unit(40, dim, 13);
        let build = || {
            Hnsw::build(
                dim,
                &db,
                HnswCodec::Trellis.recon_backend(dim, 2).as_ref(),
                38,
                false,
                8,
                64,
                42,
            )
        };
        let recon_nav = build();
        let codes_only = {
            let mut q = TrellisQuantizer::new(dim, 2).with_codes_only(true);
            q.add_batch(&db);
            build().into_codes_only(Box::new(q))
        };

        // The point of the arm: nothing fp32 is resident at query time.
        let before = recon_nav.memory_breakdown();
        let after = codes_only.memory_breakdown();
        assert!(
            before.cache_bytes > 0,
            "the control must hold a resident array"
        );
        assert_eq!(after.cache_bytes, 0, "codes-only still holds an fp32 cache");
        assert!(
            after.total_resident_bytes() < before.total_resident_bytes(),
            "codes-only resident {} is not below reconstruction-nav {}",
            after.total_resident_bytes(),
            before.total_resident_bytes()
        );

        // The graph is the same object, so the walk must still find neighbours: the
        // scorer changed, not the topology. Compared against exact cosine rather than
        // against the other arm, because the two rank by different estimators.
        let mut hits = 0usize;
        for qv in &queries {
            let exact = db
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| cosine(a, qv).total_cmp(&cosine(b, qv)))
                .map(|(i, _)| i as ItemId)
                .unwrap();
            let got = codes_only.search(qv, 10, 64);
            assert_eq!(
                got.len(),
                10,
                "codes-only search returned {} results",
                got.len()
            );
            if got.iter().any(|(id, _)| *id == exact) {
                hits += 1;
            }
        }
        assert!(
            hits * 2 > queries.len(),
            "codes-only recall@10 collapsed: {hits}/{} queries kept the true nearest",
            queries.len()
        );
    }

    fn rand_unit(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut st = seed;
        (0..n)
            .map(|_| {
                let v: Vec<f32> = (0..dim)
                    .map(|_| {
                        let u1 = next_f64(&mut st).max(1e-12);
                        let u2 = next_f64(&mut st);
                        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
                    })
                    .collect();
                let nn = l2_norm(&v).max(f32::EPSILON);
                v.iter().map(|x| x / nn).collect()
            })
            .collect()
    }

    #[test]
    fn hnsw_fp32_high_recall_vs_bruteforce() {
        // With fp32 reconstruction the graph should retrieve the exact NN with high recall@10.
        let dim = 64;
        let db = rand_unit(2000, dim, 7);
        let h = Hnsw::build(dim, &db, &Fp32Recon, dim * 4, true, 16, 100, 42);
        let queries = rand_unit(100, dim, 99);
        let mut hit = 0usize;
        for q in &queries {
            // exact top-10
            let mut gold: Vec<(usize, f32)> = db
                .iter()
                .enumerate()
                .map(|(i, v)| (i, cosine(q, v)))
                .collect();
            gold.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            let gset: std::collections::HashSet<i64> =
                gold.iter().take(10).map(|&(i, _)| i as i64).collect();
            let got = h.search(q, 10, 64);
            hit += got.iter().filter(|(id, _)| gset.contains(id)).count();
        }
        let recall = hit as f64 / (queries.len() * 10) as f64;
        assert!(recall > 0.90, "HNSW fp32 recall@10 too low: {recall}");
    }

    #[test]
    fn empty_hnsw_search_is_empty() {
        let h = Hnsw::build(8, &[], &Fp32Recon, 32, true, 8, 32, 42);
        let mut scratch = Visited::new(0);
        assert!(h.search_with(&[0.0; 8], 10, 32, &mut scratch).is_empty());
        assert_eq!(h.memory_breakdown().total_resident_bytes(), 0);
    }
}
