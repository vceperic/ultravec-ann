//! BlockQuant—"Block-Sphere Vector Quantization" (Ann, Lee, Oh, arXiv
//! 2605.19972, 2026-05-19)—implemented as an oblivious rotation-based reference
//! from Algorithm 1 and the block-marginal codebook specification.
//!
//! Method: rotate x→z (oblivious), split z into m=d/p blocks of size p, quantize
//! each block to the nearest of 2^(b·p) codewords. The codebook is **data-oblivious**:
//! it is the k-means solution for the *block marginal of a uniform sphere vector*,
//! whose density on the p-ball is f_{p,d}(z)=c·(1−‖z‖²)^((d−p−2)/2). We sample that
//! marginal exactly — draw u uniform on S^{d−1} (Gaussian then normalize), take the
//! first p coords — and k-means those samples. Depends only on (d,p,b), never data.
//!
//! Three estimator variants (the paper's, selected by `ULTRAVEC_BLOCKQUANT_EST`):
//!   - UB  (default): unbiased inner product, S = 1/ρ          (recall-oriented)
//!   - BSM: reconstruction-optimal,           S = ρ/‖x̄‖²       (MSE-oriented)
//!   - MSE: raw,                              S = 1
//! where ρ = ⟨z, z̄⟩ and z̄ is the concatenated reconstruction. Score ⟨q,x̂⟩ =
//! S·⟨q_r, z̄⟩. The block win over scalar codes is that a p-D codeword captures the
//! *joint* block geometry on the sphere (like E8 does for p=8 lattice cells), but
//! here the codebook is matched to the exact spherical block marginal.

use crate::{
    l2_norm, pack_indices, splitmix64, unpack_indices, ItemId, MemoryBreakdown, Rotor,
    VectorBackend,
};

const CODEBOOK_SEED: u64 = 0xB10C_5EED;
const KMEANS_SAMPLES: usize = 50_000;
const KMEANS_ITERS: usize = 25;
/// A codeword no training point supports is not a centroid, it is a memorized
/// sample. The block codebook holds `2^(bits*p)` entries, so the training set
/// scales with it. The samples are synthetic draws from the spherical block
/// marginal, so they cost nothing but time.
const MIN_SAMPLES_PER_CODEWORD: usize = 40;

/// Cap on `bits * p`, i.e. on the block index width.
///
/// The two constraints on the codebook pull against each other. Wide blocks are
/// the whole point of the method, but Lloyd costs `O(iters * n * k * p)` and `n`
/// has to grow with `k` for the codewords to mean anything. At `bits*p = 16` that
/// is 65,536 centers over 2.6M samples -- hours per codebook, and the previous
/// fixed 50,000-sample budget "solved" it by asking k-means for more centers than
/// it had points, which returned a codebook whose tail was thousands of copies of
/// one training sample.
///
/// Twelve bits is where both hold: 4,096 centers over 163,840 samples trains in
/// seconds and gives every codeword its 40 points. In practice this only binds at
/// four bits, where the block drops from `p = 4` to `p = 2`; every lower rate keeps
/// the largest block the 16-bit index allows.
const MAX_CODEBOOK_BITS: usize = 12;

#[derive(Clone, Copy, PartialEq)]
enum Est {
    Ub,
    Bsm,
    Mse,
}

struct BVec {
    codes: Vec<u8>, // m block indices, each b·p bits, packed
    s: f32,         // per-vector scale S (variant-dependent)
    rnorm: f32,     // ‖u − c‖ for centered decomposition (1.0 ⇒ off)
    cidx: u32,      // assigned centroid index (0 ⇒ off)
}

/// BlockQuant oblivious block-sphere quantizer.
pub struct BlockQuantizer {
    dim: usize,
    #[allow(dead_code)] // rate in bits/coord; folded into blk_bits at construction
    bits: u8,
    p: usize,      // block size (divides dim)
    m: usize,      // number of blocks = dim/p
    blk_bits: u8,  // b·p — index width per block
    n_code: usize, // 2^(b·p) codewords per block
    est: Est,
    rotation: Rotor,
    /// Shared per-block codebook (same for every block — the marginal is identical),
    /// flat: `book[i*p .. i*p+p]`.
    book: Vec<f32>,
    centroids: Vec<Vec<f32>>,
    entries: Vec<(ItemId, BVec)>,
}

/// One standard normal via Box-Muller from a splitmix64 state.
fn next_normal(state: &mut u64) -> f32 {
    let u1 = ((splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64).max(1e-12);
    let u2 = (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64;
    ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
}

/// Sample `n` exact block-marginal vectors: u ~ Uniform(S^{dim−1}) via Gaussian
/// normalize, keep the first `p` coords. (Rotation-invariant ⇒ any p coords have
/// the marginal f_{p,dim}, so the first p are a faithful sample.)
fn sample_block_marginal(dim: usize, p: usize, n: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut st = seed;
    (0..n)
        .map(|_| {
            let g: Vec<f32> = (0..dim).map(|_| next_normal(&mut st)).collect();
            let nn = l2_norm(&g).max(f32::EPSILON);
            g[..p].iter().map(|&v| v / nn).collect()
        })
        .collect()
}

/// k-means++ costs `O(k n)` sequentially, one full pass per center. That is fine
/// while `k` is a few hundred and ruinous once the block codebook reaches 2^16
/// entries over a multi-million-sample training set, so above this product the
/// seeding falls back to a deterministic stride over distinct samples. Lloyd then
/// does the work; the marginal is smooth and unimodal, which is the regime where
/// the ++ guarantee buys least.
const KMEANS_PP_BUDGET: usize = 2_000_000_000;

/// k-means (Lloyd) on `samples` (each length p), `k` centers, seeded init.
/// Returns flat `k*p` codebook.
///
/// Every returned codeword is supported by training points, and no two codewords
/// use the same seed sample. These invariants keep the fitted codebook well formed.
fn kmeans_blocks(samples: &[Vec<f32>], p: usize, k: usize, iters: usize, seed: u64) -> Vec<f32> {
    use rayon::prelude::*;

    let n = samples.len();
    assert!(n >= k, "k-means asked for {k} centers from {n} samples");
    // Flat, contiguous copy: the assignment loop is the hot path and reads it
    // `k` times per sample.
    let flat: Vec<f32> = samples.iter().flat_map(|s| s.iter().copied()).collect();
    let point = |i: usize| &flat[i * p..i * p + p];
    let sqdist =
        |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum() };

    let mut st = seed;
    let mut centers: Vec<f32> = Vec::with_capacity(k * p);
    if k.saturating_mul(n) <= KMEANS_PP_BUDGET {
        // k-means++ init.
        let first = (splitmix64(&mut st) as usize) % n;
        centers.extend_from_slice(point(first));
        let mut d2: Vec<f32> = (0..n).map(|i| sqdist(point(i), point(first))).collect();
        for _ in 1..k {
            let total: f64 = d2.iter().map(|&x| x as f64).sum();
            // sample proportional to d2
            let mut target = (splitmix64(&mut st) >> 11) as f64 / (1u64 << 53) as f64 * total;
            let mut pick = n - 1;
            for (i, &w) in d2.iter().enumerate() {
                target -= w as f64;
                if target <= 0.0 {
                    pick = i;
                    break;
                }
            }
            centers.extend_from_slice(point(pick));
            let c: Vec<f32> = point(pick).to_vec();
            d2.par_iter_mut().enumerate().for_each(|(i, slot)| {
                let nd = sqdist(&flat[i * p..i * p + p], &c);
                if nd < *slot {
                    *slot = nd;
                }
            });
        }
    } else {
        // Deterministic stride over distinct samples. `n >= k` is asserted above,
        // so no index repeats and no two centers start identical.
        let stride = n / k;
        for c in 0..k {
            centers.extend_from_slice(point(c * stride));
        }
    }

    // Lloyd iterations.
    let mut assign = vec![0u32; n];
    for _ in 0..iters {
        assign.par_iter_mut().enumerate().for_each(|(i, slot)| {
            let s = &flat[i * p..i * p + p];
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for c in 0..k {
                let dd = sqdist(s, &centers[c * p..c * p + p]);
                if dd < best_d {
                    best_d = dd;
                    best = c;
                }
            }
            *slot = best as u32;
        });

        let (sums, cnt) = assign
            .par_iter()
            .enumerate()
            .fold(
                || (vec![0.0f64; k * p], vec![0u32; k]),
                |(mut sums, mut cnt), (i, &a)| {
                    let a = a as usize;
                    cnt[a] += 1;
                    for j in 0..p {
                        sums[a * p + j] += flat[i * p + j] as f64;
                    }
                    (sums, cnt)
                },
            )
            .reduce(
                || (vec![0.0f64; k * p], vec![0u32; k]),
                |(mut sa, mut ca), (sb, cb)| {
                    for (x, y) in sa.iter_mut().zip(&sb) {
                        *x += *y;
                    }
                    for (x, y) in ca.iter_mut().zip(&cb) {
                        *x += *y;
                    }
                    (sa, ca)
                },
            );

        for c in 0..k {
            if cnt[c] > 0 {
                for j in 0..p {
                    centers[c * p + j] = (sums[c * p + j] / cnt[c] as f64) as f32;
                }
            }
        }

        // Reseed starved codewords onto the worst-served training points, the way
        // pq.rs already does. An empty cluster is never repaired by Lloyd on its
        // own -- the assignment step breaks ties toward the lowest index, so a
        // duplicated center stays duplicated for every remaining iteration.
        let empty: Vec<usize> = (0..k).filter(|&c| cnt[c] == 0).collect();
        if !empty.is_empty() {
            let mut worst: Vec<(f32, usize)> = assign
                .par_iter()
                .enumerate()
                .map(|(i, &a)| {
                    let a = a as usize;
                    (
                        sqdist(&flat[i * p..i * p + p], &centers[a * p..a * p + p]),
                        i,
                    )
                })
                .collect();
            // Deterministic: distance descending, then sample index ascending.
            worst.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            for (slot, &c) in empty.iter().enumerate() {
                if let Some(&(_, i)) = worst.get(slot) {
                    centers[c * p..c * p + p].copy_from_slice(point(i));
                }
            }
        }
    }
    centers
}

impl BlockQuantizer {
    pub fn new(dim: usize, bits: u8) -> Self {
        let p = std::env::var("ULTRAVEC_BLOCKQUANT_P")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&p| p >= 1 && dim.is_multiple_of(p) && (bits as usize * p) <= 16)
            .unwrap_or_else(|| {
                // Largest p that divides dim and keeps the block index inside
                // MAX_CODEBOOK_BITS, so the codebook stays trainable.
                //
                // The candidate list is swept rather than fixed at {4,2,1}, which
                // would cap the block at 4 where the codebook budget admits far more: at one
                // bit p=8 is a 256-entry codebook, comfortably inside the 12-bit cap and
                // dividing every corpus dimension here, and at two bits p=6 gives 4096
                // entries on the 960- and 1536-dimensional corpora. Block-sphere
                // quantization's whole premise is that a wider block captures joint
                // geometry, so capping it at 4 under-serves the method at exactly the
                // low rates where the field is closest.
                (1..=MAX_CODEBOOK_BITS)
                    .rev()
                    .find(|&p| dim.is_multiple_of(p) && (bits as usize * p) <= MAX_CODEBOOK_BITS)
                    .unwrap_or(1)
            });
        // Estimator-axis control (the FAMILY widening of EDEN's `ULTRAVEC_EDEN_BIASED`):
        // the global `ULTRAVEC_BIASED_ESTIMATOR=1` flips the default UB → the biased
        // reconstruction-MSE-optimal BSM scale `S = ρ/‖x̄‖²` (Cauchy–Schwarz ⇒ negative
        // bias), holding the codebook + nearest-codeword assignment (hence g + recon-MSE)
        // byte-identical — only the per-vector scale changes. The explicit per-codec
        // `ULTRAVEC_BLOCKQUANT_EST=bsm|mse|ub` still wins when set.
        let global_biased = std::env::var("ULTRAVEC_BIASED_ESTIMATOR").as_deref() == Ok("1");
        let est = match std::env::var("ULTRAVEC_BLOCKQUANT_EST").as_deref() {
            Ok("bsm") => Est::Bsm,
            Ok("mse") => Est::Mse,
            Ok("ub") => Est::Ub,
            _ if global_biased => Est::Bsm,
            _ => Est::Ub,
        };
        let m = dim / p;
        let blk_bits = (bits as usize * p) as u8;
        let n_code = 1usize << blk_bits;
        let train_n = KMEANS_SAMPLES.max(MIN_SAMPLES_PER_CODEWORD * n_code);
        let samples = sample_block_marginal(dim, p, train_n, CODEBOOK_SEED);
        let book = kmeans_blocks(&samples, p, n_code, KMEANS_ITERS, CODEBOOK_SEED ^ 0x99);
        Self {
            dim,
            bits,
            p,
            m,
            blk_bits,
            n_code,
            est,
            rotation: Rotor::new_oblivious(dim, crate::rotation_seed()),
            book,
            centroids: Vec::new(),
            entries: Vec::new(),
        }
    }

    pub fn with_centroids(mut self, centroids: Vec<Vec<f32>>) -> Self {
        assert!(centroids.iter().all(|c| c.len() == self.dim));
        self.centroids = centroids;
        self
    }

    pub fn with_rotation_fit(mut self, sample: &[Vec<f32>]) -> Self {
        self.rotation = Rotor::fit(self.dim, crate::rotation_seed(), sample);
        self
    }

    fn assign_centroid(&self, u: &[f32]) -> usize {
        let mut best = 0usize;
        let mut best_dot = f32::NEG_INFINITY;
        for (j, c) in self.centroids.iter().enumerate() {
            let d: f32 = u.iter().zip(c).map(|(a, b)| a * b).sum();
            if d > best_dot {
                best_dot = d;
                best = j;
            }
        }
        best
    }

    fn prep(&self, o: &[f32]) -> (Vec<f32>, f32, u32) {
        let norm = l2_norm(o).max(f32::EPSILON);
        let u: Vec<f32> = o.iter().map(|v| v / norm).collect();
        if self.centroids.is_empty() {
            (u, 1.0, 0)
        } else {
            let j = self.assign_centroid(&u);
            let mut r: Vec<f32> = u
                .iter()
                .zip(&self.centroids[j])
                .map(|(a, c)| a - c)
                .collect();
            let rn = l2_norm(&r).max(f32::EPSILON);
            r.iter_mut().for_each(|x| *x /= rn);
            (r, rn, j as u32)
        }
    }

    /// Nearest codeword index for one p-D block.
    fn nearest(&self, blk: &[f32]) -> usize {
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for i in 0..self.n_code {
            let c = &self.book[i * self.p..i * self.p + self.p];
            let mut d = 0.0f32;
            for j in 0..self.p {
                let e = blk[j] - c[j];
                d += e * e;
            }
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        best
    }

    fn encode(&self, o: &[f32]) -> BVec {
        let (r, rnorm, cidx) = self.prep(o);
        let z = self.rotation.apply(&r); // z = R·u (unit), NOT standardized: codebook is on the sphere marginal
        let mut idx = vec![0u16; self.m];
        let mut rho = 0.0f32; // ρ = ⟨z, z̄⟩
        let mut zbar_norm2 = 0.0f32; // ‖z̄‖² = ‖x̄‖² (rotation preserves norm)
        for b in 0..self.m {
            let base = b * self.p;
            let blk = &z[base..base + self.p];
            let ci = self.nearest(blk);
            idx[b] = ci as u16;
            let c = &self.book[ci * self.p..ci * self.p + self.p];
            for j in 0..self.p {
                rho += blk[j] * c[j];
                zbar_norm2 += c[j] * c[j];
            }
        }
        let s = match self.est {
            Est::Ub => 1.0 / rho.max(1e-9),
            Est::Bsm => rho / zbar_norm2.max(1e-9),
            Est::Mse => 1.0,
        };
        BVec {
            codes: pack_indices(&idx, self.blk_bits),
            s,
            rnorm,
            cidx,
        }
    }

    fn score(&self, q_r: &[f32], qm: f32, bv: &BVec, scratch: &mut [u16]) -> f32 {
        unpack_indices(&bv.codes, self.blk_bits, &mut scratch[..self.m]);
        let mut dot = 0.0f32; // ⟨q_r, z̄⟩
        for b in 0..self.m {
            let c = &self.book[scratch[b] as usize * self.p..scratch[b] as usize * self.p + self.p];
            let base = b * self.p;
            for j in 0..self.p {
                dot += q_r[base + j] * c[j];
            }
        }
        qm + bv.rnorm * bv.s * dot
    }
}

impl VectorBackend for BlockQuantizer {
    fn dimensions(&self) -> usize {
        self.dim
    }
    fn len(&self) -> usize {
        self.entries.len()
    }
    fn add(&mut self, id: ItemId, embedding: &[f32]) {
        assert_eq!(embedding.len(), self.dim);
        let bv = self.encode(embedding);
        self.entries.push((id, bv));
    }
    fn add_batch(&mut self, embeddings: &[Vec<f32>]) {
        use rayon::prelude::*;
        let dim = self.dim;
        let this: &BlockQuantizer = self;
        let mut entries = embeddings
            .par_iter()
            .enumerate()
            .map(|(i, embedding)| {
                assert_eq!(embedding.len(), dim);
                (i as ItemId, this.encode(embedding))
            })
            .collect();
        self.entries.append(&mut entries);
    }
    fn search(&self, query: &[f32], limit: usize) -> Vec<(ItemId, f32)> {
        let qn = l2_norm(query);
        if qn < f32::EPSILON {
            return Vec::new();
        }
        let u: Vec<f32> = query.iter().map(|v| v / qn).collect();
        let qc: Vec<f32> = self
            .centroids
            .iter()
            .map(|c| u.iter().zip(c).map(|(a, b)| a * b).sum())
            .collect();
        let q_r = self.rotation.apply(&u);
        let mut scratch = vec![0u16; self.m];
        let mut results: Vec<(ItemId, f32)> = self
            .entries
            .iter()
            .map(|(id, bv)| {
                let qm = qc.get(bv.cidx as usize).copied().unwrap_or(0.0);
                (*id, self.score(&q_r, qm, bv, &mut scratch))
            })
            .collect();
        results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }
    fn mem_bytes(&self) -> usize {
        // codes (dim·bits bits) + scale(4). No start state. The per-vector centroid
        // state is real whenever centering is on, contrary to what this said before.
        let center = crate::centering_bytes(self.centroids.len());
        self.entries
            .iter()
            .map(|(_, bv)| bv.codes.len() + 4 + center)
            .sum()
    }
    fn memory_breakdown(&self) -> MemoryBreakdown {
        MemoryBreakdown {
            code_bytes: self.mem_bytes(),
            model_bytes: self.rotation.allocated_bytes()
                + self.book.capacity() * std::mem::size_of::<f32>()
                + self
                    .centroids
                    .iter()
                    .map(|v| v.capacity() * std::mem::size_of::<f32>())
                    .sum::<usize>(),
            ..MemoryBreakdown::default()
        }
    }
    fn is_approximate(&self) -> bool {
        true
    }
    fn reconstruct_unit(&self, x: &[f32]) -> Option<Vec<f32>> {
        let bv = self.encode(x);
        let mut scratch = vec![0u16; self.m];
        unpack_indices(&bv.codes, self.blk_bits, &mut scratch);
        let mut zbar = vec![0.0f32; self.dim];
        for b in 0..self.m {
            let c = &self.book[scratch[b] as usize * self.p..scratch[b] as usize * self.p + self.p];
            for j in 0..self.p {
                zbar[b * self.p + j] = c[j];
            }
        }
        let mut recon = self.rotation.apply_inverse(&zbar);
        let n = l2_norm(&recon).max(f32::EPSILON);
        recon.iter_mut().for_each(|v| *v /= n);
        if self.centroids.is_empty() {
            Some(recon)
        } else {
            let c = &self.centroids[bv.cidx as usize];
            let mut full: Vec<f32> = c
                .iter()
                .zip(&recon)
                .map(|(cc, rr)| cc + bv.rnorm * rr)
                .collect();
            let nf = l2_norm(&full).max(f32::EPSILON);
            full.iter_mut().for_each(|v| *v /= nf);
            Some(full)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codebook_sized_correctly() {
        // dim 128: default p=4, b=2 → 256 codewords × 4 dims.
        let q = BlockQuantizer::new(128, 2);
        assert_eq!(q.p, 4);
        assert_eq!(q.m, 32);
        assert_eq!(q.n_code, 256);
        assert_eq!(q.book.len(), 256 * 4);
        // codewords should be inside the unit ball (block marginal of a sphere).
        for i in 0..q.n_code {
            let c = &q.book[i * 4..i * 4 + 4];
            let nrm: f32 = c.iter().map(|v| v * v).sum();
            assert!(
                nrm <= 1.5,
                "codeword norm² {nrm} implausibly large for a sphere-ball marginal"
            );
        }
    }

    #[test]
    fn block_marginal_samples_in_unit_ball() {
        let s = sample_block_marginal(64, 4, 1000, 7);
        for v in &s {
            let n2: f32 = v.iter().map(|x| x * x).sum();
            assert!(n2 <= 1.0 + 1e-5, "block-marginal sample norm² {n2} > 1");
        }
    }

    #[test]
    fn blockquant_reconstruction_correlates_with_truth() {
        let mut st = 17u64;
        let dim = 128;
        let q = BlockQuantizer::new(dim, 2);
        let mut sg = 0.0f64;
        let n = 200;
        for _ in 0..n {
            let mut o: Vec<f32> = (0..dim)
                .map(|_| crate::next_f64(&mut st) as f32 - 0.5)
                .collect();
            let nn = l2_norm(&o).max(f32::EPSILON);
            o.iter_mut().for_each(|x| *x /= nn);
            let obar = q.reconstruct_unit(&o).unwrap();
            let g: f32 = o.iter().zip(&obar).map(|(a, b)| a * b).sum();
            sg += g as f64;
        }
        assert!(
            sg / n as f64 > 0.8,
            "mean g {} too low for 2-bit BlockQuant",
            sg / n as f64
        );
    }

    #[test]
    fn blockquant_self_query_retrieves_self() {
        let mut st = 23u64;
        let dim = 64;
        let mut q = BlockQuantizer::new(dim, 2);
        let mut db: Vec<Vec<f32>> = Vec::new();
        for i in 0..50 {
            let mut v: Vec<f32> = (0..dim)
                .map(|_| crate::next_f64(&mut st) as f32 - 0.5)
                .collect();
            let nn = l2_norm(&v).max(f32::EPSILON);
            v.iter_mut().for_each(|x| *x /= nn);
            q.add(i as ItemId, &v);
            db.push(v);
        }
        let mut hits = 0;
        for (i, v) in db.iter().enumerate() {
            if q.search(v, 1)[0].0 == i as ItemId {
                hits += 1;
            }
        }
        assert!(
            hits >= 38,
            "BlockQuant 2-bit self-query hits {hits}/50 too low"
        );
    }
}
