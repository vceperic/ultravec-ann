//! Pyramid Vector Quantization (Fischer 1986; the CELT/Opus spherical code), a
//! decode-free comparison to the trellis. PVQ's codeword
//! is a sparse small-integer pulse vector `p` with `Σ|p_i| = K`, so the inner
//! product is `⟨q_r, p⟩ = Σ_j q_r[idx_j]·val_j` — an O(nnz) sparse-integer gather
//! with no state machine. It is codebook-free and training-free, and reuses the
//! same rotation, rescaled estimator, and centering for a controlled comparison.

use crate::{l2_norm, ItemId, MemoryBreakdown, Rotor, VectorBackend};

/// Number of signed integer vectors of length `b` with `Σ|p_i| = k` (the PVQ
/// codebook size; includes signs). Recurrence N(b,k)=N(b-1,k)+N(b,k-1)+N(b-1,k-1).
fn pvq_count(b: usize, k: usize) -> f64 {
    let mut prev = vec![0.0f64; k + 1]; // N(0,·): 1 at k=0, else 0
    prev[0] = 1.0;
    for _ in 1..=b {
        let mut cur = vec![0.0f64; k + 1];
        cur[0] = 1.0;
        for kk in 1..=k {
            cur[kk] = prev[kk] + cur[kk - 1] + prev[kk - 1];
        }
        prev = cur;
    }
    prev[k]
}

/// Bits to enumerate a length-`b` PVQ block at `k` pulses.
fn block_bits(b: usize, k: usize) -> f64 {
    pvq_count(b, k).max(1.0).log2()
}

/// Smallest `k` whose per-dim rate `log2 N(b,k)/b` reaches `target` bits/dim.
fn choose_k(b: usize, target: f64) -> usize {
    for k in 1..=(b * 8) {
        if block_bits(b, k) / b as f64 >= target {
            return k;
        }
    }
    b * 8
}

/// Achieved rate in bits/dimension for a block of size `b` at `k` pulses.
fn achieved_rate(b: usize, k: usize) -> f64 {
    block_bits(b, k) / b as f64
}

/// Block sizes considered when nothing is pinned, largest first.
const BLOCK_CANDIDATES: [usize; 4] = [64, 32, 16, 8];

/// Pick the block size whose pulse enumeration lands closest to `target` from above.
///
/// PVQ can only emit rates the enumeration `N(b,k)` actually reaches, and `choose_k`
/// rounds up, so the block size decides how much rate is overspent. At a fixed block
/// of 8 the one-bit setting had no `k` near 1.000 -- it emitted 1.178 bits/dimension,
/// 17.8% over the label, in the very cells where the codec is strongest. Larger blocks
/// have finer granularity: the same target costs 1.013 bits/dimension at block 64.
/// Encoding cost is `dim * k` per vector and so does not depend on the block size.
///
/// Deterministic: minimize the SERIALIZED BYTES, then prefer the larger block.
///
/// Minimizing the achieved rate directly, as this did, is the wrong objective whenever
/// two blocks serialize to the same record. At two bits on a 128-dimensional corpus,
/// block 8 achieves 2.0079 bits/dimension and block 64 achieves 2.0210 -- so a
/// rate-minimizing rule picks block 8, even though `mem_bytes` rounds both to the same
/// 37 bytes. That trade is all cost and no benefit: PVQ's shaping gain grows with block
/// dimension, so the smaller block is strictly worse at an identical record, and its
/// pulse count is higher (112 non-zeros against 88 at dim 128), which also makes the
/// scan slower. Rate honesty is only worth having where it buys a byte.
fn choose_block(dim: usize, target: f64) -> usize {
    BLOCK_CANDIDATES
        .into_iter()
        .filter(|&b| b <= dim && dim.is_multiple_of(b))
        .map(|b| {
            let rate = achieved_rate(b, choose_k(b, target));
            // Same formula `mem_bytes` uses, so the comparison is on the record that
            // is actually reported rather than on a proxy for it.
            let bytes =
                (dim.div_ceil(b) as f64 * block_bits(b, choose_k(b, target)) / 8.0).ceil() as usize;
            (bytes, rate, b)
        })
        .min_by(|a, c| {
            a.0.cmp(&c.0)
                // Larger block wins at equal bytes: more shaping for the same record.
                .then(c.2.cmp(&a.2))
        })
        .map(|(_, _, b)| b)
        .unwrap_or_else(|| 8.min(dim))
}

struct PvqVec {
    /// Sparse pulses as parallel arrays: coordinate index and signed pulse count,
    /// `nnz ≤ K·(D/b)` entries each. Split rather than `Vec<(u16, i32)>` so the
    /// query scan can gather eight indices at a time; the interleaved form also
    /// padded each pair to 8 bytes, so this stores 6 and is the smaller of the two.
    idx: Vec<u16>,
    val: Vec<f32>,
    rescale: f32, // ⟨p, u_r⟩ (rescaled-estimator denominator)
    rnorm: f32,
    cidx: u32,
}

pub struct PvqQuantizer {
    dim: usize,
    block: usize,
    k: usize,
    block_bits: f64,
    rotation: Rotor,
    centroids: Vec<Vec<f32>>,
    entries: Vec<(ItemId, PvqVec)>,
}

impl PvqQuantizer {
    pub fn new(dim: usize, bits: u8) -> Self {
        let block = std::env::var("ULTRAVEC_PVQ_BLOCK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| choose_block(dim, bits as f64))
            .min(dim);
        let k = std::env::var("ULTRAVEC_PVQ_K")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| choose_k(block, bits as f64));
        Self {
            dim,
            block,
            k,
            block_bits: block_bits(block, k),
            rotation: Rotor::new_oblivious(dim, crate::rotation_seed()),
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

    fn assign(&self, u: &[f32]) -> usize {
        let mut best = 0usize;
        let mut bd = f32::NEG_INFINITY;
        for (j, c) in self.centroids.iter().enumerate() {
            let d: f32 = u.iter().zip(c).map(|(a, b)| a * b).sum();
            if d > bd {
                bd = d;
                best = j;
            }
        }
        best
    }

    /// Unit vector or its UNIT residual to the assigned centroid (same as RaBitQ).
    fn prep(&self, o: &[f32]) -> (Vec<f32>, f32, u32) {
        let norm = l2_norm(o).max(f32::EPSILON);
        let u: Vec<f32> = o.iter().map(|v| v / norm).collect();
        if self.centroids.is_empty() {
            (u, 1.0, 0)
        } else {
            let j = self.assign(&u);
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

    /// Greedy PVQ projection of one block to `k` pulses: initial floor allocation
    /// then add pulses one at a time to the coordinate that maximizes cos²
    /// (`⟨|x|,mag⟩²/‖mag‖²`). Signs follow `x`. O(b·k).
    fn encode_block(x: &[f32], k: usize) -> Vec<i32> {
        let b = x.len();
        let absx: Vec<f32> = x.iter().map(|v| v.abs()).collect();
        let s: f32 = absx.iter().sum();
        let mut mag = vec![0i32; b];
        if s < 1e-12 {
            return mag;
        }
        let mut placed = 0i32;
        let mut dotp = 0.0f32;
        let mut norm2 = 0.0f32;
        for i in 0..b {
            let yi = (k as f32 * absx[i] / s).floor() as i32;
            mag[i] = yi;
            placed += yi;
            dotp += yi as f32 * absx[i];
            norm2 += (yi * yi) as f32;
        }
        while placed < k as i32 {
            let mut best = 0usize;
            let mut bestv = f32::NEG_INFINITY;
            for i in 0..b {
                let nd = dotp + absx[i];
                let nn = norm2 + 2.0 * mag[i] as f32 + 1.0;
                let v = nd * nd / nn;
                if v > bestv {
                    bestv = v;
                    best = i;
                }
            }
            dotp += absx[best];
            norm2 += 2.0 * mag[best] as f32 + 1.0;
            mag[best] += 1;
            placed += 1;
        }
        for i in 0..b {
            if x[i] < 0.0 {
                mag[i] = -mag[i];
            }
        }
        mag
    }

    fn encode(&self, o: &[f32]) -> PvqVec {
        let (r, rnorm, cidx) = self.prep(o);
        let u_r = self.rotation.apply(&r);
        let mut idx: Vec<u16> = Vec::new();
        let mut val: Vec<f32> = Vec::new();
        let mut rescale = 0.0f32;
        let mut blk = 0;
        while blk < self.dim {
            let end = (blk + self.block).min(self.dim);
            let p = Self::encode_block(&u_r[blk..end], self.k);
            for (j, &pv) in p.iter().enumerate() {
                if pv != 0 {
                    idx.push((blk + j) as u16);
                    val.push(pv as f32);
                    rescale += pv as f32 * u_r[blk + j];
                }
            }
            blk += self.block;
        }
        // Same degenerate-vector guard as RaBitQ/trellis (∞ ⇒ score 0).
        let rescale = if rescale > 1e-6 {
            rescale
        } else {
            f32::INFINITY
        };
        PvqVec {
            idx,
            val,
            rescale,
            rnorm,
            cidx,
        }
    }

    /// Decode-free score: O(nnz) sparse gather, no sequential state.
    ///
    /// The shared gather kernel applies directly with the roles swapped -- here the
    /// indices address the query rather than a reconstruction table -- so PVQ's scan
    /// is vectorized by the same code as the dense codecs' rather than by a
    /// hand-rolled variant.
    fn score(&self, q_r: &[f32], qm: f32, pv: &PvqVec) -> f32 {
        let dot = crate::simd::lut_dot(&pv.val, &pv.idx, q_r);
        qm + pv.rnorm * (dot / pv.rescale)
    }

    /// Average pulses-per-vector (nnz) — the decode-free scan cost (cf. trellis D
    /// sequential decode steps, RaBitQ D dense mul-adds).
    pub fn avg_nnz(&self) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }
        self.entries.iter().map(|(_, v)| v.idx.len()).sum::<usize>() as f64
            / self.entries.len() as f64
    }
}

impl VectorBackend for PvqQuantizer {
    fn dimensions(&self) -> usize {
        self.dim
    }
    fn len(&self) -> usize {
        self.entries.len()
    }
    fn add(&mut self, id: ItemId, embedding: &[f32]) {
        assert_eq!(embedding.len(), self.dim);
        let pv = self.encode(embedding);
        self.entries.push((id, pv));
    }
    fn reserve(&mut self, additional: usize) {
        self.entries.reserve_exact(additional);
    }
    fn add_batch(&mut self, embeddings: &[Vec<f32>]) {
        use rayon::prelude::*;
        let dim = self.dim;
        let this: &PvqQuantizer = self;
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
        let mut results: Vec<(ItemId, f32)> = self
            .entries
            .iter()
            .map(|(id, pv)| {
                let qm = qc.get(pv.cidx as usize).copied().unwrap_or(0.0);
                (*id, self.score(&q_r, qm, pv))
            })
            .collect();
        results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }
    fn mem_bytes(&self) -> usize {
        // Theoretical enumeration bytes: (D/b) blocks × log2 N(b,k) bits + rescale.
        let n_blocks = self.dim.div_ceil(self.block);
        let code_bytes = (n_blocks as f64 * self.block_bits / 8.0).ceil() as usize;
        self.entries.len() * (code_bytes + 4 + crate::centering_bytes(self.centroids.len()))
    }
    fn memory_breakdown(&self) -> MemoryBreakdown {
        MemoryBreakdown {
            code_bytes: self.mem_bytes(),
            model_bytes: self.rotation.allocated_bytes()
                + self
                    .centroids
                    .iter()
                    .map(|v| v.capacity() * std::mem::size_of::<f32>())
                    .sum::<usize>(),
            index_bytes: self.entries.capacity() * std::mem::size_of::<(ItemId, PvqVec)>(),
            ..MemoryBreakdown::default()
        }
    }
    fn is_approximate(&self) -> bool {
        true
    }
    fn reconstruct_unit(&self, x: &[f32]) -> Option<Vec<f32>> {
        let pv = self.encode(x);
        let mut xbar = vec![0.0f32; self.dim];
        for (&i, &v) in pv.idx.iter().zip(&pv.val) {
            xbar[i as usize] = v;
        }
        let mut recon = self.rotation.apply_inverse(&xbar);
        let n = l2_norm(&recon).max(f32::EPSILON);
        recon.iter_mut().for_each(|v| *v /= n);
        if self.centroids.is_empty() {
            Some(recon)
        } else {
            let c = &self.centroids[pv.cidx as usize];
            let mut full: Vec<f32> = c
                .iter()
                .zip(&recon)
                .map(|(cc, rr)| cc + pv.rnorm * rr)
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
    use crate::{cosine, next_f64};

    fn rand_unit(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut st = seed;
        (0..n)
            .map(|_| {
                let mut v: Vec<f32> = (0..dim)
                    .map(|_| {
                        let u1 = next_f64(&mut st).max(1e-12);
                        let u2 = next_f64(&mut st);
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
    fn pvq_count_recurrence() {
        // N(2,1)=4 (±e0, ±e1); N(2,2)=8 ((±2,0),(0,±2),(±1,±1)).
        assert_eq!(pvq_count(2, 1) as i64, 4);
        assert_eq!(pvq_count(2, 2) as i64, 8);
    }

    #[test]
    fn encode_block_sums_to_k() {
        let x = vec![0.5f32, -0.3, 0.1, 0.8, -0.2, 0.0, 0.4, -0.6];
        for k in [4usize, 9, 16] {
            let p = PvqQuantizer::encode_block(&x, k);
            let s: i32 = p.iter().map(|v| v.abs()).sum();
            assert_eq!(s, k as i32, "block pulses must sum to k");
        }
    }

    #[test]
    fn pvq_self_query_top() {
        let dim = 128;
        let mut q = PvqQuantizer::new(dim, 4);
        let db = rand_unit(300, dim, 13);
        for (i, v) in db.iter().enumerate() {
            q.add(i as ItemId, v);
        }
        assert_eq!(
            q.search(&db[0], 1)[0].0,
            0,
            "self-query should retrieve self"
        );
        let _ = cosine(&db[0], &db[0]);
    }
}
