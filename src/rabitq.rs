//! RaBitQ (SIGMOD 2024) and Extended-RaBitQ (SIGMOD 2025) implementations used
//! as data-oblivious comparison codecs. The manuscript discusses their relation
//! to TurboQuant and the scope of this artifact-native implementation.
//!
//! Core idea (oblivious): unit-normalize → random orthogonal rotation (our
//! sign-flip FWHT `Rotation`) → quantize each coord to a small grid → store the
//! codes plus one scalar `⟨x̄, u_r⟩`, and estimate the inner product with the
//! **unbiased rescaled estimator**
//!     ⟨q̄, ō⟩ ≈ ⟨q̄_r, x̄⟩ / ⟨x̄, u_r⟩
//! where `x̄` is the (centered-integer) reconstructed code, `u_r` the rotated unit
//! data, and `q̄_r` the rotated unit query. `bits=1` is canonical RaBitQ (sign
//! code); `bits>1` is the
//! Extended-RaBitQ uniform-grid form with a per-vector rescale-factor search.

use crate::{l2_norm, pack_indices, unpack_indices, ItemId, MemoryBreakdown, Rotor, VectorBackend};

struct RbqVec {
    codes: Vec<u8>,
    /// Per-vector estimator multiplier applied to `⟨q̄_r, x̄⟩`. In the default
    /// (unbiased) variant this is `1/⟨x̄,u_r⟩` (so `mul·dot` is the unbiased
    /// rescaled estimator); in the biased control it is the recon-MSE-optimal
    /// `⟨x̄,u_r⟩/⟨x̄,x̄⟩` (Cauchy–Schwarz ⇒ undershoots ⇒ negative bias). The codes
    /// (hence g + recon-MSE) are byte-identical between the two — only this scalar
    /// differs, isolating estimator variance from reconstruction.
    mul: f32,
    /// `‖u − c‖` — residual norm for the centered decomposition (1.0 when
    /// centering is off). cosine ≈ ⟨q,c⟩ + rnorm·(mul·dot).
    rnorm: f32,
    /// Index of the assigned centroid (0 when centering off / single global mean).
    cidx: u32,
}

pub struct RaBitQ {
    dim: usize,
    bits: u8,
    // u32, not u16: --bits accepts up to 16, and `1u16 << 16` overflows -- a debug
    // panic, and in release a mask to `<< 0` giving levels = 1 and a silently
    // degenerate all-zero codebook.
    levels: u32,
    /// Estimator-axis control shared with EDEN's `ULTRAVEC_EDEN_BIASED` option.
    /// `false` (default) selects the RaBitQ unbiased rescaled estimator
    /// (multiplier `1/⟨x̄,u_r⟩`, bias≈0). `true` = the biased reconstruction-MSE-optimal
    /// scale `⟨x̄,u_r⟩/⟨x̄,x̄⟩` (negative bias). Set via `ULTRAVEC_BIASED_ESTIMATOR=1`
    /// (the global family flag) or `ULTRAVEC_RABITQ_BIASED=1`. The codebook and
    /// reconstruction are identical between the two settings; only score scale differs.
    biased: bool,
    rotation: Rotor,
    /// Coarse centroids of the unit vectors for RaBitQ-style centering (empty ⇒
    /// off). 1 entry = single global mean; N entries = IVF-style per-cluster
    /// centroids. Each vector quantizes the residual `u − c_assigned` and keeps
    /// the `⟨q,c⟩` term exact — the load-bearing step RaBitQ ships with.
    centroids: Vec<Vec<f32>>,
    /// `centered()` materialized as a `levels`-entry table, so the query scan is a
    /// gather-and-FMA through `simd::lut_dot` instead of a scalar loop recomputing
    /// the same affine map per coordinate. Purely a scoring accelerant: it holds
    /// exactly the values `centered()` returns, and the encoder still uses
    /// `centered()` directly.
    book: Vec<f32>,
    entries: Vec<(ItemId, RbqVec)>,
}

impl RaBitQ {
    pub fn new(dim: usize, bits: u8) -> Self {
        let levels = 1u32 << bits;
        Self {
            dim,
            bits,
            levels,
            biased: std::env::var("ULTRAVEC_BIASED_ESTIMATOR").as_deref() == Ok("1")
                || std::env::var("ULTRAVEC_RABITQ_BIASED").as_deref() == Ok("1"),
            rotation: Rotor::new_oblivious(dim, crate::rotation_seed()),
            centroids: Vec::new(),
            book: (0..levels)
                .map(|c| c as f32 - (levels as f32 - 1.0) * 0.5)
                .collect(),
            entries: Vec::new(),
        }
    }

    /// Enable centering with a single precomputed global mean of the unit vectors.
    pub fn with_mean(self, mean: Vec<f32>) -> Self {
        self.with_centroids(vec![mean])
    }

    /// Enable IVF-style per-cluster centering with `centroids` (each `dim`-long).
    /// Each vector is assigned to its nearest centroid by cosine; the residual to
    /// that centroid is quantized and the `⟨q,c⟩` term kept exact.
    pub fn with_centroids(mut self, centroids: Vec<Vec<f32>>) -> Self {
        assert!(centroids.iter().all(|c| c.len() == self.dim));
        self.centroids = centroids;
        self
    }

    /// Replace the rotation with a learned one fit on `sample` when
    /// `ULTRAVEC_ROTATION` is pca/itq (no-op for the oblivious default).
    pub fn with_rotation_fit(mut self, sample: &[Vec<f32>]) -> Self {
        self.rotation = Rotor::fit(self.dim, crate::rotation_seed(), sample);
        self
    }

    /// Nearest centroid index for unit vector `u` (max cosine = max dot, unit u).
    fn assign(&self, u: &[f32]) -> usize {
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

    /// Centered value of an integer code (so the grid straddles 0).
    #[inline]
    fn centered(&self, code: u16) -> f32 {
        code as f32 - (self.levels as f32 - 1.0) * 0.5
    }

    /// Quantize the rotated unit vector `u_r` to a `range`-clipped uniform grid;
    /// return (codes, x̄·u_r dot, x̄·x̄, reconstruction error of the rescaled fit).
    fn quantize_at(&self, u_r: &[f32], range: f32) -> (Vec<u16>, f32, f32, f32) {
        let lmax = (self.levels - 1) as f32;
        let mut codes = vec![0u16; self.dim];
        // dot = <x̄, u_r>, x2 = <x̄, x̄>
        let mut dot = 0.0f32;
        let mut x2 = 0.0f32;
        let mut ur2 = 0.0f32;
        for i in 0..self.dim {
            let t = ((u_r[i] / range + 1.0) * 0.5 * lmax)
                .round()
                .clamp(0.0, lmax);
            let code = t as u16;
            codes[i] = code;
            let xb = self.centered(code);
            dot += xb * u_r[i];
            x2 += xb * xb;
            ur2 += u_r[i] * u_r[i];
        }
        // Rescaled-fit residual: ‖u_r − s·x̄‖² with optimal s = dot/x2.
        let err = if x2 > 0.0 { ur2 - dot * dot / x2 } else { ur2 };
        (codes, dot, x2, err)
    }

    /// The per-vector estimator multiplier on `⟨q̄_r, x̄⟩`, given `dot=⟨x̄,u_r⟩` and
    /// `x2=⟨x̄,x̄⟩`. Unbiased (default): `1/dot`. Biased control: `dot/x2` (the
    /// recon-MSE-optimal scale; same codes ⇒ same g + MSE, only this scalar moves).
    /// A zero or near-zero source can have dot≈0 with a nonzero constant code;
    /// the guard maps that case to a zero score multiplier. The biased scale
    /// dot/x2 is already ≈0 there and needs no sentinel.
    #[inline]
    fn estimator_mul(&self, dot: f32, x2: f32) -> f32 {
        if self.biased {
            dot / x2.max(1e-12)
        } else if dot > 1e-6 {
            1.0 / dot
        } else {
            0.0
        }
    }

    fn encode(&self, o: &[f32]) -> RbqVec {
        let norm = l2_norm(o).max(f32::EPSILON);
        let u: Vec<f32> = o.iter().map(|v| v / norm).collect();
        // Centering: quantize the UNIT residual to the assigned centroid; keep
        // the ⟨q,c⟩ term exact at score time. dot/rescale ≈ ⟨q,r_unit⟩ and
        // score = qm + rnorm·that = ⟨q,u⟩.
        let (r, rnorm, cidx) = if self.centroids.is_empty() {
            (u, 1.0f32, 0u32)
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
        };
        let u_r = self.rotation.apply(&r);

        if self.bits == 1 {
            // Canonical 1-bit RaBitQ: a sign code.
            //
            // The reconstruction convention has to be the decoder's. Every consumer
            // -- `score` and `reconstruct_unit` -- decodes through
            // `centered`, which at two levels yields x̄_i = ±0.5, so ⟨x̄,x̄⟩ = dim/4.
            // Deriving `dot` and `x2` from an assumed ±1 instead made the estimator
            // exactly half the rescaled value. That is a positive global constant, so
            // it never moved a ranking, but it does scale the estimator's bias and
            // standard-deviation diagnostics, and under centering it would have
            // combined an exact ⟨q,c⟩ term with a residual term twice too small.
            let mut codes = vec![0u16; self.dim];
            let mut dot = 0.0f32;
            for i in 0..self.dim {
                let bit = (u_r[i] > 0.0) as u16;
                codes[i] = bit;
                dot += self.centered(bit) * u_r[i]; // = 0.5 * Σ|u_r[i]|
            }
            let mul = self.estimator_mul(dot, self.dim as f32 * 0.25);
            return RbqVec {
                codes: pack_indices(&codes, 1),
                mul,
                rnorm,
                cidx,
            };
        }

        // Extended-RaBitQ: per-vector rescale-factor search over clip ranges
        // (multiples of the coord std), pick the min rescaled-fit residual. The
        // Codebook selection uses the reconstruction-fit `err` in both estimator
        // settings, so g and reconstruction MSE are byte-identical; only
        // the score-time multiplier below differs.
        let sigma = (u_r.iter().map(|v| v * v).sum::<f32>() / self.dim as f32).sqrt();
        let mut best: Option<(Vec<u16>, f32, f32, f32)> = None;
        // Dense rescale-factor search (Extended-RaBitQ picks the clip range
        // maximizing cosine = minimizing 1−cos², which `quantize_at` returns as
        // `err`). A 0.1-spaced grid covers the multi-bit clip-range search.
        let mut k = 0.6f32;
        while k <= 5.0 {
            let range = (sigma * k).max(1e-6);
            let (codes, dot, x2, err) = self.quantize_at(&u_r, range);
            if best.as_ref().map(|b| err < b.3).unwrap_or(true) {
                best = Some((codes, dot, x2, err));
            }
            k += 0.1;
        }
        let (codes, dot, x2, _) = best.unwrap();
        let mul = self.estimator_mul(dot, x2);
        RbqVec {
            codes: pack_indices(&codes, self.bits),
            mul,
            rnorm,
            cidx,
        }
    }

    fn score(&self, q_r: &[f32], qm: f32, rv: &RbqVec, scratch: &mut [u16]) -> f32 {
        unpack_indices(&rv.codes, self.bits, scratch);
        // Gather-and-FMA through the shared kernel rather than a scalar loop. The
        // codes index a `levels`-entry table, so this is the same arithmetic the
        // scalar form performed, vectorized -- and, more to the point, at the same
        // optimization level as the trellis scan it is compared against. Relative
        // QPS is a reported result; scoring one codec 8 lanes wide and its closest
        // comparator one lane at a time measures the harness, not the methods.
        let dot = crate::simd::lut_dot(q_r, &scratch[..q_r.len()], &self.book);
        // Rescaled estimator of the residual cosine (mul = 1/⟨x̄,u_r⟩ unbiased, or
        // the biased ⟨x̄,u_r⟩/⟨x̄,x̄⟩ control), recomposed with the exact mean term:
        // ⟨q,u⟩ = ⟨q,mean⟩ + ‖u−mean‖·⟨q,residual_unit⟩. (qm=0, rnorm=1 ⇒ the plain
        // uncentered estimator.)
        qm + rv.rnorm * (rv.mul * dot)
    }
}

impl VectorBackend for RaBitQ {
    fn dimensions(&self) -> usize {
        self.dim
    }
    fn len(&self) -> usize {
        self.entries.len()
    }
    fn add(&mut self, id: ItemId, embedding: &[f32]) {
        assert_eq!(embedding.len(), self.dim);
        let rv = self.encode(embedding);
        self.entries.push((id, rv));
    }
    fn reserve(&mut self, additional: usize) {
        self.entries.reserve_exact(additional);
    }
    fn add_batch(&mut self, embeddings: &[Vec<f32>]) {
        use rayon::prelude::*;
        let dim = self.dim;
        let this: &RaBitQ = self;
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
        // Query stays UNcentered; the exact term ⟨q,c⟩ for each vector's assigned
        // centroid is added per entry. Precompute ⟨q,c_j⟩ for every centroid once.
        let qc: Vec<f32> = self
            .centroids
            .iter()
            .map(|c| u.iter().zip(c).map(|(a, b)| a * b).sum())
            .collect();
        let q_r = self.rotation.apply(&u);
        let mut scratch = vec![0u16; self.dim];
        let mut results: Vec<(ItemId, f32)> = self
            .entries
            .iter()
            .map(|(id, rv)| {
                let qm = qc.get(rv.cidx as usize).copied().unwrap_or(0.0);
                (*id, self.score(&q_r, qm, rv, &mut scratch))
            })
            .collect();
        results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }
    fn mem_bytes(&self) -> usize {
        // codes + the 4-byte rescale scalar (no separate norm — cosine), plus the
        // per-vector state centering needs (see `crate::centering_bytes`).
        let center = crate::centering_bytes(self.centroids.len());
        self.entries
            .iter()
            .map(|(_, rv)| rv.codes.len() + 4 + center)
            .sum()
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
            index_bytes: self.entries.capacity() * std::mem::size_of::<(ItemId, RbqVec)>(),
            ..MemoryBreakdown::default()
        }
    }
    fn is_approximate(&self) -> bool {
        true
    }
    /// Reconstruct the unit direction `ō` of `x` (for the rank-distortion MSE
    /// column and the estimator-variance diagnostic). `g = ⟨ō, o⟩` — the cosine
    /// of the reconstruction to the truth — is the single quantity the encoder
    /// controls; the rescaled-estimator variance scales as √((1−g²)/g²)/√(D−1).
    fn reconstruct_unit(&self, x: &[f32]) -> Option<Vec<f32>> {
        let rv = self.encode(x);
        let mut scratch = vec![0u16; self.dim];
        unpack_indices(&rv.codes, self.bits, &mut scratch);
        let xbar: Vec<f32> = scratch.iter().map(|&c| self.centered(c)).collect();
        let mut recon = self.rotation.apply_inverse(&xbar); // residual direction in orig space
        let n = l2_norm(&recon).max(f32::EPSILON);
        recon.iter_mut().for_each(|v| *v /= n);
        if self.centroids.is_empty() {
            Some(recon)
        } else {
            // u ≈ c + ‖u−c‖·residual_unit; recompose then renormalize.
            let c = &self.centroids[rv.cidx as usize];
            let mut full: Vec<f32> = c
                .iter()
                .zip(&recon)
                .map(|(cc, rr)| cc + rv.rnorm * rr)
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

    /// The RaBitQ rescaled estimator must be ~unbiased for the cosine.
    #[test]
    fn rabitq_estimator_is_unbiased() {
        for bits in [1u8, 4] {
            let dim = 256;
            let rq = RaBitQ::new(dim, bits);
            let db = rand_unit(200, dim, 3);
            let queries = rand_unit(50, dim, 4);
            let mut bias = 0.0f64;
            let mut n = 0;
            for q in &queries {
                let qn = l2_norm(q);
                let u: Vec<f32> = q.iter().map(|v| v / qn).collect();
                let q_r = rq.rotation.apply(&u);
                let mut scratch = vec![0u16; dim];
                for v in &db {
                    let rv = rq.encode(v);
                    let est = rq.score(&q_r, 0.0, &rv, &mut scratch);
                    bias += (est - cosine(q, v)) as f64;
                    n += 1;
                }
            }
            let mean_bias = bias / n as f64;
            assert!(
                mean_bias.abs() < 0.03,
                "{bits}-bit RaBitQ mean bias {mean_bias}"
            );
        }
    }

    #[test]
    fn rabitq_self_query_top() {
        let dim = 128;
        let mut rq = RaBitQ::new(dim, 4);
        let db = rand_unit(300, dim, 9);
        for (i, v) in db.iter().enumerate() {
            rq.add(i as ItemId, v);
        }
        assert_eq!(
            rq.search(&db[0], 1)[0].0,
            0,
            "self-query should retrieve self"
        );
    }

    #[test]
    fn level_count_is_exact_across_the_accepted_bit_range() {
        // --bits validates 1..=16 and the level count is a u32, because `1u16 << 16`
        // overflows at the top of that range: a debug panic, and in release a mask to
        // `<< 0` giving levels = 1 and a silently degenerate all-zero codebook. This
        // pins the exact count at every accepted width.
        for bits in 1u8..=16 {
            let rq = RaBitQ::new(32, bits);
            assert_eq!(
                rq.levels,
                1u32 << bits,
                "level count must be exact at {bits} bits"
            );
            assert!(
                rq.levels >= 2,
                "a usable codebook needs at least two levels"
            );
        }
    }
}
