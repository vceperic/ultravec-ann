//! EDEN (Vargaftik et al., "EDEN: Communication-Efficient and Robust Distributed
//! Mean Estimation for Federated Learning", ICML 2022, arXiv 2108.08842) — the
//! oblivious-scalar retrieval reference related to TurboQuant. This module is the
//! artifact-native implementation used by the shared comparison harness.
//!
//! EDEN = **random rotation → standardize → deterministic scalar Lloyd-Max
//! quantization of the standard-normal marginal → optimal (least-squares) per-vector
//! scale.** Fully data-oblivious: the rotation is data-free (Hadamard) and the N(0,1)
//! Lloyd-Max codebook is universal (the D→∞ post-rotation marginal of a unit vector),
//! so insertion = encode-one-vector, no training/calibration.
//!
//! The defining feature relative to the `UltraQuant` Gaussian path is EDEN's
//! **optimal scale** `s = ⟨x̃,u_r⟩/⟨x̃,x̃⟩`
//! — the least-squares reconstruction scale that makes `x̂ = R⁻¹(s·x̃)` minimize
//! ‖u_r − s·x̃‖², so `⟨q,x̂⟩ = s·⟨q_r,x̃⟩`. This is EDEN's "deterministic optimal
//! scaling"; it differs from RaBitQ's rescaled estimator `⟨q_r,x̃⟩/⟨x̃,u_r⟩` (EDEN
//! divides by ⟨x̃,x̃⟩, RaBitQ by ⟨x̃,u_r⟩) and from TurboQuant's asymmetric score.
//! Same N(0,1) Lloyd-Max codebook as TurboQuant — so this isolates EDEN's estimator.
//!
use crate::{
    codebook::lloyd_max, dist::Gaussian, l2_norm, pack_indices, std_about_mean, unpack_indices,
    ItemId, MemoryBreakdown, Rotor, VectorBackend,
};

struct DVec {
    codes: Vec<u8>, // dim × bits, packed
    scale: f32,     // estimator scale applied to ⟨q_r,x̃⟩ (variant-dependent)
    rnorm: f32,     // ‖u − c‖ for centered decomposition (1.0 ⇒ off)
    cidx: u32,      // assigned centroid index (0 ⇒ off / single global mean)
}

/// EDEN oblivious scalar quantizer: N(0,1) Lloyd-Max codebook + optimal-scale
/// estimator. Mirrors the e8/trellis pipeline (rotate → standardize → quantize →
/// rescaled-class estimator) so the head-to-head isolates the quantizer.
pub struct EdenQuantizer {
    dim: usize,
    bits: u8,
    /// Estimator variant. `false` (default) selects the EDEN unbiased estimator
    /// `s = 1/⟨x̃,u_r⟩` (RaBitQ-style rescale, bias≈0). `true` = the biased
    /// reconstruction-MSE-optimal scale `s = ⟨x̃,u_r⟩/⟨x̃,x̃⟩` (Cauchy–Schwarz ⇒
    /// undershoots ⇒ negative bias). Set `ULTRAVEC_EDEN_BIASED=1` (per-codec) OR the
    /// global family flag `ULTRAVEC_BIASED_ESTIMATOR=1` for the latter — the
    /// estimator-axis control: the codebook and reconstruction are identical; only the
    /// scale differs, isolating estimator bias/variance from reconstruction.
    biased: bool,
    rotation: Rotor,
    /// 1-D N(0,1) Lloyd-Max centroids, ascending, 2^bits levels.
    centroids: Vec<f32>,
    cluster_centroids: Vec<Vec<f32>>,
    entries: Vec<(ItemId, DVec)>,
}

impl EdenQuantizer {
    pub fn new(dim: usize, bits: u8) -> Self {
        assert!((1..=8).contains(&bits), "EDEN supports bits ∈ 1..=8");
        let cb = lloyd_max(&Gaussian, bits, 500, 1e-11);
        Self {
            dim,
            bits,
            biased: std::env::var("ULTRAVEC_EDEN_BIASED").as_deref() == Ok("1")
                || std::env::var("ULTRAVEC_BIASED_ESTIMATOR").as_deref() == Ok("1"),
            rotation: Rotor::new_oblivious(dim, crate::rotation_seed()),
            centroids: cb.centroids,
            cluster_centroids: Vec::new(),
            entries: Vec::new(),
        }
    }

    pub fn with_centroids(mut self, centroids: Vec<Vec<f32>>) -> Self {
        assert!(centroids.iter().all(|c| c.len() == self.dim));
        self.cluster_centroids = centroids;
        self
    }

    pub fn with_rotation_fit(mut self, sample: &[Vec<f32>]) -> Self {
        self.rotation = Rotor::fit(self.dim, crate::rotation_seed(), sample);
        self
    }

    fn assign(&self, u: &[f32]) -> usize {
        let mut best = 0usize;
        let mut best_dot = f32::NEG_INFINITY;
        for (j, c) in self.cluster_centroids.iter().enumerate() {
            let d: f32 = u.iter().zip(c).map(|(a, b)| a * b).sum();
            if d > best_dot {
                best_dot = d;
                best = j;
            }
        }
        best
    }

    /// Unit vector (or its unit residual to the assigned centroid) to quantize.
    fn prep(&self, o: &[f32]) -> (Vec<f32>, f32, u32) {
        let norm = l2_norm(o).max(f32::EPSILON);
        let u: Vec<f32> = o.iter().map(|v| v / norm).collect();
        if self.cluster_centroids.is_empty() {
            (u, 1.0, 0)
        } else {
            let j = self.assign(&u);
            let mut r: Vec<f32> = u
                .iter()
                .zip(&self.cluster_centroids[j])
                .map(|(a, c)| a - c)
                .collect();
            let rn = l2_norm(&r).max(f32::EPSILON);
            r.iter_mut().for_each(|x| *x /= rn);
            (r, rn, j as u32)
        }
    }

    fn encode(&self, o: &[f32]) -> DVec {
        let (r, rnorm, cidx) = self.prep(o);
        let u_r = self.rotation.apply(&r);
        // Standardize for the N(0,1) codebook match; the optimal scale below is
        // computed against the ORIGINAL u_r so the per-vector std cancels.
        let std = std_about_mean(&u_r).max(f32::EPSILON);
        let mut idx = vec![0u16; self.dim];
        let mut dot_xu = 0.0f32; // ⟨x̃, u_r⟩
        let mut dot_xx = 0.0f32; // ⟨x̃, x̃⟩
        for (i, &ur) in u_r.iter().enumerate() {
            let ci = crate::nearest_index(ur / std, &self.centroids);
            idx[i] = ci;
            let c = self.centroids[ci as usize];
            dot_xu += c * ur;
            dot_xx += c * c;
        }
        // Faithful EDEN: the UNBIASED rescaled estimator s = 1/⟨x̃,u_r⟩ (so
        // s·⟨q_r,x̃⟩ is unbiased for ⟨q,o⟩, like RaBitQ). The biased control uses
        // the recon-MSE-optimal s = ⟨x̃,u_r⟩/⟨x̃,x̃⟩ (undershoots by Cauchy–Schwarz).
        let scale = if self.biased {
            dot_xu / dot_xx.max(f32::EPSILON)
        } else {
            1.0 / dot_xu.abs().max(f32::EPSILON) * dot_xu.signum()
        };
        DVec {
            codes: pack_indices(&idx, self.bits),
            scale,
            rnorm,
            cidx,
        }
    }

    fn score(&self, q_r: &[f32], qm: f32, dv: &DVec, scratch: &mut [u16]) -> f32 {
        unpack_indices(&dv.codes, self.bits, scratch);
        let mut dot = 0.0f32;
        for (i, &ix) in scratch.iter().enumerate() {
            dot += q_r[i] * self.centroids[ix as usize];
        }
        qm + dv.rnorm * dv.scale * dot
    }
}

impl VectorBackend for EdenQuantizer {
    fn dimensions(&self) -> usize {
        self.dim
    }
    fn len(&self) -> usize {
        self.entries.len()
    }
    fn add(&mut self, id: ItemId, embedding: &[f32]) {
        assert_eq!(embedding.len(), self.dim);
        let dv = self.encode(embedding);
        self.entries.push((id, dv));
    }
    fn add_batch(&mut self, embeddings: &[Vec<f32>]) {
        use rayon::prelude::*;
        let dim = self.dim;
        let this: &EdenQuantizer = self;
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
            .cluster_centroids
            .iter()
            .map(|c| u.iter().zip(c).map(|(a, b)| a * b).sum())
            .collect();
        let q_r = self.rotation.apply(&u);
        let mut scratch = vec![0u16; self.dim];
        let mut results: Vec<(ItemId, f32)> = self
            .entries
            .iter()
            .map(|(id, dv)| {
                let qm = qc.get(dv.cidx as usize).copied().unwrap_or(0.0);
                (*id, self.score(&q_r, qm, dv, &mut scratch))
            })
            .collect();
        results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }
    fn mem_bytes(&self) -> usize {
        // codes (dim·bits bits) + scale(4). No start state. `cluster_centroids`, not
        // `centroids` — the latter is EDEN's 1-D codebook, not the coarse set.
        let center = crate::centering_bytes(self.cluster_centroids.len());
        self.entries
            .iter()
            .map(|(_, dv)| dv.codes.len() + 4 + center)
            .sum()
    }
    fn memory_breakdown(&self) -> MemoryBreakdown {
        MemoryBreakdown {
            code_bytes: self.mem_bytes(),
            model_bytes: self.rotation.allocated_bytes()
                + self.centroids.capacity() * std::mem::size_of::<f32>()
                + self
                    .cluster_centroids
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
        let dv = self.encode(x);
        let mut scratch = vec![0u16; self.dim];
        unpack_indices(&dv.codes, self.bits, &mut scratch);
        let mut xbar = vec![0.0f32; self.dim];
        for (i, &ix) in scratch.iter().enumerate() {
            xbar[i] = self.centroids[ix as usize];
        }
        let mut recon = self.rotation.apply_inverse(&xbar);
        let n = l2_norm(&recon).max(f32::EPSILON);
        recon.iter_mut().for_each(|v| *v /= n);
        if self.cluster_centroids.is_empty() {
            Some(recon)
        } else {
            let c = &self.cluster_centroids[dv.cidx as usize];
            let mut full: Vec<f32> = c
                .iter()
                .zip(&recon)
                .map(|(cc, rr)| cc + dv.rnorm * rr)
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
    fn codebook_matches_bits() {
        for b in 1..=4u8 {
            let q = EdenQuantizer::new(64, b);
            assert_eq!(q.centroids.len(), 1usize << b);
            // ascending + symmetric about 0 (N(0,1) Lloyd-Max).
            for w in q.centroids.windows(2) {
                assert!(w[0] < w[1], "centroids must be ascending");
            }
            let s: f32 = q.centroids.iter().sum();
            assert!(
                s.abs() < 1e-2,
                "N(0,1) codebook should be ~symmetric, sum={s}"
            );
        }
    }

    #[test]
    fn eden_reconstruction_correlates_with_truth() {
        // g = ⟨ō,o⟩ well above chance for 2-bit EDEN on random unit vectors.
        let mut st = 5u64;
        let dim = 128;
        let q = EdenQuantizer::new(dim, 2);
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
            "mean g {} too low for 2-bit EDEN",
            sg / n as f64
        );
    }

    #[test]
    fn eden_self_query_retrieves_self() {
        let mut st = 9u64;
        let dim = 64;
        let mut q = EdenQuantizer::new(dim, 3);
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
        // Each vector's top-1 should usually be itself (3-bit is fairly accurate).
        let mut hits = 0;
        for (i, v) in db.iter().enumerate() {
            let r = q.search(v, 1);
            if r[0].0 == i as ItemId {
                hits += 1;
            }
        }
        assert!(hits >= 40, "EDEN 3-bit self-query hits {hits}/50 too low");
    }
}
