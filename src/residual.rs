//! A sketched inner-product residual implementing TurboQuant's Algorithm 2 for
//! the artifact's scale-dependent comparison.
//!
//! Pipeline (cosine domain — all vectors unit-normalized, so IP = cosine):
//!   base:     u_rot = rotate(u); scale = std(u_rot); quantize each coord to the
//!             N(0,1) Lloyd-Max codebook at `base_bits` = b−1.
//!   residual: r = u − dequant(u); store ‖r‖ + the SIGN bits of an independent
//!             SRHT sketch of r (m = dim sign bits = 1 bit/dim).
//!   score:    ip_base + √(π/2)/m · ‖r‖ · ⟨sketch(q), signs⟩  (unbiased IP est.)
//!
//! Bit budget: base (b−1)·dim/8 + signs dim/8 + 8 bytes overhead = b·dim/8 + 8
//! — **identical** to the baseline at b bits. So the comparison is bit-matched.
//!
use crate::{
    codebook::lloyd_max, dist::Gaussian, l2_norm, nearest_index, next_f64, pack_indices,
    std_about_mean, unpack_indices, ItemId, MemoryBreakdown, Rotation, VectorBackend,
};

const BASE_SEED: u64 = 42;
const SKETCH_SEED: u64 = 137;

/// SRHT sign sketch: an independent random-sign Walsh-Hadamard rotation, then a
/// fixed subsample of `m` coordinates scaled by √(d/m). Used to project the
/// residual; the DB side keeps only the sign bits.
pub(crate) struct SrhtSketch {
    rotation: Rotation,
    indices: Vec<usize>,
    scale: f32,
    pub(crate) m: usize,
}

impl SrhtSketch {
    pub(crate) fn new(dim: usize, m: usize, seed: u64) -> Self {
        let rotation = Rotation::new(dim, seed);
        let mut idx: Vec<usize> = (0..dim).collect();
        let mut state = seed ^ 0xABCD_1234;
        for i in (1..dim).rev() {
            let j = (next_f64(&mut state) * (i as f64 + 1.0)) as usize;
            idx.swap(i, j.min(i));
        }
        let mut indices = idx[..m.min(dim)].to_vec();
        indices.sort_unstable();
        let m = indices.len();
        Self {
            rotation,
            indices,
            scale: (dim as f32 / m as f32).sqrt(),
            m,
        }
    }

    /// The m-dim sketch of `x` (full-vector rotation then subsample).
    pub(crate) fn project(&self, x: &[f32]) -> Vec<f32> {
        let r = self.rotation.apply(x);
        self.indices.iter().map(|&i| r[i] * self.scale).collect()
    }

    pub(crate) fn allocated_bytes(&self) -> usize {
        self.rotation.allocated_bytes() + self.indices.capacity() * std::mem::size_of::<usize>()
    }
}

struct RVec {
    scale: f32,
    codes: Vec<u8>,
    residual_norm: f32,
    signs: Vec<u8>, // m bits packed 1-per
}

pub struct TurboQuantResidual {
    dim: usize,
    base_bits: u8,
    rotation: Rotation,
    codebook: Vec<f32>,
    sketch: SrhtSketch,
    sqrt_half_pi_over_m: f32,
    entries: Vec<(ItemId, RVec)>,
}

impl TurboQuantResidual {
    /// `eff_bits` is the bit budget to match against the baseline. Base uses
    /// `eff_bits − 1`; the sketch spends the remaining 1 bit/dim.
    pub fn new(dim: usize, eff_bits: u8) -> Self {
        // The budget identity is base = eff_bits - 1 plus a 1 bit/dim sketch;
        // budgets below two bits cannot allocate both components.
        assert!(
            eff_bits >= 2,
            "turboquant_residual needs eff_bits >= 2: the base spends eff_bits-1 and \
             the sketch spends 1 bit/dim, so a {eff_bits}-bit budget leaves nothing \
             for the base"
        );
        let base_bits = eff_bits - 1;
        let codebook = lloyd_max(&Gaussian, base_bits, 500, 1e-12).centroids;
        let m = dim; // 1 bit/dim
        Self {
            dim,
            base_bits,
            rotation: Rotation::new(dim, BASE_SEED),
            codebook,
            sketch: SrhtSketch::new(dim, m, SKETCH_SEED),
            sqrt_half_pi_over_m: (std::f32::consts::PI / 2.0).sqrt() / m as f32,
            entries: Vec::new(),
        }
    }

    fn quantize(&self, x: &[f32]) -> RVec {
        let norm = l2_norm(x);
        if norm < f32::EPSILON {
            return RVec {
                scale: 1.0,
                codes: pack_indices(&vec![0u16; self.dim], self.base_bits),
                residual_norm: 0.0,
                signs: pack_indices(&vec![0u16; self.sketch.m], 1),
            };
        }
        let u: Vec<f32> = x.iter().map(|v| v / norm).collect();
        let u_rot = self.rotation.apply(&u);
        let mut scale = std_about_mean(&u_rot);
        if scale < f32::EPSILON {
            scale = 1.0;
        }
        // Base quantize + reconstruct in the rotated domain.
        let mut residual_rot = vec![0.0f32; self.dim];
        let mut indices = vec![0u16; self.dim];
        for i in 0..self.dim {
            let idx = nearest_index(u_rot[i] / scale, &self.codebook);
            indices[i] = idx;
            residual_rot[i] = u_rot[i] - self.codebook[idx as usize] * scale;
        }
        let residual_norm = l2_norm(&residual_rot); // = ‖residual‖ (orthonormal rot)
                                                    // Residual back in the (normalized) original space, then sketch its signs.
        let residual_u = self.rotation.apply_inverse(&residual_rot);
        let proj = self.sketch.project(&residual_u);
        let sign_bits: Vec<u16> = proj.iter().map(|&v| if v >= 0.0 { 1 } else { 0 }).collect();
        RVec {
            scale,
            codes: pack_indices(&indices, self.base_bits),
            residual_norm,
            signs: pack_indices(&sign_bits, 1),
        }
    }

    fn score(&self, q_rot: &[f32], q_sketch: &[f32], rv: &RVec, scratch: &mut [u16]) -> f32 {
        // Base IP.
        unpack_indices(&rv.codes, self.base_bits, scratch);
        let mut dot = 0.0f32;
        for (c, &code) in q_rot.iter().zip(scratch.iter()) {
            dot += c * self.codebook[code as usize];
        }
        let ip_base = rv.scale * dot;
        // QJL residual correction.
        let mut signs = vec![0u16; self.sketch.m];
        unpack_indices(&rv.signs, 1, &mut signs);
        let mut s = 0.0f32;
        for (sq, &b) in q_sketch.iter().zip(signs.iter()) {
            s += sq * (b as f32 * 2.0 - 1.0); // 0/1 → ∓1/±1
        }
        ip_base + self.sqrt_half_pi_over_m * rv.residual_norm * s
    }
}

impl VectorBackend for TurboQuantResidual {
    fn dimensions(&self) -> usize {
        self.dim
    }
    fn len(&self) -> usize {
        self.entries.len()
    }
    fn add(&mut self, id: ItemId, embedding: &[f32]) {
        assert_eq!(embedding.len(), self.dim);
        let q = self.quantize(embedding);
        self.entries.push((id, q));
    }
    fn search(&self, query: &[f32], limit: usize) -> Vec<(ItemId, f32)> {
        let qnorm = l2_norm(query);
        if qnorm < f32::EPSILON {
            return Vec::new();
        }
        let u: Vec<f32> = query.iter().map(|v| v / qnorm).collect();
        let q_rot = self.rotation.apply(&u);
        let q_sketch = self.sketch.project(&u);
        let mut scratch = vec![0u16; self.dim];
        let mut results: Vec<(ItemId, f32)> = self
            .entries
            .iter()
            .map(|(id, rv)| (*id, self.score(&q_rot, &q_sketch, rv, &mut scratch)))
            .collect();
        results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }
    fn mem_bytes(&self) -> usize {
        // base codes + sign bits + scale(4) + residual_norm(4).
        self.entries
            .iter()
            .map(|(_, rv)| rv.codes.len() + rv.signs.len() + 8)
            .sum()
    }
    fn memory_breakdown(&self) -> MemoryBreakdown {
        MemoryBreakdown {
            code_bytes: self.mem_bytes(),
            model_bytes: self.rotation.allocated_bytes()
                + self.codebook.capacity() * std::mem::size_of::<f32>()
                + self.sketch.allocated_bytes(),
            ..MemoryBreakdown::default()
        }
    }
    fn is_approximate(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cosine;

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
                let nrm = l2_norm(&v).max(f32::EPSILON);
                v.iter_mut().for_each(|x| *x /= nrm);
                v
            })
            .collect()
    }

    /// The QJL correction must make the IP estimate *less biased* than base
    /// alone — the whole point of Algorithm 2. Mean signed error → ~0.
    #[test]
    fn residual_reduces_ip_bias() {
        let dim = 256;
        let q = TurboQuantResidual::new(dim, 5); // base 4-bit + sketch
        let db = rand_unit(300, dim, 7);
        let queries = rand_unit(60, dim, 9);
        let mut bias_base = 0.0f64;
        let mut bias_full = 0.0f64;
        let mut n = 0;
        for query in &queries {
            let qn = l2_norm(query);
            let u: Vec<f32> = query.iter().map(|v| v / qn).collect();
            let q_rot = q.rotation.apply(&u);
            let q_sk = q.sketch.project(&u);
            let mut scratch = vec![0u16; dim];
            for v in &db {
                let rv = q.quantize(v);
                // base-only estimate:
                unpack_indices(&rv.codes, q.base_bits, &mut scratch);
                let mut dot = 0.0f32;
                for (c, &code) in q_rot.iter().zip(scratch.iter()) {
                    dot += c * q.codebook[code as usize];
                }
                let ip_base = rv.scale * dot;
                let ip_full = q.score(&q_rot, &q_sk, &rv, &mut scratch);
                let truth = cosine(&u, v);
                bias_base += (ip_base - truth) as f64;
                bias_full += (ip_full - truth) as f64;
                n += 1;
            }
        }
        let (bb, bf) = (bias_base / n as f64, bias_full / n as f64);
        assert!(
            bf.abs() <= bb.abs() + 1e-3,
            "residual should not worsen IP bias: base {bb:.4} full {bf:.4}"
        );
    }

    #[test]
    fn bit_matched_to_baseline() {
        // (b-1)-bit base + dim sign bits + 8 = b-bit baseline footprint.
        let dim = 1024;
        let mut q = TurboQuantResidual::new(dim, 5);
        for (i, v) in rand_unit(50, dim, 3).iter().enumerate() {
            q.add(i as ItemId, v);
        }
        let per_vec = q.mem_bytes() / 50;
        // baseline @5 bits: 1024*5/8 + 8 = 648.
        assert_eq!(
            per_vec,
            1024 * 5 / 8 + 8,
            "residual must be bit-matched to 5-bit baseline"
        );
    }

    #[test]
    fn bit_matched_at_every_supported_budget() {
        // Exercise the full supported range and verify the budget identity.
        for eff_bits in 2u8..=8 {
            let dim = 1024;
            let mut q = TurboQuantResidual::new(dim, eff_bits);
            for (i, v) in rand_unit(30, dim, 3).iter().enumerate() {
                q.add(i as ItemId, v);
            }
            let per_vec = q.mem_bytes() / 30;
            assert_eq!(
                per_vec,
                dim * eff_bits as usize / 8 + 8,
                "residual must be bit-matched at eff_bits={eff_bits}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "eff_bits >= 2")]
    fn rejects_a_budget_too_small_to_honour() {
        // 1 bit/dim goes entirely to the sketch, leaving nothing for the base.
        TurboQuantResidual::new(128, 1);
    }
}
