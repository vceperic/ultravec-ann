//! A distribution-matched scalar quantizer. This is NOT the paper's codec --- that is
//! [`crate::trellis`] --- and it backs no reported table; it is the earlier
//! marginal-matching line the trellis superseded, kept because the benchmark sweeps it.
//!
//! Same rotate → per-vector-scale → scalar-quantize pipeline as
//! [`crate::baseline`], but the codebook is matched to a heavy-tailed target instead
//! of N(0,1), and the rotation is optional.
//!
//! Three knobs the benchmark sweeps:
//!   * [`Target`] — which marginal the Lloyd-Max codebook is optimal for
//!     (fixed Student-t / Laplace, or data-aware fitted from the corpus).
//!   * `rotate` — whether to apply the Walsh-Hadamard rotation at all (the
//!     "can we skip rotation at inference" question).
//!   * `bits` — 4/5/6.
//!
//! Codebooks are owned `Vec<f32>` (data-aware ones are derived at build time),
//! unlike the TurboQuant reference's published static tables.

use crate::{
    codebook::{lloyd_max, lloyd_max_weighted},
    dist::{Gaussian, Laplace, StudentT, TargetDist},
    l2_norm, nearest_index, pack_indices, std_about_mean, unpack_indices, ItemId, MemoryBreakdown,
    Rotation, VectorBackend,
};

/// Which target marginal the codebook is matched to.
#[derive(Clone, Debug)]
pub enum Target {
    /// N(0,1) — identical centroids to the baseline; isolates the *rotation*
    /// knob when comparing against the TurboQuant reference.
    Gaussian,
    /// Fixed Student-t with the given degrees of freedom (data-oblivious).
    StudentT(f64),
    /// Laplace (data-oblivious).
    Laplace,
    /// Fitted from the corpus: estimate ν from the pooled post-rotation kurtosis
    /// (the `nu` recorded here is what was fitted).
    DataAware { nu: f64 },
    /// Training-free anisotropic target: N(0,1) codebook derived
    /// from the IP-importance-weighted Lloyd-Max `E[(1+λx²)(x−Q(x))²]`. `λ=0` ≡
    /// [`Target::Gaussian`] ≡ TurboQuant. Data-oblivious — `λ` is a fixed
    /// constant, not learned.
    Anisotropic { lambda: f64 },
}

impl Target {
    fn dist(&self) -> Box<dyn TargetDist> {
        match self {
            Target::Gaussian | Target::Anisotropic { .. } => Box::new(Gaussian),
            Target::StudentT(nu) | Target::DataAware { nu } => Box::new(StudentT::new(*nu)),
            Target::Laplace => Box::new(Laplace::default()),
        }
    }

    /// The codebook for this target at `bits` (anisotropic uses weighted Lloyd-Max).
    fn codebook(&self, bits: u8) -> Vec<f32> {
        match self {
            Target::Anisotropic { lambda } => {
                lloyd_max_weighted(&Gaussian, *lambda, bits, 500, 1e-11).centroids
            }
            _ => lloyd_max(&*self.dist(), bits, 500, 1e-11).centroids,
        }
    }

    pub fn label(&self) -> String {
        match self {
            Target::Gaussian => "ultra_gaussian".into(),
            Target::StudentT(nu) => format!("ultra_t{}", *nu as i64),
            Target::Laplace => "ultra_laplace".into(),
            Target::DataAware { nu } => format!("ultra_dataaware_t{:.1}", nu),
            Target::Anisotropic { lambda } => format!("ultra_aniso_l{:.1}", lambda),
        }
    }
}

struct QuantizedVec {
    norm: f32,
    scale: f32,
    codes: Vec<u8>,
}

pub struct UltraQuant {
    dim: usize,
    bits: u8,
    rotate: bool,
    rotation: Rotation,
    codebook: Vec<f32>,
    target: Target,
    entries: Vec<(ItemId, QuantizedVec)>,
}

impl UltraQuant {
    /// Build with a fixed (data-oblivious) target.
    pub fn new(dim: usize, bits: u8, target: Target, rotate: bool) -> Self {
        let codebook = target.codebook(bits);
        Self {
            dim,
            bits,
            rotate,
            rotation: Rotation::new(dim, crate::rotation_seed()),
            codebook,
            target,
            entries: Vec::new(),
        }
    }

    /// Build data-aware: pool the (optionally rotated) coordinates of `corpus`,
    /// estimate ν from their excess kurtosis, and derive a matched codebook.
    pub fn from_corpus(dim: usize, bits: u8, rotate: bool, corpus: &[Vec<f32>]) -> Self {
        let rotation = Rotation::new(dim, crate::rotation_seed());
        let nu = fit_student_t_nu(corpus, &rotation, rotate);
        let target = Target::DataAware { nu };
        let codebook = target.codebook(bits);
        Self {
            dim,
            bits,
            rotate,
            rotation,
            codebook,
            target,
            entries: Vec::new(),
        }
    }

    pub fn target(&self) -> &Target {
        &self.target
    }

    fn transform(&self, normalized: &[f32]) -> Vec<f32> {
        if self.rotate {
            self.rotation.apply(normalized)
        } else {
            normalized.to_vec()
        }
    }

    fn quantize(&self, x: &[f32]) -> QuantizedVec {
        let norm = l2_norm(x);
        if norm < f32::EPSILON {
            return QuantizedVec {
                norm: 0.0,
                scale: 1.0,
                codes: pack_indices(&vec![0u16; self.dim], self.bits),
            };
        }
        let normalized: Vec<f32> = x.iter().map(|v| v / norm).collect();
        let transformed = self.transform(&normalized);
        let mut scale = std_about_mean(&transformed);
        if scale < f32::EPSILON {
            scale = 1.0;
        }
        let indices: Vec<u16> = transformed
            .iter()
            .map(|&v| nearest_index(v / scale, &self.codebook))
            .collect();
        QuantizedVec {
            norm,
            scale,
            codes: pack_indices(&indices, self.bits),
        }
    }

    fn score(&self, q_t: &[f32], qv: &QuantizedVec, scratch: &mut [u16]) -> f32 {
        if qv.norm < f32::EPSILON {
            return 0.0;
        }
        unpack_indices(&qv.codes, self.bits, scratch);
        let mut dot = 0.0f32;
        for (c, &code) in q_t.iter().zip(scratch.iter()) {
            dot += c * self.codebook[code as usize];
        }
        qv.scale * dot
    }
}

impl VectorBackend for UltraQuant {
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
        let normalized: Vec<f32> = query.iter().map(|v| v / qnorm).collect();
        let q_t = self.transform(&normalized);
        let mut scratch = vec![0u16; self.dim];
        let mut results: Vec<(ItemId, f32)> = self
            .entries
            .iter()
            .map(|(id, qv)| (*id, self.score(&q_t, qv, &mut scratch)))
            .collect();
        results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }
    fn mem_bytes(&self) -> usize {
        self.entries.iter().map(|(_, q)| q.codes.len() + 8).sum()
    }
    fn memory_breakdown(&self) -> MemoryBreakdown {
        MemoryBreakdown {
            code_bytes: self.mem_bytes(),
            model_bytes: self.rotation.allocated_bytes()
                + self.codebook.capacity() * std::mem::size_of::<f32>(),
            ..MemoryBreakdown::default()
        }
    }
    fn is_approximate(&self) -> bool {
        true
    }
    fn reconstruct_unit(&self, x: &[f32]) -> Option<Vec<f32>> {
        let qv = self.quantize(x);
        if qv.norm < f32::EPSILON {
            return Some(vec![0.0; self.dim]);
        }
        let mut idx = vec![0u16; self.dim];
        unpack_indices(&qv.codes, self.bits, &mut idx);
        let transformed: Vec<f32> = idx
            .iter()
            .map(|&i| self.codebook[i as usize] * qv.scale)
            .collect();
        Some(if self.rotate {
            self.rotation.apply_inverse(&transformed)
        } else {
            transformed
        })
    }
}

/// Estimate Student-t ν from the excess kurtosis of the pooled, per-vector-
/// standardized coordinates. For a t with ν>4, excess kurtosis γ₂ = 6/(ν−4) ⇒
/// ν = 4 + 6/γ₂. Light-tailed (γ₂≤0) data ⇒ large ν (≈Gaussian). Clamped to a
/// sane numerically-stable band.
pub fn fit_student_t_nu(corpus: &[Vec<f32>], rotation: &Rotation, rotate: bool) -> f64 {
    let mut pooled: Vec<f64> = Vec::new();
    for x in corpus {
        let norm = l2_norm(x);
        if norm < f32::EPSILON {
            continue;
        }
        let normalized: Vec<f32> = x.iter().map(|v| v / norm).collect();
        let transformed = if rotate {
            rotation.apply(&normalized)
        } else {
            normalized
        };
        let scale = std_about_mean(&transformed).max(f32::EPSILON);
        pooled.extend(transformed.iter().map(|&v| (v / scale) as f64));
    }
    if pooled.is_empty() {
        return 50.0;
    }
    let excess = excess_kurtosis(&pooled);
    if excess <= 0.05 {
        return 50.0; // effectively Gaussian
    }
    (4.0 + 6.0 / excess).clamp(2.5, 50.0)
}

/// Excess kurtosis (γ₂ = μ₄/σ⁴ − 3) of a sample.
pub fn excess_kurtosis(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let m2 = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let m4 = x.iter().map(|v| (v - mean).powi(4)).sum::<f64>() / n;
    if m2 < 1e-18 {
        return 0.0;
    }
    m4 / (m2 * m2) - 3.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{cosine, next_f64};

    /// On a heavy-tailed synthetic source, the t-matched codebook should
    /// reconstruct vectors at least as well (cosine) as the Gaussian codebook —
    /// the core mechanism, before any real data.
    #[test]
    fn t_matched_beats_gaussian_on_heavy_tailed_reconstruction() {
        let dim = 256;
        let n = 400;
        // Heavy-tailed coords: ratio of two normals ≈ Cauchy-ish per coord, then
        // assemble unit vectors. (We don't rotate here; we want the raw marginal
        // heavy so the matched codebook can show its edge.)
        let mut state = 7u64;
        let normal = |s: &mut u64| -> f32 {
            // Box-Muller.
            let u1 = next_f64(s).max(1e-12);
            let u2 = next_f64(s);
            ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
        };
        let mut vecs: Vec<Vec<f32>> = Vec::new();
        for _ in 0..n {
            let v: Vec<f32> = (0..dim)
                .map(|_| {
                    let a = normal(&mut state);
                    let b = normal(&mut state).abs() + 0.3;
                    a / b // heavy-tailed
                })
                .collect();
            vecs.push(v);
        }

        let gaussian = UltraQuant::new(dim, 5, Target::Gaussian, false);
        let t_matched = UltraQuant::new(dim, 5, Target::StudentT(3.0), false);

        let recon_cosine = |q: &UltraQuant| -> f32 {
            let mut total = 0.0;
            for v in &vecs {
                let qv = q.quantize(v);
                // Reconstruct in the (un-rotated) standardized domain and compare
                // direction — scale/norm cancel in cosine.
                let mut idx = vec![0u16; dim];
                unpack_indices(&qv.codes, q.bits, &mut idx);
                let recon: Vec<f32> = idx
                    .iter()
                    .map(|&i| q.codebook[i as usize] * qv.scale)
                    .collect();
                let norm = l2_norm(v);
                let normalized: Vec<f32> = v.iter().map(|x| x / norm).collect();
                total += cosine(&normalized, &recon);
            }
            total / vecs.len() as f32
        };

        let g = recon_cosine(&gaussian);
        let t = recon_cosine(&t_matched);
        assert!(
            t >= g - 1e-4,
            "t-matched cosine {t} should be ≥ Gaussian {g}"
        );
    }

    #[test]
    fn fit_recovers_heavy_tail_as_low_nu() {
        // A pooled heavy-tailed sample should fit a small ν; a Gaussian sample a
        // large one.
        let dim = 128;
        let rot = Rotation::new(dim, 42);
        let mut state = 11u64;
        let normal = |s: &mut u64| -> f32 {
            let u1 = next_f64(s).max(1e-12);
            let u2 = next_f64(s);
            ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
        };
        let heavy: Vec<Vec<f32>> = (0..200)
            .map(|_| {
                (0..dim)
                    .map(|_| normal(&mut state) / (normal(&mut state).abs() + 0.2))
                    .collect()
            })
            .collect();
        let gauss: Vec<Vec<f32>> = (0..200)
            .map(|_| (0..dim).map(|_| normal(&mut state)).collect())
            .collect();
        let nu_heavy = fit_student_t_nu(&heavy, &rot, false);
        let nu_gauss = fit_student_t_nu(&gauss, &rot, false);
        assert!(nu_heavy < 10.0, "heavy-tailed ν {nu_heavy} should be small");
        assert!(
            nu_gauss > nu_heavy,
            "gaussian ν {nu_gauss} should exceed heavy {nu_heavy}"
        );
    }
}
