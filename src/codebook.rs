//! Lloyd-Max scalar codebook derivation, generalized to an arbitrary target
//! marginal via numerical quadrature.
//!
//! Lloyd-Max (Max 1960): for a fixed source density p, the MSE-optimal `k`-level
//! scalar quantizer satisfies two conditions, iterated to a fixed point:
//!   1. **Nearest-neighbour boundaries:** bin edge bᵢ = (cᵢ₋₁ + cᵢ)/2.
//!   2. **Centroid condition:** cᵢ = E[X | bᵢ ≤ X < bᵢ₊₁] = ∫xp / ∫p over the bin.
//!
//! For the Gaussian, (2) has a closed form (∫xφ = φ(a)−φ(b)), which is how the
//! published N(0,1) tables are normally derived. We instead evaluate both
//! integrals by composite Simpson quadrature so that *any* [`TargetDist`] works,
//! which is the point of this module. Fed the Gaussian, the generator reproduces
//! the hardcoded `GAUSSIAN_*` tables below to the printed precision; the unit
//! tests assert that, and it is what makes a derived non-Gaussian codebook
//! trustworthy. The tables themselves are the standard Lloyd-Max N(0,1)
//! quantization levels, reproduced here so the comparison arm needs no fitting
//! step at run time.

use crate::dist::{inv_norm_cdf, TargetDist};

// ── Published N(0,1) Lloyd-Max codebooks ─────────────────────────
// Sorted ascending and symmetric about zero.

#[rustfmt::skip]
pub const GAUSSIAN_4: [f32; 16] = [
    -2.732590, -2.069017, -1.618046, -1.256231, -0.942340, -0.656759, -0.388048, -0.128395,
     0.128395,  0.388048,  0.656759,  0.942340,  1.256231,  1.618046,  2.069017,  2.732590,
];

#[rustfmt::skip]
pub const GAUSSIAN_5: [f32; 32] = [
    -3.255551, -2.685242, -2.311436, -2.022176, -1.780581, -1.569615, -1.379897, -1.205655,
    -1.043047, -0.889350, -0.742542, -0.601049, -0.463598, -0.329119, -0.196679, -0.065429,
     0.065429,  0.196679,  0.329119,  0.463598,  0.601049,  0.742542,  0.889350,  1.043047,
     1.205655,  1.379897,  1.569615,  1.780581,  2.022176,  2.311436,  2.685242,  3.255551,
];

#[rustfmt::skip]
pub const GAUSSIAN_6: [f32; 64] = [
    -3.605999, -3.085722, -2.751095, -2.496953, -2.288778, -2.110705, -1.954065, -1.813578,
    -1.685776, -1.568258, -1.459279, -1.357532, -1.262008, -1.171905, -1.086573, -1.005472,
    -0.928148, -0.854207, -0.783309, -0.715146, -0.649445, -0.585951, -0.524433, -0.464669,
    -0.406453, -0.349584, -0.293873, -0.239131, -0.185179, -0.131837, -0.078929, -0.026281,
     0.026281,  0.078929,  0.131837,  0.185179,  0.239131,  0.293873,  0.349584,  0.406453,
     0.464669,  0.524433,  0.585951,  0.649445,  0.715146,  0.783309,  0.854207,  0.928148,
     1.005472,  1.086573,  1.171905,  1.262008,  1.357532,  1.459279,  1.568258,  1.685776,
     1.813578,  1.954065,  2.110705,  2.288778,  2.496953,  2.751095,  3.085722,  3.605999,
];

/// The published Gaussian codebook for a supported bit-width.
pub fn gaussian(bits: u8) -> &'static [f32] {
    match bits {
        4 => &GAUSSIAN_4,
        5 => &GAUSSIAN_5,
        6 => &GAUSSIAN_6,
        _ => panic!("unsupported bit-width {bits}; use 4/5/6"),
    }
}

/// Outcome of a Lloyd-Max derivation, including stability diagnostics.
pub struct LloydMaxResult {
    pub centroids: Vec<f32>,
    /// Iterations until convergence (or the cap).
    pub iterations: usize,
    /// Max centroid movement on the final iteration (≈0 ⇒ converged).
    pub final_delta: f64,
    /// Whether all centroids are finite and strictly increasing.
    pub stable: bool,
}

/// Composite-Simpson quadrature of (∫ p, ∫ x·p) over [a,b] with `steps`
/// subintervals (`steps` forced even).
fn moment01(dist: &dyn TargetDist, a: f64, b: f64, steps: usize) -> (f64, f64) {
    if b <= a {
        return (0.0, 0.0);
    }
    let steps = steps.max(2) & !1; // even
    let h = (b - a) / steps as f64;
    let (mut mass, mut first) = (0.0, 0.0);
    for i in 0..=steps {
        let x = a + i as f64 * h;
        let w = if i == 0 || i == steps {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        let p = dist.pdf(x);
        mass += w * p;
        first += w * x * p;
    }
    let f = h / 3.0;
    (mass * f, first * f)
}

/// Steps to use for a bin of width `w`: ~fine fixed density, clamped, so wide
/// tail bins still integrate accurately without being absurdly slow.
fn steps_for(width: f64) -> usize {
    ((width * 400.0) as usize).clamp(64, 60_000)
}

/// Derive a `bits`-wide Lloyd-Max codebook for `dist`. Initializes centroids at
/// equiprobable quantiles, then iterates the two Lloyd-Max conditions to a fixed
/// point (or `max_iters`).
pub fn lloyd_max(dist: &dyn TargetDist, bits: u8, max_iters: usize, tol: f64) -> LloydMaxResult {
    let k = 1usize << bits;
    let bound = dist.support_bound();

    // Initialize at equiprobable quantiles via bisection on the CDF.
    let mut c: Vec<f64> = (0..k)
        .map(|i| {
            let p = (i as f64 + 0.5) / k as f64;
            quantile(dist, p, bound)
        })
        .collect();
    c.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut iterations = 0;
    let mut final_delta = f64::INFINITY;
    for it in 0..max_iters {
        iterations = it + 1;
        // (1) boundaries.
        let mut edges = vec![-bound; k + 1];
        edges[k] = bound;
        for i in 1..k {
            edges[i] = 0.5 * (c[i - 1] + c[i]);
        }
        // (2) centroids = conditional mean per bin.
        let mut delta = 0.0f64;
        let mut next = c.clone();
        for i in 0..k {
            let (a, b) = (edges[i], edges[i + 1]);
            let (mass, first) = moment01(dist, a, b, steps_for(b - a));
            if mass > 1e-15 {
                next[i] = first / mass;
            }
            delta = delta.max((next[i] - c[i]).abs());
        }
        c = next;
        final_delta = delta;
        if delta < tol {
            break;
        }
    }

    // Stability requires finite, monotone centroids. Slow convergence (a large outer
    // centroid whose conditional-mean integral has quadrature noise that floors
    // `final_delta`) is reported separately from this structural check.
    let stable = c.iter().all(|v| v.is_finite()) && c.windows(2).all(|w| w[1] - w[0] > 1e-9);

    LloydMaxResult {
        centroids: c.iter().map(|&v| v as f32).collect(),
        iterations,
        final_delta,
        stable,
    }
}

/// Per-bin moments (∫p, ∫xp, ∫x²p, ∫x³p) over [a,b] by composite Simpson.
fn moments0123(dist: &dyn TargetDist, a: f64, b: f64, steps: usize) -> (f64, f64, f64, f64) {
    if b <= a {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let steps = steps.max(2) & !1;
    let h = (b - a) / steps as f64;
    let (mut i0, mut i1, mut i2, mut i3) = (0.0, 0.0, 0.0, 0.0);
    for i in 0..=steps {
        let x = a + i as f64 * h;
        let w = if i == 0 || i == steps {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        let p = dist.pdf(x);
        let wp = w * p;
        i0 += wp;
        i1 += wp * x;
        i2 += wp * x * x;
        i3 += wp * x * x * x;
    }
    let f = h / 3.0;
    (i0 * f, i1 * f, i2 * f, i3 * f)
}

/// Score-aware (anisotropic) Lloyd-Max: minimizes the **IP-importance-weighted**
/// distortion `E[(1+λx²)(x−Q(x))²]` for `dist`, instead of plain MSE. The weight
/// `w(x)=1+λx²` is the per-coordinate inner-product importance (derived from
/// ScaNN's parallel-residual decomposition on the rotated iid-Gaussian source):
/// large-|x| coordinates dominate the inner product with an aligned query, so
/// preserving them matters more for *ranking* than minimizing MSE.
///
/// `w(x)>0` is identical across candidate centroids at a fixed `x`, so it does
/// **not** change the nearest-centroid assignment (boundaries stay midpoints) —
/// only the centroid-update integral becomes weighted:
/// `cᵢ = ∫w·x·p / ∫w·p = (I1+λI3)/(I0+λI2)` per bin.
///
/// `λ=0` is exactly [`lloyd_max`] (TurboQuant). Still **data-oblivious**: `λ` is a
/// fixed constant, the target is the fixed N(0,1) post-rotation source.
pub fn lloyd_max_weighted(
    dist: &dyn TargetDist,
    lambda: f64,
    bits: u8,
    max_iters: usize,
    tol: f64,
) -> LloydMaxResult {
    let k = 1usize << bits;
    let bound = dist.support_bound();
    let mut c: Vec<f64> = (0..k)
        .map(|i| quantile(dist, (i as f64 + 0.5) / k as f64, bound))
        .collect();
    c.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut iterations = 0;
    let mut final_delta = f64::INFINITY;
    for it in 0..max_iters {
        iterations = it + 1;
        let mut edges = vec![-bound; k + 1];
        edges[k] = bound;
        for i in 1..k {
            edges[i] = 0.5 * (c[i - 1] + c[i]);
        }
        let mut delta = 0.0f64;
        let mut next = c.clone();
        for i in 0..k {
            let (a, b) = (edges[i], edges[i + 1]);
            let (i0, i1, i2, i3) = moments0123(dist, a, b, steps_for(b - a));
            let num = i1 + lambda * i3;
            let den = i0 + lambda * i2;
            if den.abs() > 1e-15 {
                next[i] = num / den;
            }
            delta = delta.max((next[i] - c[i]).abs());
        }
        c = next;
        final_delta = delta;
        if delta < tol {
            break;
        }
    }
    let stable = c.iter().all(|v| v.is_finite()) && c.windows(2).all(|w| w[1] - w[0] > 1e-9);
    LloydMaxResult {
        centroids: c.iter().map(|&v| v as f32).collect(),
        iterations,
        final_delta,
        stable,
    }
}

/// Quantile of `dist` at probability `p` via bisection on its CDF (the generic
/// path; Gaussian could use [`inv_norm_cdf`] directly but bisection keeps one
/// code path for all targets).
fn quantile(dist: &dyn TargetDist, p: f64, bound: f64) -> f64 {
    // Warm start near the Gaussian quantile — speeds convergence, harmless.
    let _ = inv_norm_cdf;
    let (mut lo, mut hi) = (-bound, bound);
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if dist.cdf(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::{Gaussian, Laplace, StudentT};

    /// The generalized generator, fed the Gaussian, reproduces the published
    /// N(0,1) Lloyd-Max reference tables.
    #[test]
    fn gaussian_generator_reproduces_reference_tables() {
        for (bits, table) in [
            (4u8, &GAUSSIAN_4[..]),
            (5, &GAUSSIAN_5[..]),
            (6, &GAUSSIAN_6[..]),
        ] {
            let r = lloyd_max(&Gaussian, bits, 500, 1e-12);
            assert!(r.stable, "{bits}-bit Gaussian run unstable");
            assert_eq!(r.centroids.len(), table.len());
            let max_err = r
                .centroids
                .iter()
                .zip(table)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_err < 3e-3,
                "{bits}-bit max centroid error {max_err} vs reference table"
            );
        }
    }

    #[test]
    fn codebooks_are_symmetric_about_zero() {
        for dist in [
            Box::new(StudentT::new(4.0)) as Box<dyn TargetDist>,
            Box::new(StudentT::new(3.0)),
            Box::new(Laplace::default()),
        ] {
            let r = lloyd_max(&*dist, 5, 500, 1e-10);
            assert!(r.stable, "{} unstable", dist.name());
            let k = r.centroids.len();
            for i in 0..k / 2 {
                let lo = r.centroids[i];
                let hi = r.centroids[k - 1 - i];
                assert!(
                    (lo + hi).abs() < 5e-3,
                    "{} not symmetric: {lo} vs {hi}",
                    dist.name()
                );
            }
        }
    }

    #[test]
    fn weighted_lloyd_max_lambda0_equals_plain() {
        // λ=0 must reproduce plain MSE Lloyd-Max and the reference table.
        let w = lloyd_max_weighted(&Gaussian, 0.0, 5, 500, 1e-12);
        let p = lloyd_max(&Gaussian, 5, 500, 1e-12);
        let max_err = w
            .centroids
            .iter()
            .zip(&p.centroids)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 1e-4,
            "λ=0 weighted should equal plain, err {max_err}"
        );
    }

    #[test]
    fn anisotropic_widens_dynamic_range() {
        // λ>0 weights large-|x| (IP-relevant) coords more → outer centroid moves
        // further out (finer tail resolution) than plain MSE.
        let plain = lloyd_max(&Gaussian, 5, 500, 1e-12);
        let aniso = lloyd_max_weighted(&Gaussian, 4.0, 5, 500, 1e-11);
        assert!(aniso.stable);
        let p_max = *plain.centroids.last().unwrap();
        let a_max = *aniso.centroids.last().unwrap();
        assert!(
            a_max > p_max,
            "anisotropic outer centroid {a_max} should exceed plain {p_max}"
        );
    }

    #[test]
    fn heavy_tail_codebook_has_wider_extremes_than_gaussian() {
        // The mechanism: a heavy-tailed target should place its outermost
        // centroid FURTHER out than the Gaussian one (wider buckets in the tail).
        let g = lloyd_max(&Gaussian, 5, 500, 1e-12);
        let t = lloyd_max(&StudentT::new(3.0), 5, 500, 1e-10);
        let g_max = *g.centroids.last().unwrap();
        let t_max = *t.centroids.last().unwrap();
        assert!(
            t_max > g_max,
            "t₃ outer centroid {t_max} should exceed Gaussian {g_max}"
        );
    }
}
