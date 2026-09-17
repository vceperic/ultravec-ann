//! Target marginal distributions for codebook derivation and goodness-of-fit tests.
//!
//! Each distribution is **standardized** so its scale matches how the quantizer
//! scales data: the quantize step divides rotated coordinates by their stddev, so
//! the codebook must be optimal for a *unit-variance* target. For Gaussian that's
//! N(0,1); for Student-t with ν>2 we rescale the standard t (whose variance is
//! ν/(ν−2)) to unit variance; for Laplace we pick scale b=1/√2 (variance 2b²=1).
//!
//! ν=2 is an edge case (the standard t₂ has infinite variance, so unit-variance
//! standardization is ill-defined). We standardize t₂ by its IQR instead.

/// A continuous, symmetric-about-0 target distribution.
pub trait TargetDist: Send + Sync {
    /// Probability density at `x`.
    fn pdf(&self, x: f64) -> f64;
    /// Cumulative distribution at `x`.
    fn cdf(&self, x: f64) -> f64;
    /// A finite integration/clip bound capturing essentially all the mass — used
    /// as the outer Lloyd-Max boundary and the quadrature limit. Heavy tails need
    /// a larger bound.
    fn support_bound(&self) -> f64;
    /// Human-readable label, e.g. `"student_t_nu4"`.
    fn name(&self) -> String;
}

// ── Gaussian N(0,1) ────────────────────────────────────────────

pub struct Gaussian;

impl TargetDist for Gaussian {
    fn pdf(&self, x: f64) -> f64 {
        (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
    }
    fn cdf(&self, x: f64) -> f64 {
        0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
    }
    fn support_bound(&self) -> f64 {
        10.0
    }
    fn name(&self) -> String {
        "gaussian".into()
    }
}

// ── Student-t, standardized to unit variance (ν>2) / unit IQR (ν=2) ──

pub struct StudentT {
    pub nu: f64,
    /// Scale `s` applied to the *standard* t so the modelled X = s·T has unit
    /// spread. variance-standardized for ν>2, IQR-standardized for ν≤2.
    scale: f64,
    /// Normalizing constant of the standard-t pdf: Γ((ν+1)/2)/(√(νπ)·Γ(ν/2)).
    norm: f64,
}

impl StudentT {
    pub fn new(nu: f64) -> Self {
        let norm = (lgamma((nu + 1.0) / 2.0) - lgamma(nu / 2.0)).exp()
            / (nu * std::f64::consts::PI).sqrt();
        // Standardize. For ν>2 the variance is ν/(ν−2); pick s = √((ν−2)/ν) so
        // Var(s·T)=1. For ν≤2 variance is infinite → use the IQR instead: a
        // standard t has a known 75th percentile q; scale so the modelled IQR
        // equals a unit-variance Gaussian's IQR (2·0.6745) for comparability.
        let scale = if nu > 2.0 {
            ((nu - 2.0) / nu).sqrt()
        } else {
            let q75 = student_t_quantile(0.75, nu, norm);
            // Gaussian IQR half-width is Φ⁻¹(0.75)=0.674489; match it.
            0.674489 / q75
        };
        Self { nu, scale, norm }
    }

    fn std_pdf(&self, t: f64) -> f64 {
        self.norm * (1.0 + t * t / self.nu).powf(-(self.nu + 1.0) / 2.0)
    }
}

impl TargetDist for StudentT {
    fn pdf(&self, x: f64) -> f64 {
        // X = s·T ⇒ f_X(x) = f_T(x/s)/s.
        self.std_pdf(x / self.scale) / self.scale
    }
    fn cdf(&self, x: f64) -> f64 {
        student_t_cdf(x / self.scale, self.nu, self.norm)
    }
    fn support_bound(&self) -> f64 {
        // Heavier tails for smaller ν. Generous but finite; the first moment of
        // the tail bin converges for ν>1, so a clip here is a tiny bias only.
        let base = match self.nu as i64 {
            2 => 200.0,
            3 => 80.0,
            4 => 50.0,
            _ => 30.0,
        };
        base * self.scale
    }
    fn name(&self) -> String {
        format!("student_t_nu{}", self.nu as i64)
    }
}

// ── Laplace, standardized to unit variance ─────────────────────

pub struct Laplace {
    b: f64,
}

impl Default for Laplace {
    fn default() -> Self {
        // Variance of Laplace(b) is 2b²; unit variance ⇒ b = 1/√2.
        Self {
            b: 1.0 / std::f64::consts::SQRT_2,
        }
    }
}

impl TargetDist for Laplace {
    fn pdf(&self, x: f64) -> f64 {
        (-(x.abs()) / self.b).exp() / (2.0 * self.b)
    }
    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.5 * (x / self.b).exp()
        } else {
            1.0 - 0.5 * (-x / self.b).exp()
        }
    }
    fn support_bound(&self) -> f64 {
        20.0 * self.b
    }
    fn name(&self) -> String {
        "laplace".into()
    }
}

// ── Special functions (pure Rust ports) ────────────────────────

/// Abramowitz-Stegun erf approximation (maximum error approximately 1.5e-7).
pub fn erf(x: f64) -> f64 {
    let (a1, a2, a3, a4, a5) = (
        0.254829592,
        -0.284496736,
        1.421413741,
        -1.453152027,
        1.061405429,
    );
    let p = 0.3275911;
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

/// Inverse standard-normal CDF (Beasley-Springer-Moro) — used to initialize
/// Gaussian codebooks and for the QQ comparison.
pub fn inv_norm_cdf(p: f64) -> f64 {
    let a = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577518672690e2,
        -3.066479806614716e1,
        2.506628277459239e0,
    ];
    let b = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    ];
    let c = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838e0,
        -2.549732539343734e0,
        4.374664141464968e0,
        2.938163982698783e0,
    ];
    let d = [
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996e0,
        3.754408661907416e0,
    ];
    let p_low = 0.02425;
    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= 1.0 - p_low {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0))
    }
}

/// Lanczos log-gamma (g=7, n=9). Accurate to ~1e-13 for x>0.
pub fn lgamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection: Γ(x)Γ(1−x) = π/sin(πx).
        (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - lgamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = C[0];
        let t = x + G + 0.5;
        for (i, &c) in C.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

/// Regularized incomplete beta I_x(a,b) via continued fraction (Lentz). Used for
/// the Student-t CDF.
fn betai(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let lbeta = lgamma(a) + lgamma(b) - lgamma(a + b);
    let front = (a * x.ln() + b * (1.0 - x).ln() - lbeta).exp() / a;
    // Continued fraction for I_x(a,b)/front.
    let mut f = 1.0;
    let mut c = 1.0;
    let mut d = 0.0;
    for i in 0..200 {
        let m = i / 2;
        let m_f = m as f64;
        let numerator = if i == 0 {
            1.0
        } else if i % 2 == 0 {
            m_f * (b - m_f) * x / ((a + 2.0 * m_f - 1.0) * (a + 2.0 * m_f))
        } else {
            -((a + m_f) * (a + b + m_f) * x) / ((a + 2.0 * m_f) * (a + 2.0 * m_f + 1.0))
        };
        d = 1.0 + numerator * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        d = 1.0 / d;
        c = 1.0 + numerator / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        let cd = c * d;
        f *= cd;
        if (1.0 - cd).abs() < 1e-12 {
            break;
        }
    }
    front * (f - 1.0)
}

/// Standard Student-t CDF (location 0, scale 1). `_norm` unused but kept so the
/// signature documents the dependency.
fn student_t_cdf(t: f64, nu: f64, _norm: f64) -> f64 {
    let x = nu / (nu + t * t);
    let ib = 0.5 * betai(x, nu / 2.0, 0.5);
    if t > 0.0 {
        1.0 - ib
    } else {
        ib
    }
}

/// Quantile of the *standard* Student-t via bisection on its CDF.
fn student_t_quantile(p: f64, nu: f64, norm: f64) -> f64 {
    let (mut lo, mut hi) = (-1.0e4, 1.0e4);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if student_t_cdf(mid, nu, norm) < p {
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

    #[test]
    fn erf_known_values() {
        assert!((erf(0.0)).abs() < 1e-9);
        assert!((erf(1.0) - 0.842_700_79).abs() < 1e-5);
    }

    #[test]
    fn gaussian_cdf_symmetric() {
        let g = Gaussian;
        assert!((g.cdf(0.0) - 0.5).abs() < 1e-6);
        assert!((g.cdf(1.96) - 0.975).abs() < 1e-3);
    }

    #[test]
    fn lgamma_matches_factorials() {
        // Γ(n) = (n−1)!  ⇒ lgamma(5) = ln(24).
        assert!((lgamma(5.0) - 24.0_f64.ln()).abs() < 1e-9);
        assert!((lgamma(1.0)).abs() < 1e-9);
        assert!((lgamma(0.5) - std::f64::consts::PI.sqrt().ln()).abs() < 1e-9);
    }

    #[test]
    fn student_t_cdf_sane() {
        // Standard t with large ν approaches the normal.
        let nu = 100.0;
        let norm = (lgamma((nu + 1.0) / 2.0) - lgamma(nu / 2.0)).exp()
            / (nu * std::f64::consts::PI).sqrt();
        assert!((student_t_cdf(0.0, nu, norm) - 0.5).abs() < 1e-6);
        assert!((student_t_cdf(1.96, nu, norm) - 0.975).abs() < 5e-3);
    }

    #[test]
    fn student_t_unit_variance_when_nu_gt_2() {
        // Numerically check Var≈1 for the standardized ν=5 target via quadrature.
        let t = StudentT::new(5.0);
        let bound = t.support_bound();
        let n = 200_000;
        let h = 2.0 * bound / n as f64;
        let mut m2 = 0.0;
        for i in 0..=n {
            let x = -bound + i as f64 * h;
            let w = if i == 0 || i == n {
                1.0
            } else if i % 2 == 1 {
                4.0
            } else {
                2.0
            };
            m2 += w * x * x * t.pdf(x);
        }
        m2 *= h / 3.0;
        assert!((m2 - 1.0).abs() < 0.02, "variance {m2} not ≈1");
    }

    #[test]
    fn densities_integrate_to_one() {
        for d in [
            Box::new(Gaussian) as Box<dyn TargetDist>,
            Box::new(StudentT::new(4.0)),
            Box::new(Laplace::default()),
        ] {
            let bound = d.support_bound();
            let n = 200_000;
            let h = 2.0 * bound / n as f64;
            let mut mass = 0.0;
            for i in 0..=n {
                let x = -bound + i as f64 * h;
                let w = if i == 0 || i == n {
                    1.0
                } else if i % 2 == 1 {
                    4.0
                } else {
                    2.0
                };
                mass += w * d.pdf(x);
            }
            mass *= h / 3.0;
            assert!((mass - 1.0).abs() < 1e-3, "{} mass {mass}", d.name());
        }
    }
}
