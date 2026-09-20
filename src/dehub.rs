//! PCA-head residual codec with an adaptively selected rank, implemented as a
//! first-class Rust `VectorBackend` (legacy backend name: `dehub`).
//!
//! The codec stores quantized coefficients for the top-`r` principal directions of
//! the corpus and codes the orthogonal-complement residual with the data-oblivious
//! trellis. The PCA head is batch fitted and codebook-free.
//!
//! At a fixed bit rate, adding exact head dimensions increases fidelity while also
//! increasing storage, so rank selection is defined under a per-vector byte budget.
//! The codec selects the rank/bit-rate split that maximizes held-out geometric
//! Recall@10; rank zero remains available when the PCA head does not improve the
//! budget-matched objective.
//!
//! Pure Rust and BLAS-free. The top-`r` PCA uses subspace (Rayleigh-Ritz)
//! iteration on the data matrix — no d×d eigendecomposition (jacobi_eigen is O(d^3), infeasible at
//! d=4096) — reusing `learned_rotation::jacobi_eigen` only for the small r×r Ritz step.

use crate::learned_rotation::jacobi_eigen;
use crate::trellis::TrellisQuantizer;
use crate::{l2_norm, next_f64, ItemId, MemoryBreakdown, VectorBackend};
use rayon::prelude::*;

/// Candidate ranks swept by the adaptive selector (capped at min(r_max, dim-1)).
const CANDIDATES: &[usize] = &[0, 8, 16, 32, 64, 96, 128];

/// Explicit, reproducible configuration for fitting the PCA-head residual codec.
///
/// The command-line compatibility path can still populate this structure from
/// `ULTRAVEC_DEHUB_*` variables, but library callers and artifact experiments
/// should pass a value directly so concurrent runs cannot affect one another.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DehubConfig {
    pub rank: usize,
    pub max_rank: usize,
    pub pca_iterations: usize,
    pub head_bits: u8,
    pub complement: bool,
    pub seed: u64,
}

impl Default for DehubConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            max_rank: 128,
            pca_iterations: 24,
            head_bits: 8,
            complement: false,
            seed: 0xA5E5,
        }
    }
}

impl DehubConfig {
    /// Legacy command-line configuration. Prefer constructing `DehubConfig`
    /// explicitly in new code and in reproducibility scripts.
    pub fn from_env() -> Self {
        let defaults = Self::default();
        Self {
            rank: env_usize("ULTRAVEC_DEHUB_R", defaults.rank),
            max_rank: env_usize("ULTRAVEC_DEHUB_RMAX", defaults.max_rank),
            pca_iterations: env_usize("ULTRAVEC_DEHUB_ITERS", defaults.pca_iterations),
            head_bits: env_usize("ULTRAVEC_DEHUB_HEAD_BITS", defaults.head_bits as usize)
                .try_into()
                .unwrap_or(u8::MAX),
            complement: std::env::var("ULTRAVEC_DEHUB_COMPLEMENT").as_deref() == Ok("1"),
            seed: defaults.seed,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.pca_iterations == 0 {
            return Err("pca_iterations must be at least 1".to_string());
        }
        if !(1..=16).contains(&self.head_bits) {
            return Err("head_bits must be in 1..=16".to_string());
        }
        Ok(())
    }

    fn assert_valid(&self) {
        if let Err(message) = self.validate() {
            panic!("invalid PCA-head configuration: {message}");
        }
    }
}

/// Top-`r` principal directions of the second-moment matrix `C = (1/n) Σ xᵢ xᵢᵀ` of `data`
/// (rows used as-is — the caller passes unit-normalized rows, matching the Python `DB.T@DB/n`),
/// via subspace iteration. Returns `(eigvals_desc, basis)` with `basis` row-major `r×d` (row k =
/// k-th eigenvector) and `eigvals_desc` the Ritz values (approximate top-r eigenvalues of C).
pub fn top_r_pca(
    data: &[Vec<f32>],
    d: usize,
    r: usize,
    iters: usize,
    seed: u64,
) -> (Vec<f64>, Vec<f32>) {
    let n = data.len().max(1);
    let r = r.min(d);
    // Q: r×d f64, random rows, modified-Gram-Schmidt orthonormalized.
    let mut st = seed;
    let mut q = vec![0.0f64; r * d];
    for x in q.iter_mut() {
        let u1 = next_f64(&mut st).max(1e-12);
        let u2 = next_f64(&mut st);
        *x = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    }
    mgs_rows(&mut q, r, d);

    let cq = |q: &[f64]| -> Vec<f64> {
        // Y = C Q  (r×d):  Y_k = (1/n) Σ_i (xᵢ·q_k) xᵢ.  Z[i][k] = xᵢ·q_k, parallel over i.
        let z: Vec<f64> = data
            .par_iter()
            .flat_map_iter(|x| {
                (0..r).map(move |k| {
                    let qk = &q[k * d..(k + 1) * d];
                    let mut s = 0.0f64;
                    for j in 0..d {
                        s += x[j] as f64 * qk[j];
                    }
                    s
                })
            })
            .collect(); // n*r row-major (i,k)
                        // Y_k[j] = (1/n) Σ_i z[i*r+k] x_i[j]. Parallel over k.
        (0..r)
            .into_par_iter()
            .flat_map_iter(|k| {
                let mut yk = vec![0.0f64; d];
                for (i, x) in data.iter().enumerate() {
                    let zik = z[i * r + k];
                    if zik != 0.0 {
                        for j in 0..d {
                            yk[j] += zik * x[j] as f64;
                        }
                    }
                }
                for v in yk.iter_mut() {
                    *v /= n as f64;
                }
                yk.into_iter()
            })
            .collect::<Vec<f64>>()
    };

    for _ in 0..iters.max(1) {
        let mut y = cq(&q);
        mgs_rows(&mut y, r, d);
        q = y;
    }

    // Rayleigh-Ritz: M = Qᵀ C Q ≈ M[k][l] = q_k · (C q_l). Then eig(M) rotates Q to the Ritz basis.
    let y = cq(&q); // C Q
    let mut m = vec![0.0f64; r * r];
    for k in 0..r {
        for l in 0..r {
            let mut s = 0.0f64;
            for j in 0..d {
                s += q[k * d + j] * y[l * d + j];
            }
            m[k * r + l] = s;
        }
    }
    let (theta, w) = jacobi_eigen(m, r); // w row j = j-th eigenvector of M (desc)
                                         // Basis row j = Σ_k w[j][k] q_k.
    let mut basis = vec![0.0f32; r * d];
    for j in 0..r {
        for k in 0..r {
            let wjk = w[j * r + k];
            if wjk != 0.0 {
                let qk = &q[k * d..(k + 1) * d];
                let bj = &mut basis[j * d..(j + 1) * d];
                for c in 0..d {
                    bj[c] += (wjk * qk[c]) as f32;
                }
            }
        }
    }
    (theta, basis)
}

/// In-place modified Gram-Schmidt orthonormalization of the `r` rows of a row-major `r×d` matrix.
fn mgs_rows(a: &mut [f64], r: usize, d: usize) {
    for i in 0..r {
        for j in 0..i {
            let mut dot = 0.0f64;
            for c in 0..d {
                dot += a[i * d + c] * a[j * d + c];
            }
            for c in 0..d {
                a[i * d + c] -= dot * a[j * d + c];
            }
        }
        let mut nrm = 0.0f64;
        for c in 0..d {
            nrm += a[i * d + c] * a[i * d + c];
        }
        let nrm = nrm.sqrt().max(1e-18);
        for c in 0..d {
            a[i * d + c] /= nrm;
        }
    }
}

/// Householder reduction that maps the top-`r` head subspace to the first `r` axes, so the residual
/// lives in the last `dim-r` coordinates and the trellis codes `pad2(dim-r)` dims instead of `dim`
/// (the orthogonal-complement coding of `dir-dehub-complement.md`, O(r·dim)/vector, no explicit
/// complement basis). `ULTRAVEC_DEHUB_COMPLEMENT=1` enables it.
struct Householder {
    v: Vec<f32>,    // r reflector vectors, row-major r×dim (v_k has zeros in coords < k)
    beta: Vec<f32>, // 2/(v_k·v_k) per reflector
    sign: Vec<f32>, // s_k = sign((Qᵀ b_k)_k) ∈ {±1}: Qᵀ b_k = s_k e_k, so (Qᵀx)_k = s_k (x·b_k)
    dim: usize,
    r: usize,
}

impl Householder {
    /// Build from the orthonormal top-`r` basis (row k = b_k) via Householder QR of [b_0|…|b_{r-1}].
    fn build(basis: &[f32], dim: usize, r: usize) -> Self {
        // B columns = basis vectors; reduce to upper-triangular, recording reflectors.
        let mut b = vec![0.0f32; dim * r]; // dim×r column-major-as-rows: b[i*r+k] = (b_k)_i
        for k in 0..r {
            for i in 0..dim {
                b[i * r + k] = basis[k * dim + i];
            }
        }
        let mut v = vec![0.0f32; r * dim];
        let mut beta = vec![0.0f32; r];
        for k in 0..r {
            // x = B[k.., k]; reflector zeroes it below row k.
            let mut nrm = 0.0f32;
            for i in k..dim {
                nrm += b[i * r + k] * b[i * r + k];
            }
            let nrm = nrm.sqrt();
            if nrm < 1e-20 {
                continue;
            }
            let alpha = if b[k * r + k] >= 0.0 { -nrm } else { nrm };
            let vk = &mut v[k * dim..(k + 1) * dim];
            for i in k..dim {
                vk[i] = b[i * r + k];
            }
            vk[k] -= alpha;
            let vv = vk[k..dim].iter().map(|x| x * x).sum::<f32>();
            if vv < 1e-30 {
                continue;
            }
            beta[k] = 2.0 / vv;
            // Apply H_k to the remaining columns k+1..r of B.
            for c in (k + 1)..r {
                let mut dot = 0.0f32;
                for i in k..dim {
                    dot += vk[i] * b[i * r + c];
                }
                let f = beta[k] * dot;
                for i in k..dim {
                    b[i * r + c] -= f * vk[i];
                }
            }
        }
        let mut hh = Householder {
            v,
            beta,
            sign: vec![1.0; r],
            dim,
            r,
        };
        // s_k = sign of (Qᵀ b_k)_k, so the head coeff in the rotated frame is s_k·(x·b_k).
        for k in 0..r {
            let bk = &basis[k * dim..(k + 1) * dim];
            let yk = hh.qt(bk)[k];
            hh.sign[k] = if yk >= 0.0 { 1.0 } else { -1.0 };
        }
        hh
    }

    /// y = Qᵀ x = H_{r-1}…H_0 x. y[0..r] = head coeffs; y[r..] = residual complement coords.
    fn qt(&self, x: &[f32]) -> Vec<f32> {
        let mut y = x.to_vec();
        for k in 0..self.r {
            let vk = &self.v[k * self.dim..(k + 1) * self.dim];
            if self.beta[k] == 0.0 {
                continue;
            }
            let mut dot = 0.0f32;
            for i in k..self.dim {
                dot += vk[i] * y[i];
            }
            let f = self.beta[k] * dot;
            for i in k..self.dim {
                y[i] -= f * vk[i];
            }
        }
        y
    }

    /// x = Q y = H_0…H_{r-1} y (reflectors in reverse order).
    fn q(&self, y: &[f32]) -> Vec<f32> {
        let mut x = y.to_vec();
        for k in (0..self.r).rev() {
            let vk = &self.v[k * self.dim..(k + 1) * self.dim];
            if self.beta[k] == 0.0 {
                continue;
            }
            let mut dot = 0.0f32;
            for i in k..self.dim {
                dot += vk[i] * x[i];
            }
            let f = self.beta[k] * dot;
            for i in k..self.dim {
                x[i] -= f * vk[i];
            }
        }
        x
    }

    fn allocated_bytes(&self) -> usize {
        self.v.capacity() * std::mem::size_of::<f32>()
            + self.beta.capacity() * std::mem::size_of::<f32>()
            + self.sign.capacity() * std::mem::size_of::<f32>()
    }
}

/// The fitted PCA-head codec: a top-`r` quantized head plus an oblivious trellis on the residual.
pub struct DehubBackend {
    dim: usize,
    r: usize,
    basis: Vec<f32>, // r×dim row-major (row k = k-th principal direction)
    q8_lo: Vec<f32>, // per-coefficient 8-bit range, fit on the fit set
    q8_scale: Vec<f32>,
    trellis: TrellisQuantizer,
    // Optional orthogonal-complement coding: Householder reduction + a trellis sized pad2(dim-r).
    complement: Option<(Householder, TrellisQuantizer, usize)>, // (hh, comp_trellis, comp_dim=dim-r)
    entries: Vec<(ItemId, Vec<f32>)>, // reconstructed unit directions (decode cache)
    head_levels: f32,
    bytes_per_vec: usize,
}

/// Number of intervals in the quantized PCA head.
fn head_levels(bits: u8) -> f32 {
    ((1u32 << bits) - 1) as f32
}

/// Per-vector byte cost of the PCA-head codec at (rank r, bit-rate b): trellis codes + free-start state
/// + r head coefficients at `head_bits` each. With complement coding the residual codes only
///
/// `pad2(dim-r)` dims instead of `dim`.
pub fn dehub_bytes(dim: usize, r: usize, b: u8) -> usize {
    let config = DehubConfig::from_env();
    dehub_bytes_with_config(dim, r, b, &config)
}

/// Deterministic per-vector byte cost for an explicit configuration.
pub fn dehub_bytes_with_config(dim: usize, r: usize, b: u8, config: &DehubConfig) -> usize {
    config.assert_valid();
    let code_dim = if r > 0 && r < dim && config.complement {
        (dim - r).next_power_of_two()
    } else {
        dim
    };
    (b as usize * code_dim).div_ceil(8) + 2 + (r * config.head_bits as usize).div_ceil(8)
}

impl DehubBackend {
    /// Fixed (r, b): `recon --backend dehub --bits b` with `ULTRAVEC_DEHUB_R=r`. For validation
    /// against the Python fixed-r PCA-head control. r=0 ⇒ plain trellis.
    pub fn fit(dim: usize, bits: u8, data: &[Vec<f32>]) -> Self {
        Self::fit_with_config(dim, bits, data, DehubConfig::from_env())
    }

    /// Fit a fixed-rate codec with an explicit, process-local configuration.
    pub fn fit_with_config(dim: usize, bits: u8, data: &[Vec<f32>], config: DehubConfig) -> Self {
        config.assert_valid();
        Self::build(dim, bits, config.rank, data, "fixed", config)
    }

    /// Budget-driven adaptive: pick the (rank r, bit-rate b) maximizing held-out geom-Recall@10
    /// subject to `dehub_bytes(dim,r,b) <= budget_bytes`. `recon --backend dehub --budget-bytes B`.
    pub fn fit_budget(dim: usize, budget_bytes: usize, data: &[Vec<f32>]) -> Self {
        Self::fit_budget_with_config(dim, budget_bytes, data, DehubConfig::from_env())
    }

    /// Budget-driven fit with an explicit, process-local configuration.
    pub fn fit_budget_with_config(
        dim: usize,
        budget_bytes: usize,
        data: &[Vec<f32>],
        config: DehubConfig,
    ) -> Self {
        config.assert_valid();
        let minimum = dehub_bytes_with_config(dim, 0, 1, &config);
        assert!(
            budget_bytes >= minimum,
            "PCA-head budget {budget_bytes}B is too small: minimum is {minimum}B for a 1-bit code"
        );
        let r_max = config.max_rank.min(dim.saturating_sub(1));
        let (_eig, basis_full) = if r_max == 0 {
            (vec![], vec![])
        } else {
            top_r_pca(data, dim, r_max, config.pca_iterations, config.seed)
        };
        let r_have = basis_full.len().checked_div(dim).unwrap_or(0);
        let levels = head_levels(config.head_bits);
        let (q8_lo, q8_scale) = fit_q8(data, &basis_full, dim, r_have, levels);
        let (r, b) = select_config(
            dim,
            budget_bytes,
            data,
            &basis_full,
            &q8_lo,
            &q8_scale,
            r_have,
            levels,
            &config,
        );
        eprintln!(
            "dehub: dim={dim} budget={budget_bytes}B fit_n={} r_max={r_have} -> r={r} b={b} ({}B used)",
            data.len(), dehub_bytes_with_config(dim, r, b, &config)
        );
        Self::from_basis(dim, b, r, basis_full, q8_lo, q8_scale, levels, config)
    }

    fn build(
        dim: usize,
        bits: u8,
        r_req: usize,
        data: &[Vec<f32>],
        tag: &str,
        config: DehubConfig,
    ) -> Self {
        let r_max = r_req.min(dim.saturating_sub(1));
        let (_eig, basis_full) = if r_max == 0 {
            (vec![], vec![])
        } else {
            top_r_pca(data, dim, r_max, config.pca_iterations, config.seed)
        };
        let r_have = basis_full.len().checked_div(dim).unwrap_or(0);
        let levels = head_levels(config.head_bits);
        let (q8_lo, q8_scale) = fit_q8(data, &basis_full, dim, r_have, levels);
        let r = r_req.min(r_have);
        eprintln!(
            "dehub: dim={dim} bits={bits} fit_n={} -> r={r} ({tag}, {}B)",
            data.len(),
            dehub_bytes_with_config(dim, r, bits, &config)
        );
        Self::from_basis(dim, bits, r, basis_full, q8_lo, q8_scale, levels, config)
    }

    #[allow(clippy::too_many_arguments)]
    fn from_basis(
        dim: usize,
        bits: u8,
        r: usize,
        basis_full: Vec<f32>,
        q8_lo: Vec<f32>,
        q8_scale: Vec<f32>,
        levels: f32,
        config: DehubConfig,
    ) -> Self {
        let basis = basis_full[..r * dim].to_vec();
        // Orthogonal-complement coding: a Householder reduction + a trellis sized pad2(dim-r), so the
        // residual codes pad2(dim-r) dims instead of dim when explicitly configured.
        let complement = if r > 0 && r < dim && config.complement {
            let comp_dim = dim - r;
            let pad = comp_dim.next_power_of_two();
            Some((
                Householder::build(&basis, dim, r),
                TrellisQuantizer::new(pad, bits),
                comp_dim,
            ))
        } else {
            None
        };
        DehubBackend {
            dim,
            r,
            basis,
            q8_lo: q8_lo[..r].to_vec(),
            q8_scale: q8_scale[..r].to_vec(),
            trellis: TrellisQuantizer::new(dim, bits),
            complement,
            entries: Vec::new(),
            head_levels: levels,
            bytes_per_vec: dehub_bytes_with_config(dim, r, bits, &config),
        }
    }

    /// PCA-head reconstruction of a single unit-direction vector: 8-bit head + trellis residual,
    /// coding the residual in its orthogonal complement (Householder) when enabled.
    fn recon_one(&self, x: &[f32]) -> Vec<f32> {
        match &self.complement {
            Some((hh, ct, comp_dim)) => recon_complement(
                x,
                self.dim,
                self.r,
                &self.q8_lo,
                &self.q8_scale,
                hh,
                ct,
                *comp_dim,
                self.head_levels,
            ),
            None => recon_with(
                x,
                self.dim,
                self.r,
                &self.basis,
                &self.q8_lo,
                &self.q8_scale,
                &self.trellis,
                self.head_levels,
            ),
        }
    }
}

/// Complement reconstruction: head coeffs from Qᵀx[0..r] (= x·b_k, so no explicit basis needed),
/// trellis on the pad2(dim-r) complement coords, then map back through Q.
#[allow(clippy::too_many_arguments)]
fn recon_complement(
    x: &[f32],
    d: usize,
    r: usize,
    q8_lo: &[f32],
    q8_scale: &[f32],
    hh: &Householder,
    ct: &TrellisQuantizer,
    comp_dim: usize,
    levels: f32,
) -> Vec<f32> {
    let y = hh.qt(x); // y[0..r] = s_k·c_k (rotated-frame head coeffs); y[r..] = complement residual coords
    let mut yhat = vec![0.0f32; d];
    // exact head: undo the frame sign to recover c_k = x·b_k, quantize with the c_k-fit ranges, re-sign.
    for k in 0..r {
        let c = hh.sign[k] * y[k]; // = c_k, in the frame q8_lo/q8_scale were fit in
        let cl = c.clamp(q8_lo[k], q8_lo[k] + q8_scale[k] * levels);
        let cq = ((cl - q8_lo[k]) / q8_scale[k]).round() * q8_scale[k] + q8_lo[k];
        yhat[k] = hh.sign[k] * cq;
    }
    // trellis-code the comp_dim complement coords (pad to pow2), direction + exact norm.
    let pad = comp_dim.next_power_of_two();
    let mut z = vec![0.0f32; pad];
    z[..comp_dim].copy_from_slice(&y[r..r + comp_dim]);
    let zn = l2_norm(&z);
    if zn > 0.0 {
        if let Some(u) = ct.reconstruct_unit(&z) {
            for j in 0..comp_dim {
                yhat[r + j] = u[j] * zn;
            }
        }
    }
    hh.q(&yhat) // map back to ambient
}

/// Per-coefficient 8-bit range fit (Python `q8`: per-column min/max → 255 levels).
fn fit_q8(
    data: &[Vec<f32>],
    basis: &[f32],
    d: usize,
    r: usize,
    levels: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut lo = vec![f32::INFINITY; r];
    let mut hi = vec![f32::NEG_INFINITY; r];
    for x in data {
        for k in 0..r {
            let bk = &basis[k * d..(k + 1) * d];
            let mut c = 0.0f32;
            for j in 0..d {
                c += x[j] * bk[j];
            }
            lo[k] = lo[k].min(c);
            hi[k] = hi[k].max(c);
        }
    }
    let scale: Vec<f32> = (0..r)
        .map(|k| {
            let s = (hi[k] - lo[k]) / levels;
            if s <= 0.0 {
                1.0
            } else {
                s
            }
        })
        .collect();
    (lo, scale)
}

/// Core reconstruction shared by fit-time selection and the backend: exact 8-bit head over the
/// top-`r` directions + the trellis on the orthogonal-complement residual (magnitude restored,
/// matching the Python `recon(residual)` = trellis_unit × ‖residual‖).
#[allow(clippy::too_many_arguments)]
fn recon_with(
    x: &[f32],
    d: usize,
    r: usize,
    basis: &[f32],
    q8_lo: &[f32],
    q8_scale: &[f32],
    trellis: &TrellisQuantizer,
    levels: f32,
) -> Vec<f32> {
    // Coefficients c_k = x · basis_k (exact), and the exact projection head_proj.
    let mut head_proj = vec![0.0f32; d];
    let mut head_q = vec![0.0f32; d];
    for k in 0..r {
        let bk = &basis[k * d..(k + 1) * d];
        let mut c = 0.0f32;
        for j in 0..d {
            c += x[j] * bk[j];
        }
        // Head quantization: clamp to the fit range, then round to the configured number of levels.
        let cl = c.clamp(q8_lo[k], q8_lo[k] + q8_scale[k] * levels);
        let cq = ((cl - q8_lo[k]) / q8_scale[k]).round() * q8_scale[k] + q8_lo[k];
        for j in 0..d {
            head_proj[j] += c * bk[j];
            head_q[j] += cq * bk[j];
        }
    }
    // Residual = x − exact-projection; trellis-reconstruct it, restoring magnitude.
    let resid: Vec<f32> = (0..d).map(|j| x[j] - head_proj[j]).collect();
    let rn = l2_norm(&resid);
    let mut out = head_q;
    if rn > 0.0 {
        if let Some(u) = trellis.reconstruct_unit(&resid) {
            for j in 0..d {
                out[j] += u[j] * rn;
            }
        }
    }
    out
}

/// Budget-driven adaptive selection: over (rank r, bit-rate b) with `dehub_bytes <= budget`, pick
/// the config maximizing held-out geom-Recall@10 (ties → fewest bytes). Returns (r, b). Falls to
/// r=0 (plain trellis) when the head does not earn its bytes. Held-out split = a slice of the fit
/// set as pseudo-queries with fp32 cosine gold against the rest.
#[allow(clippy::too_many_arguments)]
fn select_config(
    dim: usize,
    budget: usize,
    data: &[Vec<f32>],
    basis: &[f32],
    q8_lo: &[f32],
    q8_scale: &[f32],
    r_have: usize,
    levels: f32,
    config: &DehubConfig,
) -> (usize, u8) {
    let n = data.len();
    // The caller verifies that at least 1 bit fits. With too little validation
    // data, spend as much of the budget as possible on the plain trellis.
    let bset: Vec<u8> = (1u8..=4)
        .filter(|&b| dehub_bytes_with_config(dim, 0, b, config) <= budget)
        .collect();
    let bfloor = *bset.last().expect("validated minimum PCA-head budget");
    if n < 50 {
        return (0, bfloor);
    }
    let nval = (n / 5).clamp(1, 500);
    let val = &data[n - nval..];
    let dbsel: Vec<&Vec<f32>> = data[..n - nval]
        .iter()
        .step_by(((n - nval) / 2500).max(1))
        .collect();
    let gold: Vec<Vec<usize>> = val.par_iter().map(|qv| topk_dot(qv, &dbsel, 10)).collect();

    // Candidate (r, b) within budget. One trellis per distinct b.
    let mut best = (0usize, bfloor);
    let mut best_g = -1.0f64;
    let mut best_bytes = usize::MAX;
    for &b in &bset {
        for &r in CANDIDATES
            .iter()
            .filter(|&&r| r <= r_have && dehub_bytes_with_config(dim, r, b, config) <= budget)
        {
            let rb = &basis[..r * dim];
            let trellis = TrellisQuantizer::new(dim, b);
            let complement = if config.complement && r > 0 && r < dim {
                let comp_dim = dim - r;
                Some((
                    Householder::build(rb, dim, r),
                    TrellisQuantizer::new(comp_dim.next_power_of_two(), b),
                    comp_dim,
                ))
            } else {
                None
            };
            let reconstruct = |x: &[f32]| match &complement {
                Some((hh, ct, comp_dim)) => {
                    recon_complement(x, dim, r, q8_lo, q8_scale, hh, ct, *comp_dim, levels)
                }
                None => recon_with(x, dim, r, rb, q8_lo, q8_scale, &trellis, levels),
            };
            let rec_db: Vec<Vec<f32>> = dbsel.par_iter().map(|x| reconstruct(x)).collect();
            let rec_db_ref: Vec<&Vec<f32>> = rec_db.iter().collect();
            let hit: f64 = val
                .par_iter()
                .zip(gold.par_iter())
                .map(|(qv, g)| {
                    let qr = reconstruct(qv);
                    let got = topk_dot(&qr, &rec_db_ref, 10);
                    let gs: std::collections::HashSet<usize> = g.iter().copied().collect();
                    got.iter().filter(|i| gs.contains(i)).count() as f64 / 10.0
                })
                .sum();
            let g = hit / val.len() as f64;
            let bytes = dehub_bytes_with_config(dim, r, b, config);
            if g > best_g + 1e-9 || (g > best_g - 1e-9 && bytes < best_bytes) {
                best_g = g;
                best = (r, b);
                best_bytes = bytes;
            }
        }
    }
    best
}

/// Top-`k` indices into `db` by cosine to `q` (db + q normalized internally).
fn topk_dot(q: &[f32], db: &[&Vec<f32>], k: usize) -> Vec<usize> {
    let qn = l2_norm(q).max(f32::EPSILON);
    let mut scored: Vec<(usize, f32)> = db
        .iter()
        .enumerate()
        .map(|(i, x)| {
            let xn = l2_norm(x).max(f32::EPSILON);
            let mut s = 0.0f32;
            for j in 0..q.len() {
                s += q[j] * x[j];
            }
            (i, s / (qn * xn))
        })
        .collect();
    let k = k.min(scored.len());
    if k == 0 {
        return Vec::new();
    }
    scored.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(k);
    scored.into_iter().map(|(i, _)| i).collect()
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

impl VectorBackend for DehubBackend {
    fn dimensions(&self) -> usize {
        self.dim
    }
    fn len(&self) -> usize {
        self.entries.len()
    }
    fn add(&mut self, id: ItemId, embedding: &[f32]) {
        assert_eq!(embedding.len(), self.dim);
        let mut recon = self.recon_one(embedding);
        let norm = l2_norm(&recon);
        if norm > f32::EPSILON {
            recon.iter_mut().for_each(|x| *x /= norm);
        }
        self.entries.push((id, recon));
    }
    fn search(&self, query: &[f32], limit: usize) -> Vec<(ItemId, f32)> {
        assert_eq!(query.len(), self.dim);
        if limit == 0 || self.entries.is_empty() {
            return Vec::new();
        }
        let qnorm = l2_norm(query);
        if qnorm <= f32::EPSILON {
            return Vec::new();
        }
        let mut results: Vec<(ItemId, f32)> = self
            .entries
            .iter()
            .map(|(id, recon)| {
                let score = query.iter().zip(recon).map(|(q, x)| q * x).sum::<f32>() / qnorm;
                (*id, score)
            })
            .collect();
        results.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        results.truncate(limit);
        results
    }
    fn mem_bytes(&self) -> usize {
        // Logical compressed footprint. The fp32 reconstructions in `entries`
        // are the query-time decode cache, consistent with the trellis/IVF
        // memory-reporting convention used by the artifact.
        self.bytes_per_vec * self.entries.len()
    }
    fn memory_breakdown(&self) -> MemoryBreakdown {
        let trellis_model = self.trellis.memory_breakdown().model_bytes;
        let complement_model = self.complement.as_ref().map_or(0, |(hh, trellis, _)| {
            hh.allocated_bytes() + trellis.memory_breakdown().model_bytes
        });
        MemoryBreakdown {
            // The current search backend keeps decoded vectors. This column is
            // the serialized representation required by the deployment design;
            // counting it together with the cache is deliberately conservative.
            code_bytes: self.mem_bytes(),
            model_bytes: self.basis.capacity() * std::mem::size_of::<f32>()
                + self.q8_lo.capacity() * std::mem::size_of::<f32>()
                + self.q8_scale.capacity() * std::mem::size_of::<f32>()
                + trellis_model
                + complement_model,
            index_bytes: self.entries.capacity() * std::mem::size_of::<(ItemId, Vec<f32>)>(),
            cache_bytes: self
                .entries
                .iter()
                .map(|(_, recon)| recon.capacity() * std::mem::size_of::<f32>())
                .sum(),
        }
    }
    fn is_approximate(&self) -> bool {
        true
    }
    fn reconstruct_unit(&self, x: &[f32]) -> Option<Vec<f32>> {
        let mut recon = self.recon_one(x);
        let norm = l2_norm(&recon);
        if norm > f32::EPSILON {
            recon.iter_mut().for_each(|v| *v /= norm);
        }
        Some(recon)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn anisotropic(n: usize, d: usize, scales: &[f64], seed: u64) -> Vec<Vec<f32>> {
        let mut st = seed;
        (0..n)
            .map(|_| {
                let v: Vec<f32> = (0..d)
                    .map(|i| {
                        let u1 = next_f64(&mut st).max(1e-12);
                        let u2 = next_f64(&mut st);
                        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                        (z * scales[i]) as f32
                    })
                    .collect();
                let nn = l2_norm(&v).max(f32::EPSILON);
                v.iter().map(|x| x / nn).collect()
            })
            .collect()
    }

    #[test]
    fn top_r_pca_recovers_dominant_directions() {
        // Energy on axes 0,1; subspace iteration should recover a basis whose top-2 rows span
        // {e0,e1} (captured-variance fraction ~ matches), even though individual vectors may mix.
        let d = 8;
        let data = anisotropic(2000, d, &[6.0, 4.0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4], 7);
        let (eig, basis) = top_r_pca(&data, d, 4, 30, 1);
        assert!(
            eig[0] >= eig[1] && eig[1] >= eig[2],
            "eigvals not descending"
        );
        // The top-2 Ritz basis should live almost entirely in the {e0,e1} plane.
        for k in 0..2 {
            let leak: f32 = (2..d).map(|j| basis[k * d + j].powi(2)).sum();
            assert!(
                leak < 0.05,
                "top eigvec {k} leaks {leak} outside the energetic plane"
            );
        }
    }

    #[test]
    fn budget_picks_head_on_hub_and_skips_on_isotropic() {
        // At a fixed byte budget: a strong hub ⇒ spend bytes on the exact head (r>0); isotropic
        // ⇒ no head earns its bytes vs more trellis bits (r=0). d=64 so a budget admits b=2 + head.
        let d = 64;
        let mut hub_scales = vec![1.0f64; d];
        for (i, s) in hub_scales.iter_mut().enumerate().take(8) {
            *s = 12.0 - i as f64; // 12,11,...,5 on the first 8 axes — a strong hub
        }
        let hub = anisotropic(1500, d, &hub_scales, 3);
        // budget = b=2 (16B) + 2 state + up to ~16 head dims.
        let budget = dehub_bytes(d, 16, 2);
        let be = DehubBackend::fit_budget(d, budget, &hub);
        assert!(
            be.r > 0,
            "budget PCA-head should buy a head on a strong hub, got r={}",
            be.r
        );

        let iso = anisotropic(1500, vec![1.0f64; d].len(), &vec![1.0f64; d], 9);
        let be2 = DehubBackend::fit_budget(d, budget, &iso);
        assert_eq!(
            be2.r, 0,
            "budget PCA-head should skip the head on isotropic data, got r={}",
            be2.r
        );
    }

    #[test]
    fn householder_qt_q_roundtrip_and_axis_map() {
        // Qᵀ then Q reconstructs x; and Qᵀ b_k = e_k (head subspace maps to the first r axes).
        let d = 16;
        let data = anisotropic(
            800,
            d,
            &[
                6.0, 4.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ],
            5,
        );
        let (_e, basis) = top_r_pca(&data, d, 4, 30, 1);
        let hh = Householder::build(&basis, d, 4);
        let x: Vec<f32> = (0..d).map(|i| (i as f32 * 0.3 - 2.0).sin()).collect();
        let back = hh.q(&hh.qt(&x));
        let err: f32 = (0..d).map(|i| (back[i] - x[i]).powi(2)).sum::<f32>().sqrt();
        assert!(err < 1e-4, "Householder Qᵀ/Q round-trip error {err}");
        for k in 0..4 {
            let y = hh.qt(&basis[k * d..(k + 1) * d]); // Qᵀ b_k should be ±e_k
            for (i, &yi) in y.iter().enumerate() {
                let want = if i == k { yi.abs() } else { 0.0 }; // allow sign on the k-th
                assert!(
                    (yi.abs() - want).abs() < 1e-3 || i == k,
                    "Qᵀ b_{k} leaks at coord {i}: {yi}"
                );
            }
            assert!(
                y[k].abs() > 0.99,
                "Qᵀ b_{k} should map to ±e_{k}, got {}",
                y[k]
            );
        }
    }

    #[test]
    fn complement_reconstructs_a_hub_vector() {
        // The complement codec should reconstruct a hub-heavy vector about as well as the
        // full PCA-head residual, while coding pad2(d-r) dims. Compare cosine to the true unit vector.
        let d = 64;
        let mut scales = vec![1.0f64; d];
        for (i, s) in scales.iter_mut().enumerate().take(8) {
            *s = 10.0 - i as f64;
        }
        let data = anisotropic(1500, d, &scales, 7);
        let probe = &data[0];
        let full_config = DehubConfig {
            rank: 16,
            ..DehubConfig::default()
        };
        let compact_config = DehubConfig {
            complement: true,
            ..full_config
        };
        let full = DehubBackend::fit_with_config(d, 2, &data, full_config);
        let comp = DehubBackend::fit_with_config(d, 2, &data, compact_config);
        let cos = |a: &[f32], b: &[f32]| {
            let (mut d, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
            for i in 0..a.len() {
                d += a[i] * b[i];
                na += a[i] * a[i];
                nb += b[i] * b[i];
            }
            d / (na.sqrt() * nb.sqrt() + 1e-9)
        };
        let cf = cos(&full.recon_one(probe), probe);
        let cc = cos(&comp.recon_one(probe), probe);
        assert!(cc > 0.9, "complement recon cosine too low: {cc}");
        assert!(
            (cc - cf).abs() < 0.1,
            "complement {cc} should track full-residual {cf}"
        );
        // the complement codes fewer residual dims: pad2(64-16)=64 here (no saving at r=16), but at
        // r=32 -> pad2(32)=32; assert the byte accounting reflects it.
        assert!(dehub_bytes_with_config(64, 32, 2, &compact_config) < dehub_bytes_full(64, 32, 2));
    }

    #[test]
    fn backend_add_search_and_memory_are_functional() {
        let d = 8;
        let data = vec![
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let config = DehubConfig {
            rank: 0,
            ..DehubConfig::default()
        };
        let mut backend = DehubBackend::build(d, 2, 0, &data, "test", config);
        assert_eq!(backend.len(), 0);
        assert_eq!(backend.mem_bytes(), 0);
        backend.add(10, &data[0]);
        backend.add(20, &data[1]);
        assert_eq!(backend.len(), 2);
        assert_eq!(backend.mem_bytes(), 2 * dehub_bytes(d, 0, 2));
        let memory = backend.memory_breakdown();
        assert_eq!(memory.code_bytes, backend.mem_bytes());
        assert!(memory.model_bytes > 0);
        assert_eq!(memory.cache_bytes, 2 * d * std::mem::size_of::<f32>());
        assert!(memory.total_resident_bytes() > backend.mem_bytes());
        let got = backend.search(&data[0], 1);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, 10);
    }

    #[test]
    #[should_panic(expected = "minimum is")]
    fn budget_below_minimum_is_rejected() {
        let d = 64;
        let minimum = dehub_bytes(d, 0, 1);
        let _ = DehubBackend::fit_budget(d, minimum - 1, &[]);
    }

    #[test]
    fn accepted_budget_is_never_exceeded() {
        let d = 64;
        let data = anisotropic(32, d, &vec![1.0; d], 11);
        let budget = dehub_bytes(d, 0, 2);
        let backend = DehubBackend::fit_budget(d, budget, &data);
        assert!(backend.bytes_per_vec <= budget);
    }

    #[test]
    fn explicit_config_controls_byte_accounting_without_environment_state() {
        let full = DehubConfig {
            head_bits: 8,
            complement: false,
            ..DehubConfig::default()
        };
        let compact = DehubConfig {
            head_bits: 4,
            complement: true,
            ..DehubConfig::default()
        };
        assert_eq!(dehub_bytes_with_config(128, 32, 2, &full), 66);
        assert_eq!(dehub_bytes_with_config(128, 32, 2, &compact), 50);
    }

    #[test]
    fn invalid_explicit_config_is_rejected() {
        let config = DehubConfig {
            head_bits: 0,
            ..DehubConfig::default()
        };
        assert!(config.validate().is_err());
    }

    fn dehub_bytes_full(dim: usize, r: usize, b: u8) -> usize {
        (b as usize * dim).div_ceil(8) + 2 + (r * 8).div_ceil(8)
    }
}
