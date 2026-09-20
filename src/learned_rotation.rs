//! Learned rotations as an alternative to the random blockwise-Hadamard
//! `Rotation`. The implementation evaluates whether a learned dense rotation—
//! **PCA** (decorrelate to the eigenbasis) or **ITQ** (Gong-Lazebnik iterative
//! quantization: rotate to minimize the low-bit hypercube quantization error) —
//! changes the trellis-over-RaBitQ margin. Both are data-dependent (one matrix
//! fit on a DB sample), so they break the "oblivious/training-free" headline and
//! are reported in their own bucket. All math is pure-Rust f64, BLAS-free.
//!
//! PCA concentrates variance in the leading eigendirections, while
//! in the top eigen-directions), the opposite of what the rescaled estimator
//! ITQ re-spreads coordinates toward the hypercube.

use crate::{l2_norm, next_f64};

/// Covariance of the unit-normalized, mean-subtracted sample. Row-major d×d, f64.
fn covariance(sample: &[Vec<f32>], d: usize) -> Vec<f64> {
    let n = sample.len().max(1);
    let units: Vec<Vec<f64>> = sample
        .iter()
        .map(|v| {
            let nn = l2_norm(v).max(f32::EPSILON) as f64;
            v.iter().map(|x| *x as f64 / nn).collect()
        })
        .collect();
    let mut mean = vec![0.0f64; d];
    for u in &units {
        for i in 0..d {
            mean[i] += u[i];
        }
    }
    for m in &mut mean {
        *m /= n as f64;
    }
    let mut cov = vec![0.0f64; d * d];
    for u in &units {
        for i in 0..d {
            let ui = u[i] - mean[i];
            if ui == 0.0 {
                continue;
            }
            for j in i..d {
                cov[i * d + j] += ui * (u[j] - mean[j]);
            }
        }
    }
    for i in 0..d {
        for j in i..d {
            let v = cov[i * d + j] / n as f64;
            cov[i * d + j] = v;
            cov[j * d + i] = v;
        }
    }
    cov
}

/// Symmetric cyclic-Jacobi eigensolver (f64, d×d row-major, `a` consumed).
/// Returns `(eigenvalues, eigenvectors)` sorted by eigenvalue DESCENDING, with the
/// eigenvectors ROW-MAJOR — row `k` is the k-th eigenvector. So `y = E·x` projects
/// `x` onto the eigenbasis (the PCA rotation).
pub fn jacobi_eigen(mut a: Vec<f64>, d: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0.0f64; d * d];
    for i in 0..d {
        v[i * d + i] = 1.0;
    }
    for _sweep in 0..80 {
        let mut off = 0.0f64;
        for p in 0..d {
            for q in (p + 1)..d {
                off += a[p * d + q] * a[p * d + q];
            }
        }
        if off.sqrt() <= 1e-10 {
            break;
        }
        for p in 0..d {
            for q in (p + 1)..d {
                let apq = a[p * d + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let app = a[p * d + p];
                let aqq = a[q * d + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // A := Jᵀ A J : rotate columns p,q then rows p,q.
                for k in 0..d {
                    let akp = a[k * d + p];
                    let akq = a[k * d + q];
                    a[k * d + p] = c * akp - s * akq;
                    a[k * d + q] = s * akp + c * akq;
                }
                for k in 0..d {
                    let apk = a[p * d + k];
                    let aqk = a[q * d + k];
                    a[p * d + k] = c * apk - s * aqk;
                    a[q * d + k] = s * apk + c * aqk;
                }
                // V := V J  (eigenvectors accumulate as columns).
                for k in 0..d {
                    let vkp = v[k * d + p];
                    let vkq = v[k * d + q];
                    v[k * d + p] = c * vkp - s * vkq;
                    v[k * d + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let raw: Vec<f64> = (0..d).map(|i| a[i * d + i]).collect();
    let mut idx: Vec<usize> = (0..d).collect();
    idx.sort_by(|&i, &j| {
        raw[j]
            .partial_cmp(&raw[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let eigvals: Vec<f64> = idx.iter().map(|&i| raw[i]).collect();
    let mut evec = vec![0.0f64; d * d]; // row k = eigenvector (column idx[k] of v)
    for (k, &i) in idx.iter().enumerate() {
        for r in 0..d {
            evec[k * d + r] = v[r * d + i];
        }
    }
    (eigvals, evec)
}

/// PCA rotation: the eigenbasis of the sample covariance. Row-major d×d, `y = E·x`.
pub fn pca_rotation(sample: &[Vec<f32>], d: usize) -> Vec<f32> {
    let cov = covariance(sample, d);
    let (_e, evec) = jacobi_eigen(cov, d);
    evec.iter().map(|&x| x as f32).collect()
}

/// A seeded random orthogonal matrix (Gram-Schmidt on a Gaussian), row-major d×d.
fn random_orthogonal(d: usize, seed: u64) -> Vec<f64> {
    let mut st = seed;
    let mut m = vec![0.0f64; d * d];
    for x in m.iter_mut() {
        let u1 = next_f64(&mut st).max(1e-12);
        let u2 = next_f64(&mut st);
        *x = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    }
    for i in 0..d {
        for j in 0..i {
            let mut dot = 0.0;
            for k in 0..d {
                dot += m[i * d + k] * m[j * d + k];
            }
            for k in 0..d {
                m[i * d + k] -= dot * m[j * d + k];
            }
        }
        let mut nrm = 0.0;
        for k in 0..d {
            nrm += m[i * d + k] * m[i * d + k];
        }
        let nrm = nrm.sqrt().max(1e-12);
        for k in 0..d {
            m[i * d + k] /= nrm;
        }
    }
    m
}

/// ITQ rotation (Gong & Lazebnik): PCA front-end, then iterate `B = sign(WR)` /
/// orthogonal-Procrustes `R = argmin‖B − WR‖` until the projected data aligns to
/// the {±1} hypercube. The Procrustes step uses one symmetric eigendecomposition
/// of `MᵀM` (M = BᵀW) — no separate SVD needed: `R = Σ_k u_k v_kᵀ`, `u_k = M v_k/σ_k`.
/// Returns the composed rotation `mat = Rᵀ·E` (row-major d×d, `y = mat·x`).
pub fn itq_rotation(sample: &[Vec<f32>], d: usize, iters: usize, seed: u64) -> Vec<f32> {
    let cov = covariance(sample, d);
    let (_e, ev) = jacobi_eigen(cov, d); // row k = eigenvector_k
                                         // Project a capped sample into the PCA basis: w_i = E · x_i.
    let cap = sample.len().min(8000);
    let w: Vec<Vec<f64>> = sample
        .iter()
        .take(cap)
        .map(|x| {
            let nn = l2_norm(x).max(f32::EPSILON) as f64;
            let xu: Vec<f64> = x.iter().map(|v| *v as f64 / nn).collect();
            (0..d)
                .map(|k| {
                    let mut s = 0.0;
                    for r in 0..d {
                        s += ev[k * d + r] * xu[r];
                    }
                    s
                })
                .collect()
        })
        .collect();
    let mut r = random_orthogonal(d, seed);
    for _ in 0..iters.max(1) {
        // M = Bᵀ W where B = sign(W R).  M_{kl} = Σ_i b_ik w_il.
        let mut m = vec![0.0f64; d * d];
        for wi in &w {
            let mut b = vec![0.0f64; d];
            for k in 0..d {
                let mut s = 0.0;
                for j in 0..d {
                    s += wi[j] * r[j * d + k];
                }
                b[k] = if s >= 0.0 { 1.0 } else { -1.0 };
            }
            for k in 0..d {
                let bk = b[k];
                for l in 0..d {
                    m[k * d + l] += bk * wi[l];
                }
            }
        }
        // MᵀM, eig → right singular vectors v_k (rows) + σ_k²; R = Σ u_k v_kᵀ.
        let mut mtm = vec![0.0f64; d * d];
        for k in 0..d {
            for l in k..d {
                let mut s = 0.0;
                for t in 0..d {
                    s += m[t * d + k] * m[t * d + l];
                }
                mtm[k * d + l] = s;
                mtm[l * d + k] = s;
            }
        }
        let (evals, vsv) = jacobi_eigen(mtm, d);
        let mut newr = vec![0.0f64; d * d];
        for k in 0..d {
            let sigma = evals[k].max(0.0).sqrt();
            if sigma < 1e-9 {
                continue;
            }
            // u_k = M v_k / σ_k ; accumulate R += u_k v_kᵀ.
            let mut u = vec![0.0f64; d];
            for a in 0..d {
                let mut s = 0.0;
                for cc in 0..d {
                    s += m[a * d + cc] * vsv[k * d + cc];
                }
                u[a] = s / sigma;
            }
            for a in 0..d {
                let ua = u[a];
                for b in 0..d {
                    newr[a * d + b] += ua * vsv[k * d + b];
                }
            }
        }
        r = newr;
    }
    // Compose: y = Rᵀ (E x) → mat = Rᵀ·E. mat[a][c] = Σ_k Rᵀ_{ak} E_{kc} = Σ_k r[k*d+a]·ev[k*d+c].
    let mut mat = vec![0.0f32; d * d];
    for a in 0..d {
        for c in 0..d {
            let mut s = 0.0;
            for k in 0..d {
                s += r[k * d + a] * ev[k * d + c];
            }
            mat[a * d + c] = s as f32;
        }
    }
    mat
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_orthonormal(mat: &[f32], d: usize) -> bool {
        // matᵀ·mat ≈ I (rows orthonormal ⇒ columns too for square).
        for i in 0..d {
            for j in 0..d {
                let mut s = 0.0f32;
                for k in 0..d {
                    s += mat[k * d + i] * mat[k * d + j];
                }
                let want = if i == j { 1.0 } else { 0.0 };
                if (s - want).abs() > 1e-3 {
                    return false;
                }
            }
        }
        true
    }

    fn anisotropic_gaussian(n: usize, d: usize, scales: &[f64], seed: u64) -> Vec<Vec<f32>> {
        let mut st = seed;
        (0..n)
            .map(|_| {
                (0..d)
                    .map(|i| {
                        let u1 = next_f64(&mut st).max(1e-12);
                        let u2 = next_f64(&mut st);
                        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                        (z * scales[i]) as f32
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn pca_is_orthonormal_and_recovers_axes() {
        let d = 6;
        // Strong variance on axis 0, weak elsewhere (after the random sample is
        // unit-normalized the ordering is preserved).
        let sample = anisotropic_gaussian(800, d, &[5.0, 3.0, 1.0, 1.0, 0.5, 0.5], 7);
        let mat = pca_rotation(&sample, d);
        assert!(is_orthonormal(&mat, d), "PCA rotation not orthonormal");
    }

    #[test]
    fn itq_is_orthonormal() {
        let d = 8;
        let sample = anisotropic_gaussian(1000, d, &[4.0, 3.0, 2.5, 2.0, 1.5, 1.0, 0.8, 0.5], 11);
        let mat = itq_rotation(&sample, d, 10, 42);
        assert!(is_orthonormal(&mat, d), "ITQ rotation not orthonormal");
    }
}
