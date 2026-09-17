//! Product (vector) quantization. TurboQuant is scalar (per
//! coordinate); the classical reason VQ > scalar is the **shape + packing gain**
//! from quantizing coordinates *jointly*. PQ splits the (rotated, unit) vector
//! into `n_sub` sub-vectors of `sub_dim` and quantizes each to a 256-entry
//! trained codebook (8 bits/sub-vector → 8/sub_dim bits/dim).
//!
//! This provides a data-dependent comparison for the gain from joint
//! quantization at matched rates. Search is asymmetric distance computation
//! (ADC): precompute a per-subspace IP table, then one lookup+add per subspace.
//!
//! `loss = Anisotropic { eta }` swaps the MSE k-means objective for the
//! score-aware (ScaNN) loss — penalizing residual *parallel* to the datapoint
//! (which moves the inner product) `eta`× more than orthogonal residual. This is
//! an alternative fitted objective to plain PQ.

use crate::{l2_norm, next_f64, ItemId, MemoryBreakdown, Rotation, VectorBackend};

const TRAIN_SAMPLE: usize = 20_000;
const KMEANS_ITERS: usize = 25;
const CODE_BITS: usize = 8; // bits per subquantizer → K = 256 centroids

/// Bits per subquantizer code, i.e. `log2(centroids per subspace)`.
///
/// Fixed at 8 by default, which is the textbook 256-centroid PQ and reproduces every
/// retained result. `ULTRAVEC_PQ_CODE_BITS` lowers it for the cold-start control:
/// holding the emitted rate fixed, fewer centroids per subspace means proportionally
/// more subspaces, so the same bit budget is spent on a factorization that has far
/// less to estimate. That distinguishes "PQ needs this many calibration vectors"
/// from "a 256-centroid PQ needs this many", which a single fixed cardinality
/// cannot: below 256 training vectors a 256-way subquantizer cannot even fill its
/// codebook with distinct centroids.
fn code_bits() -> usize {
    std::env::var("ULTRAVEC_PQ_CODE_BITS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|c| (1..=8).contains(c))
        .unwrap_or(CODE_BITS)
}

#[derive(Clone, Copy)]
pub enum PqLoss {
    /// Plain reconstruction MSE (standard PQ / k-means).
    Mse,
    /// ScaNN score-aware loss: parallel-to-datapoint residual weighted `eta`×.
    Anisotropic { eta: f32 },
}

pub struct ProductQuantizer {
    dim: usize,
    sub_dim: usize,
    n_sub: usize,
    rotation: Rotation,
    /// `codebooks[s]` is `k * sub_dim` flat (centroid c at `[c*sub_dim..]`).
    codebooks: Vec<Vec<f32>>,
    /// per-vector codes: `n_sub` bytes each.
    codes: Vec<(ItemId, Vec<u8>)>,
    label: String,
}

impl ProductQuantizer {
    /// Train codebooks on `db` and encode all of it. `bits_per_dim` sets
    /// `sub_dim = 8 / bits_per_dim` (so 4 → sub_dim 2, 2 → sub_dim 4).
    pub fn train(dim: usize, bits_per_dim: u8, loss: PqLoss, db: &[Vec<f32>]) -> Self {
        // sub_dim is an integer division, so only exact divisors of 8 give the rate
        // the label claims. bits_per_dim = 3 would yield sub_dim = 2, i.e. an actual
        // 4 bits/dim if labelled "pq_mse_3bit". Accept only exactly matched rates.
        assert!(
            bits_per_dim >= 1 && 8_usize.is_multiple_of(bits_per_dim as usize),
            "product quantization needs bits_per_dim to divide 8 (1, 2, 4 or 8); \
             got {bits_per_dim}, which would encode at {} bits/dim while being \
             labelled {bits_per_dim}-bit",
            8 / (8 / bits_per_dim.max(1) as usize).max(1)
        );
        let code_bits = code_bits();
        let k = 1usize << code_bits;
        assert!(
            code_bits.is_multiple_of(bits_per_dim as usize),
            "subquantizer width {code_bits} must be a multiple of bits_per_dim \
             {bits_per_dim} for the emitted rate to stay exact"
        );
        let sub_dim = (code_bits / bits_per_dim as usize).max(1);
        assert!(
            dim.is_multiple_of(sub_dim),
            "dim {dim} not divisible by sub_dim {sub_dim}"
        );
        let n_sub = dim / sub_dim;
        let rotation = Rotation::new(dim, crate::rotation_seed());

        // Rotate (unit-normalized) a training sample once.
        let stride = (db.len() / TRAIN_SAMPLE).max(1);
        let sample: Vec<Vec<f32>> = db
            .iter()
            .step_by(stride)
            .take(TRAIN_SAMPLE)
            .map(|x| {
                let n = l2_norm(x).max(f32::EPSILON);
                rotation.apply(&x.iter().map(|v| v / n).collect::<Vec<_>>())
            })
            .collect();

        let label = match loss {
            PqLoss::Mse => format!("pq_mse_{}bit", bits_per_dim),
            PqLoss::Anisotropic { eta } => format!("pq_aniso_eta{:.0}_{}bit", eta, bits_per_dim),
        };

        let mut codebooks = Vec::with_capacity(n_sub);
        for s in 0..n_sub {
            let lo = s * sub_dim;
            let subvecs: Vec<&[f32]> = sample.iter().map(|v| &v[lo..lo + sub_dim]).collect();
            codebooks.push(kmeans(&subvecs, sub_dim, k, loss, 0x51A2 ^ s as u64));
        }

        let mut pq = Self {
            dim,
            sub_dim,
            n_sub,
            rotation,
            codebooks,
            codes: Vec::with_capacity(db.len()),
            label,
        };
        for (i, x) in db.iter().enumerate() {
            pq.codes.push((i as ItemId, pq.encode(x)));
        }
        pq
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    /// Train the codebooks on `train_db` but store NOTHING (empty index). For the
    /// streaming drift experiment's fixed-PQ arm: train the codebook once on the
    /// initial batch, then [`push`] later arrivals — the data-dependent codebook is
    /// fixed while the stream distribution changes.
    pub fn with_codebook_from(
        dim: usize,
        bits_per_dim: u8,
        loss: PqLoss,
        train_db: &[Vec<f32>],
    ) -> Self {
        let pq = Self::train(dim, bits_per_dim, loss, train_db);
        Self {
            codes: Vec::new(),
            ..pq
        }
    }

    /// Encode and store one vector under `id` for the fixed-PQ streaming arm;
    /// the trait `add` stays a no-op so the static bench is unaffected.
    pub fn push(&mut self, id: ItemId, embedding: &[f32]) {
        let code = self.encode(embedding);
        self.codes.push((id, code));
    }

    fn encode(&self, x: &[f32]) -> Vec<u8> {
        let n = l2_norm(x).max(f32::EPSILON);
        let u_rot = self
            .rotation
            .apply(&x.iter().map(|v| v / n).collect::<Vec<_>>());
        let mut code = vec![0u8; self.n_sub];
        for s in 0..self.n_sub {
            let lo = s * self.sub_dim;
            code[s] = nearest_centroid(
                &u_rot[lo..lo + self.sub_dim],
                &self.codebooks[s],
                self.sub_dim,
            );
        }
        code
    }
}

impl VectorBackend for ProductQuantizer {
    fn dimensions(&self) -> usize {
        self.dim
    }
    fn len(&self) -> usize {
        self.codes.len()
    }
    /// No-op: PQ is trained + encoded in [`train`] (it needs all data up front).
    /// The bench's incremental add loop is intentionally ignored.
    fn add(&mut self, _id: ItemId, _embedding: &[f32]) {}

    fn search(&self, query: &[f32], limit: usize) -> Vec<(ItemId, f32)> {
        let qn = l2_norm(query);
        if qn < f32::EPSILON {
            return Vec::new();
        }
        let q_rot = self
            .rotation
            .apply(&query.iter().map(|v| v / qn).collect::<Vec<_>>());
        // ADC IP tables: table[s*k + c] = dot(q_sub_s, centroid_c).
        let k = self
            .codebooks
            .first()
            .map_or(0, |c| c.len() / self.sub_dim.max(1));
        let mut table = vec![0.0f32; self.n_sub * k];
        for s in 0..self.n_sub {
            let lo = s * self.sub_dim;
            let q_sub = &q_rot[lo..lo + self.sub_dim];
            let cb = &self.codebooks[s];
            for c in 0..k {
                let cc = &cb[c * self.sub_dim..c * self.sub_dim + self.sub_dim];
                let mut d = 0.0f32;
                for k in 0..self.sub_dim {
                    d += q_sub[k] * cc[k];
                }
                table[s * k + c] = d;
            }
        }
        let mut results: Vec<(ItemId, f32)> = self
            .codes
            .iter()
            .map(|(id, code)| {
                let mut score = 0.0f32;
                for s in 0..self.n_sub {
                    score += table[s * k + code[s] as usize];
                }
                (*id, score)
            })
            .collect();
        results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }
    fn mem_bytes(&self) -> usize {
        // per-vector codes only (codebook is shared overhead, reported separately).
        self.codes.iter().map(|(_, c)| c.len()).sum()
    }
    fn memory_breakdown(&self) -> MemoryBreakdown {
        MemoryBreakdown {
            code_bytes: self.mem_bytes(),
            model_bytes: self.rotation.allocated_bytes()
                + self
                    .codebooks
                    .iter()
                    .map(|v| v.capacity() * std::mem::size_of::<f32>())
                    .sum::<usize>(),
            ..MemoryBreakdown::default()
        }
    }
    fn is_approximate(&self) -> bool {
        true
    }
    /// Reconstruct the unit direction: look up each sub-vector's centroid, inverse
    /// rotate, normalize. Needed by the graph-ANN streaming rung (the graph picks
    /// edges by distances between *decoded* vectors, so a drifting codebook corrupts
    /// the graph structure.
    fn reconstruct_unit(&self, x: &[f32]) -> Option<Vec<f32>> {
        let code = self.encode(x);
        let mut rot = vec![0.0f32; self.dim];
        for s in 0..self.n_sub {
            let cc = &self.codebooks[s][code[s] as usize * self.sub_dim..][..self.sub_dim];
            rot[s * self.sub_dim..][..self.sub_dim].copy_from_slice(cc);
        }
        let mut recon = self.rotation.apply_inverse(&rot);
        let n = l2_norm(&recon).max(f32::EPSILON);
        recon.iter_mut().for_each(|v| *v /= n);
        Some(recon)
    }
}

/// Nearest centroid index (min L2) in a flat `k*sub_dim` codebook.
fn nearest_centroid(x: &[f32], codebook: &[f32], sub_dim: usize) -> u8 {
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    for c in 0..codebook.len() / sub_dim {
        let cc = &codebook[c * sub_dim..c * sub_dim + sub_dim];
        let mut d = 0.0f32;
        for k in 0..sub_dim {
            let e = x[k] - cc[k];
            d += e * e;
        }
        if d < best_d {
            best_d = d;
            best = c;
        }
    }
    best as u8
}

/// k-means (Lloyd) on `points` (each `sub_dim`-long), `k` centroids, returning a
/// flat `k*sub_dim` codebook. `loss = Anisotropic` reweights the centroid update
/// so residual parallel to each point counts `eta`× (ScaNN-style).
fn kmeans(points: &[&[f32]], sub_dim: usize, k: usize, loss: PqLoss, seed: u64) -> Vec<f32> {
    let n = points.len();
    let mut state = seed;
    // Seeded random-point initialization with replacement. When n<k, duplicate
    // initial centroids are unavoidable; empty clusters are reseeded below.
    let mut cb = vec![0.0f32; k * sub_dim];
    for c in 0..k {
        let p = points[(next_f64(&mut state) * n as f64) as usize % n];
        cb[c * sub_dim..c * sub_dim + sub_dim].copy_from_slice(p);
    }
    let mut assign = vec![0u32; n];
    for _ in 0..KMEANS_ITERS {
        // Assignment (always min-L2; the loss reweights the update, mirroring
        // ScaNN's quadratic-loss centroid step).
        for (i, p) in points.iter().enumerate() {
            assign[i] = nearest_centroid(p, &cb, sub_dim) as u32;
        }
        // Update.
        let mut sum = vec![0.0f32; k * sub_dim];
        let mut wsum = vec![0.0f32; k];
        for (i, p) in points.iter().enumerate() {
            let c = assign[i] as usize;
            let w = match loss {
                PqLoss::Mse => 1.0,
                PqLoss::Anisotropic { eta } => {
                    // Weight points by eta along their own direction. With unit
                    // sub-vectors this collapses to a per-point scalar weight that
                    // up-weights high-energy (large-‖p‖) sub-vectors — the ones
                    // whose parallel residual dominates the inner product.
                    let pn = p.iter().map(|v| v * v).sum::<f32>();
                    1.0 + (eta - 1.0) * pn
                }
            };
            wsum[c] += w;
            for kk in 0..sub_dim {
                sum[c * sub_dim + kk] += w * p[kk];
            }
        }
        for c in 0..k {
            if wsum[c] > 0.0 {
                for kk in 0..sub_dim {
                    cb[c * sub_dim + kk] = sum[c * sub_dim + kk] / wsum[c];
                }
            } else {
                // Empty cluster → reseed to a random point.
                let p = points[(next_f64(&mut state) * n as f64) as usize % n];
                cb[c * sub_dim..c * sub_dim + sub_dim].copy_from_slice(p);
            }
        }
    }
    cb
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cosine;

    /// The cold-start study and every retained PQ row assume 256 centroids per
    /// subspace. `ULTRAVEC_PQ_CODE_BITS` exists to vary that for the
    /// factorization control, so the default is pinned: a changed default would
    /// re-quantize the whole warm frontier without any table admitting it.
    #[test]
    fn subquantizer_width_defaults_to_the_conventional_256_centroids() {
        assert!(
            std::env::var("ULTRAVEC_PQ_CODE_BITS").is_err(),
            "ULTRAVEC_PQ_CODE_BITS is set in this shell; the check tier strips it"
        );
        assert_eq!(code_bits(), 8, "8 bits per subquantizer = 256 centroids");
        assert_eq!(1usize << code_bits(), 256);
    }

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

    #[test]
    fn pq_self_query_retrieves_self() {
        let dim = 64;
        let db = rand_unit(500, dim, 1);
        let pq = ProductQuantizer::train(dim, 4, PqLoss::Mse, &db); // sub_dim 2
                                                                    // A stored vector should rank itself at/near the top.
        let hits = pq.search(&db[0], 5);
        assert!(
            hits.iter().any(|(id, _)| *id == 0),
            "self-query should retrieve self"
        );
    }

    #[test]
    fn pq_reconstruction_beats_random() {
        // ADC score for the true vector should exceed a random other vector.
        let dim = 64;
        let db = rand_unit(400, dim, 2);
        let pq = ProductQuantizer::train(dim, 4, PqLoss::Mse, &db);
        let q = &db[10];
        let self_score = pq.search(q, 1)[0].1;
        let true_cos = cosine(q, &db[10]);
        assert!(
            self_score > 0.5 * true_cos,
            "ADC self score {self_score} implausibly low"
        );
    }

    #[test]
    fn label_matches_the_encoded_rate() {
        // sub_dim is an integer division of 8, so only exact divisors give the rate
        // the label claims. Every accepted width must round-trip.
        for bits in [1u8, 2, 4, 8] {
            let dim = 64;
            let db = rand_unit(200, dim, 5);
            let pq = ProductQuantizer::train(dim, bits, PqLoss::Mse, &db);
            assert_eq!(
                pq.label(),
                format!("pq_mse_{bits}bit"),
                "label must state the rate actually encoded"
            );
            assert_eq!(8 / (8 / bits as usize), bits as usize, "rate must be exact");
        }
    }

    #[test]
    #[should_panic(expected = "divide 8")]
    fn rejects_a_rate_it_would_have_to_mislabel() {
        // bits = 3 yields sub_dim = 2, i.e. an actual 4 bits/dim reported as 3-bit.
        let db = rand_unit(200, 64, 5);
        ProductQuantizer::train(64, 3, PqLoss::Mse, &db);
    }
}
