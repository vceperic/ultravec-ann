//! E8 (Gosset) lattice quantizer for comparing an 8-D structured lattice with
//! scalar and long-memory trellis quantization.
//!
//! Prior art (QTIP, NeurIPS'24, reported MSE @2-bit on the iid-Gaussian source the
//! rotation produces): scalar 0.118 → **E8 0.089** → QTIP-trellis 0.069 → RD-limit
//! 0.063. So E8 is expected to sit *between* scalar and the high-memory trellis —
//! it recovers the 8-D space-filling (shaping) gain a scalar code misses, but less
//! than the trellis's long memory. The artifact evaluates it on embeddings, on the g /
//! recall axis (not just MSE), at matched bytes.
//!
//! Why a standalone VQ and not E8-in-the-trellis: a V=8 trellis step has
//! `2^(bits·8)` emissions (65536 at 2-bit) — the exact-Viterbi emission loop is
//! computationally infeasible. The lattice's whole point is that the nearest point
//! is found in O(8) (Conway-Sloane) instead of by searching `2^(bits·8)` codewords,
//! so the implementation uses per-block nearest-E8 with no Viterbi pass.
//!
//! Construction (fully oblivious — no data, like the trellis computed code): the
//! codebook is the `2^(8·bits)` lowest-norm E8 points, scaled to unit per-coordinate
//! variance (Gaussian-shaped *density*, lattice-regular *local* arrangement — the
//! textbook optimal-VQ structure). Encode = nearest codeword per 8-D block of the
//! rotated+standardized vector; score = RaBitQ's rescaled estimator `⟨q_r,x̃⟩/⟨x̃,u_r⟩`,
//! identical to the trellis so the head-to-head isolates the quantizer alone.

use crate::{
    l2_norm, pack_indices_wide, std_about_mean, unpack_indices_wide, ItemId, MemoryBreakdown,
    Rotor, VectorBackend,
};

/// Nearest E8 lattice point to an 8-vector `y` (Conway & Sloane). E8 = D8 ∪ (D8+½):
/// decode to the nearest point of each coset, return the closer. Used for the
/// validity test and an optional fast encode path; the codebook encode is exact
/// brute force. Returns standard E8 coordinates (all-integer or all-half-integer,
/// even coordinate sum).
pub fn e8_nearest(y: &[f32]) -> [f32; 8] {
    let a = nearest_d8(y, 0.0);
    let b = nearest_d8(y, 0.5);
    let da: f32 = y.iter().zip(&a).map(|(p, q)| (p - q) * (p - q)).sum();
    let db: f32 = y.iter().zip(&b).map(|(p, q)| (p - q) * (p - q)).sum();
    if da <= db {
        a
    } else {
        b
    }
}

/// Nearest point of the `D8 + offset` coset (offset ∈ {0, ½}). Round each coord to
/// the offset grid; if the integer part's sum is odd (not in D8), flip the single
/// least-confidently-rounded coord to its second-nearest grid point (parity fix →
/// the nearest D8-coset point).
fn nearest_d8(y: &[f32], offset: f32) -> [f32; 8] {
    let mut pt = [0f32; 8];
    let mut isum: i64 = 0;
    let mut worst = 0usize;
    let mut worst_err = -1f32;
    let mut worst_dir = 1f32;
    for i in 0..8 {
        let s = y[i] - offset;
        let r = s.round();
        pt[i] = r + offset;
        isum += r as i64;
        let resid = s - r;
        let err = resid.abs();
        if err > worst_err {
            worst_err = err;
            worst = i;
            worst_dir = if resid >= 0.0 { 1.0 } else { -1.0 };
        }
    }
    if isum.rem_euclid(2) != 0 {
        pt[worst] += worst_dir;
    }
    pt
}

/// Enumerate all E8 points with `‖x‖² ≤ r2_max`, returned as **raw doubled** integer
/// coords `R_i = 2·x_i`. E8 in raw coords: all `R_i` the same parity (all even =
/// integer coset, all odd = half-integer coset) and `ΣR_i ≡ 0 (mod 4)` (the even-sum
/// rule). `ΣR_i² ≤ 4·r2_max`.
pub(crate) fn enumerate_e8_raw(r2_max: f32) -> Vec<[i16; 8]> {
    let budget = (4.0 * r2_max).floor() as i64;
    let mut out: Vec<[i16; 8]> = Vec::new();
    // Two parities: even (integer coset) and odd (half-integer coset).
    for parity in [0i64, 1] {
        let mut cur = [0i16; 8];
        enum_rec(0, budget, 0, 0, parity, &mut cur, &mut out);
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn enum_rec(
    pos: usize,
    sumsq_left: i64,
    sum_so_far: i64,
    _depth: usize,
    parity: i64,
    cur: &mut [i16; 8],
    out: &mut Vec<[i16; 8]>,
) {
    if pos == 8 {
        if sum_so_far.rem_euclid(4) == 0 {
            out.push(*cur);
        }
        return;
    }
    // R_pos ranges over integers of the chosen parity with R_pos² ≤ sumsq_left.
    let bound = (sumsq_left as f64).sqrt().floor() as i64;
    let mut r = -bound;
    // align r to the parity (r ≡ parity mod 2)
    while r.rem_euclid(2) != parity {
        r += 1;
    }
    while r <= bound {
        let r2 = r * r;
        if r2 <= sumsq_left {
            cur[pos] = r as i16;
            enum_rec(
                pos + 1,
                sumsq_left - r2,
                sum_so_far + r,
                _depth + 1,
                parity,
                cur,
                out,
            );
        }
        r += 2; // keep parity
    }
}

/// Enumerate E8 points, in raw doubled coordinates, whose squared distance to
/// `centre` is at most `budget`. `centre` is `2y` for a lattice-space query `y`, so
/// the budget is `(2d)^2` for a search radius `d`.
///
/// Same lattice rules as [`enumerate_e8_raw`] -- all coordinates share a parity and
/// the coordinate sum is `0 mod 4` -- but bounded by distance from a point rather
/// than by norm. E8 has determinant one, so a radius-`d` ball holds about
/// `pi^4 d^8 / 24` points: four at `d=1`, a thousand at `d=2`. That is what makes
/// the exact fallback affordable against a 2^24-point codebook.
pub(crate) fn enumerate_e8_ball(centre: &[f32; 8], budget: f64) -> Vec<[i16; 8]> {
    let mut out: Vec<[i16; 8]> = Vec::new();
    for parity in [0i64, 1] {
        let mut cur = [0i16; 8];
        ball_rec(0, budget, 0, parity, centre, &mut cur, &mut out);
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn ball_rec(
    pos: usize,
    budget_left: f64,
    sum_so_far: i64,
    parity: i64,
    centre: &[f32; 8],
    cur: &mut [i16; 8],
    out: &mut Vec<[i16; 8]>,
) {
    if pos == 8 {
        if sum_so_far.rem_euclid(4) == 0 {
            out.push(*cur);
        }
        return;
    }
    let c = centre[pos] as f64;
    let span = budget_left.max(0.0).sqrt();
    let lo = (c - span).floor() as i64;
    let hi = (c + span).ceil() as i64;
    let mut r = lo;
    while r.rem_euclid(2) != parity {
        r += 1;
    }
    while r <= hi {
        let d = r as f64 - c;
        let d2 = d * d;
        if d2 <= budget_left {
            cur[pos] = r as i16;
            ball_rec(
                pos + 1,
                budget_left - d2,
                sum_so_far + r,
                parity,
                centre,
                cur,
                out,
            );
        }
        r += 2;
    }
}

struct EVec {
    codes: Vec<u8>, // (dim/8) block indices, each 8·bits wide, packed
    rescale: f32,   // ⟨x̃, u_r⟩  (rescaled-estimator denominator)
    rnorm: f32,     // ‖u − c‖ for centered decomposition (1.0 ⇒ off)
    cidx: u32,      // assigned centroid index (0 ⇒ off / single global mean)
}

/// Oblivious E8 lattice vector quantizer. Mirrors the trellis pipeline
/// (rotate → standardize → quantize → rescaled estimator) with per-8-D-block
/// nearest-lattice quantization in place of the Viterbi.
pub struct E8Quantizer {
    dim: usize,
    #[allow(dead_code)] // rate in bits/coord; folded into blk_bits at construction
    bits: u8, // rate in bits/coordinate; index per block = 8·bits bits
    blk_bits: u8, // 8·bits — index width per 8-D block
    rotation: Rotor,
    /// Codebook in **unit-per-coord-variance** space: `2^(8·bits)` E8 points,
    /// the lowest-norm ones, mean-centered and variance-normalized to match the
    /// standardized N(0,1) source. Flat: `book[i*8 .. i*8+8]`.
    book: Vec<f32>,
    /// The same points as `book`, in raw doubled integer coordinates and in the
    /// same order: sorted by squared norm, then lexically. Retained so the exact
    /// encoder can resolve a Conway-Sloane hit to its index by binary search
    /// instead of scanning the codebook.
    raw: Vec<[i16; 8]>,
    /// Affine map from codebook space back to lattice space: `x = v*std + mean`.
    scale_mean: f32,
    scale_std: f32,
    n_code: usize,
    centroids: Vec<Vec<f32>>,
    entries: Vec<(ItemId, EVec)>,
}

impl E8Quantizer {
    pub fn new(dim: usize, bits: u8) -> Self {
        assert!(
            dim.is_multiple_of(8),
            "E8 quantizer requires dim % 8 == 0 (got {dim})"
        );
        // Three bits is a 2^24-point codebook (537 MB) and a 24-bit block index.
        // Four would be 2^32 points, 137 GB, so the ceiling is the enumeration
        // rather than the lattice: nearest-point search in E8 is closed-form.
        assert!(
            (1..=3).contains(&bits),
            "E8 quantizer supports bits ∈ {{1,2,3}} (2^(8·bits) codebook); got {bits}"
        );
        let n_code = 1usize << (8 * bits as usize);
        // Enumerate E8 points by growing radius until we have ≥ n_code, then keep
        // the n_code lowest-norm (deterministic: norm², then raw coords lexically).
        let mut r2 = 2.0f32;
        let mut raw = enumerate_e8_raw(r2);
        while raw.len() < n_code {
            r2 += 2.0;
            raw = enumerate_e8_raw(r2);
        }
        raw.sort_by(|a, b| {
            let na: i64 = a.iter().map(|&x| (x as i64) * (x as i64)).sum();
            let nb: i64 = b.iter().map(|&x| (x as i64) * (x as i64)).sum();
            na.cmp(&nb).then_with(|| a.cmp(b))
        });
        raw.truncate(n_code);
        // Unit-variance scaling: x = R/2; subtract mean (~0 by E8 symmetry) and
        // divide by population std over all n_code·8 coordinates.
        let mut sum = 0.0f64;
        let mut sumsq = 0.0f64;
        for p in &raw {
            for &r in p {
                let x = r as f64 / 2.0;
                sum += x;
                sumsq += x * x;
            }
        }
        let cnt = (n_code * 8) as f64;
        let mean = sum / cnt;
        let std = ((sumsq / cnt) - mean * mean).max(1e-12).sqrt();
        let mut book = vec![0.0f32; n_code * 8];
        for (i, p) in raw.iter().enumerate() {
            for j in 0..8 {
                book[i * 8 + j] = ((p[j] as f64 / 2.0 - mean) / std) as f32;
            }
        }
        Self {
            dim,
            bits,
            blk_bits: 8 * bits,
            rotation: Rotor::new_oblivious(dim, crate::rotation_seed()),
            book,
            raw,
            scale_mean: mean as f32,
            scale_std: std as f32,
            n_code,
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

    /// Unit vector (or its unit residual to the assigned centroid) to quantize.
    /// (vector_to_quantize, rnorm, cidx). rnorm=1.0/cidx=0 when centering off.
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

    /// Squared-norm-then-lexical order, the order `raw` is sorted in.
    fn raw_key(p: &[i16; 8]) -> (i64, [i16; 8]) {
        (p.iter().map(|&x| (x as i64) * (x as i64)).sum(), *p)
    }

    /// Index of `p` in the truncated codebook, or `None` if it was not kept.
    fn index_of_raw(&self, p: &[i16; 8]) -> Option<usize> {
        let key = Self::raw_key(p);
        self.raw
            .binary_search_by(|c| Self::raw_key(c).cmp(&key))
            .ok()
    }

    /// Nearest codeword index for one 8-D (standardized) block.
    ///
    /// Conway-Sloane returns the nearest point of the *infinite* E8 lattice in
    /// O(8). The codebook is a subset of that lattice, so whenever the returned
    /// point was kept, it is necessarily also the nearest codeword and the answer
    /// is exactly what a full scan would give. When it was dropped by the
    /// low-norm truncation the nearest codeword is some other point, and only
    /// then does this fall back to scanning. The result is identical to the brute
    /// force it replaces; only the cost differs, which matters because the
    /// three-bit codebook is 537 MB and a scan streams all of it per block.
    fn nearest_block(&self, blk: &[f32]) -> usize {
        let mut y = [0.0f32; 8];
        for j in 0..8 {
            y[j] = blk[j] * self.scale_std + self.scale_mean;
        }
        let pt = e8_nearest(&y);
        let mut doubled = [0i16; 8];
        for j in 0..8 {
            doubled[j] = (pt[j] * 2.0).round() as i16;
        }
        if let Some(i) = self.index_of_raw(&doubled) {
            return i;
        }
        self.nearest_block_outside_shell(&y)
    }

    /// Exact nearest codeword when the infinite-lattice nearest point was dropped by
    /// the low-norm truncation.
    ///
    /// Scanning all `n_code` codewords is correct but streams the whole 537 MB table
    /// at three bits, which dominates encoding. Instead: find any codeword `c0` by
    /// shrinking `y` until Conway-Sloane lands inside the kept set, then enumerate
    /// every lattice point within `|y - c0|` of `y` and keep the nearest that was
    /// itself kept.
    ///
    /// This is exact, not approximate. The optimum `c*` satisfies
    /// `|y - c*| <= |y - c0|` because `c0` is itself a candidate, so the ball of that
    /// radius contains it, and enumerating the ball therefore yields a superset of the
    /// candidates. The scan remains as a guard for the rare query far enough outside
    /// the shell that its ball would be larger than the codebook.
    fn nearest_block_outside_shell(&self, y: &[f32; 8]) -> usize {
        let (seed, d2) = self.seed_inside_shell(y);
        // pi^4 d^8 / 24 points in a radius-d ball; bail out to the scan rather than
        // enumerate something larger than the table we are trying to avoid reading.
        let est = std::f64::consts::PI.powi(4) * (d2 as f64).powi(4) / 24.0;
        if !est.is_finite() || est > self.n_code as f64 {
            return self.nearest_block_exhaustive_lattice(y);
        }
        let mut centre = [0.0f32; 8];
        for j in 0..8 {
            centre[j] = y[j] * 2.0;
        }
        // A hair of slack so a codeword sitting exactly at the bound is not lost to
        // rounding; the result is a superset, which costs nothing but a comparison.
        let budget = 4.0 * d2 as f64 * (1.0 + 1e-6) + 1e-6;
        let mut best = seed;
        let mut best_d2 = d2;
        for p in enumerate_e8_ball(&centre, budget) {
            if let Some(i) = self.index_of_raw(&p) {
                let mut d = 0.0f32;
                for j in 0..8 {
                    let e = y[j] - p[j] as f32 / 2.0;
                    d += e * e;
                }
                if d < best_d2 {
                    best_d2 = d;
                    best = i;
                }
            }
        }
        best
    }

    /// Largest shrink of `y` whose Conway-Sloane point is still in the codebook,
    /// with its squared distance to `y`. The origin is the lowest-norm E8 point and
    /// is always kept, so this always returns a codeword.
    fn seed_inside_shell(&self, y: &[f32; 8]) -> (usize, f32) {
        let dist2 = |i: usize| -> f32 {
            let c = &self.raw[i];
            let mut d = 0.0f32;
            for j in 0..8 {
                let e = y[j] - c[j] as f32 / 2.0;
                d += e * e;
            }
            d
        };
        let origin = self.index_of_raw(&[0i16; 8]).unwrap_or(0);
        let (mut best, mut lo, mut hi) = (origin, 0.0f32, 1.0f32);
        for _ in 0..14 {
            let mid = 0.5 * (lo + hi);
            let mut z = [0.0f32; 8];
            for j in 0..8 {
                z[j] = y[j] * mid;
            }
            let pt = e8_nearest(&z);
            let mut doubled = [0i16; 8];
            for j in 0..8 {
                doubled[j] = (pt[j] * 2.0).round() as i16;
            }
            match self.index_of_raw(&doubled) {
                Some(i) => {
                    if dist2(i) < dist2(best) {
                        best = i;
                    }
                    lo = mid;
                }
                None => hi = mid,
            }
        }
        (best, dist2(best))
    }

    /// Full scan in lattice space; the guard path for a query too far outside the
    /// shell for ball enumeration to pay.
    fn nearest_block_exhaustive_lattice(&self, y: &[f32; 8]) -> usize {
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for (i, c) in self.raw.iter().enumerate() {
            let mut d = 0.0f32;
            for j in 0..8 {
                let e = y[j] - c[j] as f32 / 2.0;
                d += e * e;
            }
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        best
    }

    /// Brute-force nearest codeword index for one 8-D (standardized) block.
    #[cfg(test)]
    fn nearest_block_exhaustive(&self, blk: &[f32]) -> usize {
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for i in 0..self.n_code {
            let c = &self.book[i * 8..i * 8 + 8];
            let mut d = 0.0f32;
            for j in 0..8 {
                let e = blk[j] - c[j];
                d += e * e;
            }
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        best
    }

    fn encode(&self, o: &[f32]) -> EVec {
        let (r, rnorm, cidx) = self.prep(o);
        let u_r = self.rotation.apply(&r);
        // Standardize so the rotated coords match the unit-variance codebook; the
        // rescale below uses the ORIGINAL u_r so the per-vector scale cancels and
        // the cosine estimate stays scale-invariant (identical to the trellis).
        let scale = std_about_mean(&u_r).max(f32::EPSILON);
        let n_blk = self.dim / 8;
        let mut idx = vec![0u32; n_blk];
        let mut rescale = 0.0f32;
        for b in 0..n_blk {
            let base = b * 8;
            let mut blk = [0.0f32; 8];
            for j in 0..8 {
                blk[j] = u_r[base + j] / scale;
            }
            let ci = self.nearest_block(&blk);
            idx[b] = ci as u32;
            let c = &self.book[ci * 8..ci * 8 + 8];
            for j in 0..8 {
                rescale += c[j] * u_r[base + j];
            }
        }
        EVec {
            codes: pack_indices_wide(&idx, self.blk_bits),
            rescale: rescale.max(f32::EPSILON),
            rnorm,
            cidx,
        }
    }

    fn score(&self, q_r: &[f32], qm: f32, ev: &EVec, scratch: &mut [u32]) -> f32 {
        let n_blk = self.dim / 8;
        unpack_indices_wide(&ev.codes, self.blk_bits, &mut scratch[..n_blk]);
        let mut dot = 0.0f32;
        for b in 0..n_blk {
            let c = &self.book[scratch[b] as usize * 8..scratch[b] as usize * 8 + 8];
            let base = b * 8;
            for j in 0..8 {
                dot += q_r[base + j] * c[j];
            }
        }
        qm + ev.rnorm * (dot / ev.rescale)
    }
}

impl VectorBackend for E8Quantizer {
    fn dimensions(&self) -> usize {
        self.dim
    }
    fn len(&self) -> usize {
        self.entries.len()
    }
    fn add(&mut self, id: ItemId, embedding: &[f32]) {
        assert_eq!(embedding.len(), self.dim);
        let ev = self.encode(embedding);
        self.entries.push((id, ev));
    }
    fn add_batch(&mut self, embeddings: &[Vec<f32>]) {
        use rayon::prelude::*;
        let dim = self.dim;
        let this: &E8Quantizer = self;
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
        let mut scratch = vec![0u32; self.dim / 8];
        let mut results: Vec<(ItemId, f32)> = self
            .entries
            .iter()
            .map(|(id, ev)| {
                let qm = qc.get(ev.cidx as usize).copied().unwrap_or(0.0);
                (*id, self.score(&q_r, qm, ev, &mut scratch))
            })
            .collect();
        results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }
    fn mem_bytes(&self) -> usize {
        // codes (dim·bits bits) + rescale(4). No start state (E8 has no trellis
        // memory) ⇒ ⌈M/8⌉ B leaner than the trellis at matched bits/coord. Plus the
        // residual norm and centroid index when centering is on.
        let center = crate::centering_bytes(self.centroids.len());
        self.entries
            .iter()
            .map(|(_, ev)| ev.codes.len() + 4 + center)
            .sum()
    }
    fn memory_breakdown(&self) -> MemoryBreakdown {
        MemoryBreakdown {
            code_bytes: self.mem_bytes(),
            model_bytes: self.rotation.allocated_bytes()
                + self.book.capacity() * std::mem::size_of::<f32>()
                + self
                    .centroids
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
        let ev = self.encode(x);
        let n_blk = self.dim / 8;
        let mut scratch = vec![0u32; n_blk];
        unpack_indices_wide(&ev.codes, self.blk_bits, &mut scratch);
        let mut xbar = vec![0.0f32; self.dim];
        for b in 0..n_blk {
            let c = &self.book[scratch[b] as usize * 8..scratch[b] as usize * 8 + 8];
            for j in 0..8 {
                xbar[b * 8 + j] = c[j];
            }
        }
        let mut recon = self.rotation.apply_inverse(&xbar);
        let n = l2_norm(&recon).max(f32::EPSILON);
        recon.iter_mut().for_each(|v| *v /= n);
        if self.centroids.is_empty() {
            Some(recon)
        } else {
            let c = &self.centroids[ev.cidx as usize];
            let mut full: Vec<f32> = c
                .iter()
                .zip(&recon)
                .map(|(cc, rr)| cc + ev.rnorm * rr)
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

    fn is_e8(p: &[f32; 8]) -> bool {
        // all-integer or all-half-integer, and even coordinate sum.
        let all_int = p.iter().all(|&x| (x - x.round()).abs() < 1e-4);
        let all_half = p
            .iter()
            .all(|&x| ((x - 0.5) - (x - 0.5).round()).abs() < 1e-4);
        if !(all_int || all_half) {
            return false;
        }
        let sum: f32 = p.iter().sum();
        (sum - sum.round()).abs() < 1e-3 && (sum.round() as i64).rem_euclid(2) == 0
    }

    #[test]
    fn the_local_enumeration_fallback_is_exact_at_three_bits() {
        // The path the three-bit encoder actually spends its time in. Blocks are
        // drawn deliberately far from the origin so the Conway-Sloane point falls
        // OUTSIDE the kept shell and the fallback runs -- the previous agreement
        // test only covered one and two bits, which is how the three-bit fallback
        // rate went unmeasured and cost a day of compute.
        let q = E8Quantizer::new(8, 3);
        let mut state = 0x0BAD_C0FF_EE0D_DF00u64;
        let (mut forced, mut checked) = (0usize, 0usize);
        for _ in 0..300 {
            // inflate the radius so a good fraction land past the truncation
            let scale = 1.0 + 2.5 * ((state >> 33) as f32 / (1u64 << 31) as f32);
            let blk: Vec<f32> = (0..8)
                .map(|_| {
                    let u = (crate::splitmix64(&mut state) >> 11) as f32 / (1u64 << 53) as f32;
                    let v = (crate::splitmix64(&mut state) >> 11) as f32 / (1u64 << 53) as f32;
                    let u = u.max(1e-9f32);
                    scale * (-2.0f32 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
                })
                .collect();
            let mut y = [0.0f32; 8];
            for j in 0..8 {
                y[j] = blk[j] * q.scale_std + q.scale_mean;
            }
            let pt = e8_nearest(&y);
            let mut doubled = [0i16; 8];
            for j in 0..8 {
                doubled[j] = (pt[j] * 2.0).round() as i16;
            }
            if q.index_of_raw(&doubled).is_some() {
                continue; // fast path; not what this test is for
            }
            forced += 1;
            let fast = q.nearest_block(&blk);
            let slow = q.nearest_block_exhaustive(&blk);
            checked += 1;
            if fast != slow {
                let d = |i: usize| -> f32 {
                    let c = &q.book[i * 8..i * 8 + 8];
                    blk.iter().zip(c).map(|(a, b)| (a - b) * (a - b)).sum()
                };
                assert!(
                    (d(fast) - d(slow)).abs() < 1e-5,
                    "local enumeration returned a strictly worse codeword: \
                     fast d2={} slow d2={}",
                    d(fast),
                    d(slow)
                );
            }
        }
        assert!(
            forced >= 30,
            "only {forced} of 300 blocks exercised the fallback; the draw is too tame \
             to test what it claims to test"
        );
        assert_eq!(forced, checked);
    }

    #[test]
    fn the_fast_encoder_agrees_with_the_exhaustive_one() {
        // The Conway-Sloane path is an optimization, not a different quantizer:
        // every block must land on the same codeword a full scan would pick. If
        // this ever fails, every reported E8 cell moves.
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        for bits in [1u8, 2] {
            let q = E8Quantizer::new(8, bits);
            let mut fallbacks = 0usize;
            for _ in 0..4000 {
                let blk: Vec<f32> = (0..8)
                    .map(|_| {
                        // crude standard-normal-ish draw in the codebook's own space
                        let u = (crate::splitmix64(&mut state) >> 11) as f32 / (1u64 << 53) as f32;
                        let v = (crate::splitmix64(&mut state) >> 11) as f32 / (1u64 << 53) as f32;
                        let u = u.max(1e-9f32);
                        (-2.0f32 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
                    })
                    .collect();
                let fast = q.nearest_block(&blk);
                let slow = q.nearest_block_exhaustive(&blk);
                if fast != slow {
                    // Permit only exact distance ties, where either answer is nearest.
                    let d = |i: usize| -> f32 {
                        let c = &q.book[i * 8..i * 8 + 8];
                        blk.iter().zip(c).map(|(a, b)| (a - b) * (a - b)).sum()
                    };
                    assert!(
                        (d(fast) - d(slow)).abs() < 1e-6,
                        "fast encoder picked a strictly worse codeword at {bits} bits"
                    );
                }
                let mut y = [0.0f32; 8];
                for j in 0..8 {
                    y[j] = blk[j] * q.scale_std + q.scale_mean;
                }
                let pt = e8_nearest(&y);
                let mut doubled = [0i16; 8];
                for j in 0..8 {
                    doubled[j] = (pt[j] * 2.0).round() as i16;
                }
                if q.index_of_raw(&doubled).is_none() {
                    fallbacks += 1;
                }
            }
            // The fallback is correct but slow; if it dominates there is no speedup.
            assert!(
                fallbacks < 4000 / 2,
                "{bits} bits: {fallbacks}/4000 blocks fell back to the full scan"
            );
        }
    }

    #[test]
    fn e8_decoder_returns_valid_lattice_points() {
        // A spread of random 8-vectors must all decode to valid E8 points.
        let mut st = 7u64;
        for _ in 0..2000 {
            let y: Vec<f32> = (0..8)
                .map(|_| (crate::next_f64(&mut st) as f32 - 0.5) * 8.0)
                .collect();
            let p = e8_nearest(&y);
            assert!(is_e8(&p), "decoded point not in E8: {p:?} from {y:?}");
        }
    }

    #[test]
    fn e8_lattice_points_decode_to_themselves() {
        // Exact E8 points (both cosets) are their own nearest lattice point.
        for p in [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], // sum 0, even
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  // half-int, sum 4
            [0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5], // half-int, sum 2
        ] {
            assert!(is_e8(&p), "test point not E8: {p:?}");
            let d = e8_nearest(&p);
            let dist: f32 = p.iter().zip(&d).map(|(a, b)| (a - b) * (a - b)).sum();
            assert!(
                dist < 1e-6,
                "E8 point did not decode to itself: {p:?} -> {d:?}"
            );
        }
    }

    #[test]
    fn e8_decoder_beats_rounding_in_mse() {
        // E8 (lattice) quantization must have lower MSE than independent rounding
        // (Z^8) — the whole point of the lattice (lower normalized second moment).
        let mut st = 99u64;
        let (mut e8_mse, mut z8_mse) = (0.0f64, 0.0f64);
        let n = 20000;
        for _ in 0..n {
            let y: Vec<f32> = (0..8)
                .map(|_| (crate::next_f64(&mut st) as f32 - 0.5) * 6.0)
                .collect();
            let e = e8_nearest(&y);
            for k in 0..8 {
                let de = (y[k] - e[k]) as f64;
                e8_mse += de * de;
                let dz = (y[k] - y[k].round()) as f64;
                z8_mse += dz * dz;
            }
        }
        assert!(e8_mse < z8_mse, "E8 MSE {e8_mse} should beat Z^8 {z8_mse}");
    }

    #[test]
    fn codebook_is_valid_and_sized() {
        let q = E8Quantizer::new(64, 1);
        assert_eq!(q.n_code, 256);
        assert_eq!(q.book.len(), 256 * 8);
        // Unit per-coord variance (≈1) and zero mean (≈0) by construction.
        let m: f64 = q.book.iter().map(|&x| x as f64).sum::<f64>() / q.book.len() as f64;
        let v: f64 = q
            .book
            .iter()
            .map(|&x| (x as f64 - m) * (x as f64 - m))
            .sum::<f64>()
            / q.book.len() as f64;
        assert!(m.abs() < 1e-3, "codebook mean {m} not ~0");
        assert!((v - 1.0).abs() < 1e-3, "codebook var {v} not ~1");
    }

    #[test]
    fn e8_reconstruction_correlates_with_truth() {
        // Random unit vectors: g = ⟨ō, o⟩ must be well above chance.
        let mut st = 13u64;
        let dim = 64;
        let q = E8Quantizer::new(dim, 2);
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
        let mean_g = sg / n as f64;
        assert!(
            mean_g > 0.8,
            "mean g {mean_g} too low for 2-bit E8 on dim-64"
        );
    }
}
