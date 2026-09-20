//! Trellis-coded VECTOR quantization over E8 cosets: a research arm.
//!
//! The scalar trellis buys long memory; the E8 lattice buys eight-dimensional
//! packing gain. They are orthogonal, and two-bit GIST -- the one flat cell the
//! trellis does not hold, where E8 leads .941 to .937 -- is the cell that motivates
//! combining them. This is Marcellin--Fischer trellis-coded quantization with a
//! lattice codebook, built on Ungerboeck set partitioning.
//!
//! The partition. E8 is split into eight cosets of an index-8 sublattice, labelled
//! by three bits `((R0-R1)/2, (R2-R3)/2, (R4-R5)/2) mod 2` in raw doubled
//! coordinates. A test checks the partition is closed (a coset plus a sublattice
//! point stays in the coset) and balanced on the low-norm shell (0.92 min/max), which
//! the parity-based partitions are not: the half-integer coset dominates the shell
//! and starves the others of codewords.
//!
//! The rule that gives the trellis its teeth. A step emits `k=2` branch bits, but a
//! state may only name the four cosets whose high label bit equals the state's low
//! bit. A step therefore chooses among four cosets while the trellis as a whole
//! draws on eight, and reaching the other half costs a transition the Viterbi must
//! have planned for. That is the entire gain of TCQ over memoryless VQ: a rate-k
//! trellis selecting from a rate-(k+1) codebook. Without the rule every state
//! reaches every coset, the per-block choice is free, and the encoder is E8
//! exactly -- measured bit-identical at M=1 and M=16 before the rule was added.
//!
//! Cost. Per step, `2^M` states times four branches, with the branch cost a
//! closed-form nearest-in-coset computed once per block for all eight cosets. At
//! M=16 that is the same order per vector as the scalar trellis. The recorded
//! reason for never doing this was that a V=8 step has `2^(8b)` emissions; that is
//! true only for an unstructured emission set.
//!
//! What it trades. The scalar trellis reconstructs through a computed Gaussian
//! table (marginal shaping); a lattice codebook is uniform in a ball (packing gain,
//! no marginal shaping). Whether memory plus packing beats memory plus shaping at
//! 1--3 bits was decided by measurement, with the pre-registered test being the
//! two-bit GIST cell: beat both parents there or it is a negative result.
//!
//! RESULT (2026-09-14): a measured negative. Two-bit GIST, 20k vectors, 500 queries,
//! M=16, seed 42: coset trellis .926 against E8 .929 and the scalar trellis .931.
//! SIFT 5k at the same rate: .654 against .677 and .728. Memory does bite once the
//! subset rule uses full-state parity (M=1 .647, M=8 .657, M=16 .654 on SIFT), so
//! this is the construction working as designed and losing, not a broken encoder.
//! Two reasons, both structural. Splitting the codebook into eight per-coset books
//! keeps `8 x 2^(m)` per-coset-lowest-norm points, a worse-packed union than E8's
//! `2^(m+3)` globally-lowest-norm points, so the lattice arm starts behind E8 before
//! memory is spent. And the gain memory then buys is small against the Gaussian
//! marginal the scalar trellis keeps and this arm gives up: at one to three bits
//! shaping is worth more than packing. Retained as an opt-in arm so the negative is
//! reproducible; not tabulated in the manuscript.
//!
//! Rates. A step emits `k+m` bits over eight coordinates, `(k+m)/8` per coordinate.
//! Each of the eight cosets keeps `2^(8b-k-1)` lowest-norm points: 32, 8,192 and
//! 2,097,152 at 1, 2, 3 bits. Four bits is beyond enumeration, as for E8 itself.

use crate::{
    e8::{e8_nearest, enumerate_e8_ball, enumerate_e8_raw},
    l2_norm, pack_indices_wide, std_about_mean, unpack_indices_wide, ItemId, MemoryBreakdown,
    Rotor, VectorBackend,
};

const K_BITS: u8 = 2; // branch bits per step
const N_BRANCH: usize = 1 << K_BITS; // cosets a state can reach
const N_COSETS: usize = 1 << (K_BITS + 1); // cosets in the partition

/// Three-bit coset label of an E8 point in raw doubled coordinates.
#[inline]
fn coset_of(r: &[i16; 8]) -> usize {
    let a = (((r[0] - r[1]) / 2) & 1) as usize;
    let b = (((r[2] - r[3]) / 2) & 1) as usize;
    let c = (((r[4] - r[5]) / 2) & 1) as usize;
    (a << 2) | (b << 1) | c
}

/// Set partitioning: the coset a `(state, branch)` pair names.
///
/// The subset a state may use must depend on the WHOLE state, not on its last
/// emission. With the bitshift transition `s' = ((s << k) | br) & mask`, the low
/// bit of the new state is the low bit of the branch just emitted, so a rule like
/// `state & 1` lets the encoder steer with one symbol of lookahead and memory beyond
/// that is never used -- measured bit-identical at M=1, 8 and 16. Parity over every
/// state bit is a function of the last `mem/k` branches, which is what makes the
/// subset choice a genuinely delayed decision the Viterbi has to plan across.
#[inline]
fn coset_for(state: usize, branch: usize) -> usize {
    let parity = (state.count_ones() & 1) as usize;
    (parity << K_BITS) | branch
}

struct CVec {
    /// `ceil(mem/8)` bytes of start state, then per block `k` branch bits and `m`
    /// index bits, packed.
    codes: Vec<u8>,
    rescale: f32,
    rnorm: f32,
    cidx: u32,
}

pub struct CosetTrellis {
    dim: usize,
    mem: u8,
    m_bits: u8,
    rotation: Rotor,
    /// Per coset: its `2^m` lowest-norm points, sorted by (norm, lex), in raw
    /// doubled coordinates. The index within a coset is the position here.
    books: Vec<Vec<[i16; 8]>>,
    /// Lowest-norm point of each coset, the seed for its nearest-point search.
    shifts: [[i16; 8]; N_COSETS],
    /// Unit-variance scaling over the union of kept points, as E8 does.
    scale_mean: f32,
    scale_std: f32,
    centroids: Vec<Vec<f32>>,
    entries: Vec<(ItemId, CVec)>,
}

impl CosetTrellis {
    pub fn new(dim: usize, bits: u8) -> Self {
        assert!(
            dim.is_multiple_of(8),
            "coset trellis requires dim % 8 == 0 (got {dim})"
        );
        assert!(
            (1..=3).contains(&bits),
            "coset trellis supports bits in 1..=3 (2^(8b-3) points per coset); got {bits}"
        );
        let mem = std::env::var("ULTRAVEC_TRELLIS_MEM")
            .ok()
            .and_then(|s| s.parse::<u8>().ok())
            .unwrap_or(12)
            .clamp(1, 20);
        let m_bits = 8 * bits - (K_BITS + 1);
        let per_coset = 1usize << m_bits;
        let mut r2 = 2.0f32;
        let mut raw = enumerate_e8_raw(r2);
        loop {
            let mut counts = [0usize; N_COSETS];
            for p in &raw {
                counts[coset_of(p)] += 1;
            }
            if counts.iter().all(|&c| c >= per_coset) {
                break;
            }
            r2 += 2.0;
            raw = enumerate_e8_raw(r2);
        }
        raw.sort_by(|a, b| {
            let na: i64 = a.iter().map(|&x| (x as i64) * (x as i64)).sum();
            let nb: i64 = b.iter().map(|&x| (x as i64) * (x as i64)).sum();
            na.cmp(&nb).then_with(|| a.cmp(b))
        });
        let mut books: Vec<Vec<[i16; 8]>> = (0..N_COSETS)
            .map(|_| Vec::with_capacity(per_coset))
            .collect();
        for p in &raw {
            let c = coset_of(p);
            if books[c].len() < per_coset {
                books[c].push(*p);
            }
        }
        let mut shifts = [[0i16; 8]; N_COSETS];
        for (c, book) in books.iter().enumerate() {
            shifts[c] = book[0];
        }
        let (mut sum, mut sumsq, mut cnt) = (0.0f64, 0.0f64, 0.0f64);
        for book in &books {
            for p in book {
                for &r in p {
                    let x = r as f64 / 2.0;
                    sum += x;
                    sumsq += x * x;
                    cnt += 1.0;
                }
            }
        }
        let mean = sum / cnt;
        let std = ((sumsq / cnt) - mean * mean).max(1e-12).sqrt();
        Self {
            dim,
            mem,
            m_bits,
            rotation: Rotor::new_oblivious(dim, crate::rotation_seed()),
            books,
            shifts,
            scale_mean: mean as f32,
            scale_std: std as f32,
            centroids: Vec::new(),
            entries: Vec::new(),
        }
    }

    pub fn with_centroids(mut self, centroids: Vec<Vec<f32>>) -> Self {
        assert!(centroids.iter().all(|c| c.len() == self.dim));
        self.centroids = centroids;
        self
    }

    #[inline]
    fn raw_key(p: &[i16; 8]) -> (i64, [i16; 8]) {
        (p.iter().map(|&x| (x as i64) * (x as i64)).sum(), *p)
    }

    fn index_in_coset(&self, c: usize, p: &[i16; 8]) -> Option<usize> {
        let key = Self::raw_key(p);
        self.books[c]
            .binary_search_by(|q| Self::raw_key(q).cmp(&key))
            .ok()
    }

    /// Nearest KEPT point of coset `c` to lattice-space `y`, with its squared
    /// distance. Exact, the same way the E8 encoder's fallback is: a seed `c0` in
    /// the coset bounds the answer at `d0 = |y - c0|`; every E8 point within `d0`
    /// is enumerated; the nearest that is in this coset's book wins. The optimum
    /// lies within `d0` because `c0` is itself a candidate, so the ball contains it.
    /// E8 has determinant one, so the ball holds about `pi^4 d^8 / 24` points and a
    /// good seed keeps that in the hundreds. Verified against a full scan in a test.
    fn nearest_in_coset(&self, c: usize, y: &[f32; 8]) -> (usize, f32) {
        let dist2 = |p: &[i16; 8]| -> f32 {
            let mut d = 0.0f32;
            for j in 0..8 {
                let e = y[j] - p[j] as f32 / 2.0;
                d += e * e;
            }
            d
        };
        let t = self.shifts[c];
        let (mut best, mut best_d) = (0usize, dist2(&self.books[c][0]));
        let (mut lo, mut hi) = (0.0f32, 1.0f32);
        for _ in 0..12 {
            let a = 0.5 * (lo + hi);
            let mut z = [0.0f32; 8];
            for j in 0..8 {
                z[j] = a * y[j] - t[j] as f32 / 2.0;
            }
            let pt = e8_nearest(&z);
            let mut r = [0i16; 8];
            for j in 0..8 {
                r[j] = (pt[j] * 2.0).round() as i16 + t[j];
            }
            let hit = if coset_of(&r) == c {
                self.index_in_coset(c, &r)
            } else {
                None
            };
            match hit {
                Some(i) => {
                    let d = dist2(&r);
                    if d < best_d {
                        best_d = d;
                        best = i;
                    }
                    lo = a;
                }
                None => hi = a,
            }
        }
        let est = std::f64::consts::PI.powi(4) * (best_d as f64).powi(4) / 24.0;
        if !est.is_finite() || est > self.books[c].len() as f64 {
            return self.nearest_in_coset_exhaustive(c, y);
        }
        let mut centre = [0.0f32; 8];
        for j in 0..8 {
            centre[j] = y[j] * 2.0;
        }
        let budget = 4.0 * best_d as f64 * (1.0 + 1e-6) + 1e-6;
        for p in enumerate_e8_ball(&centre, budget) {
            if coset_of(&p) != c {
                continue;
            }
            if let Some(i) = self.index_in_coset(c, &p) {
                let d = dist2(&p);
                if d < best_d {
                    best_d = d;
                    best = i;
                }
            }
        }
        (best, best_d)
    }

    /// Full scan of one coset's book: the guard for an absurd seed, and the oracle
    /// the exactness test compares against.
    fn nearest_in_coset_exhaustive(&self, c: usize, y: &[f32; 8]) -> (usize, f32) {
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for (i, p) in self.books[c].iter().enumerate() {
            let mut d = 0.0f32;
            for j in 0..8 {
                let e = y[j] - p[j] as f32 / 2.0;
                d += e * e;
            }
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        (best, best_d)
    }

    fn prep(&self, o: &[f32]) -> (Vec<f32>, f32, u32) {
        let n = l2_norm(o).max(f32::EPSILON);
        let u: Vec<f32> = o.iter().map(|v| v / n).collect();
        if self.centroids.is_empty() {
            return (u, 1.0, 0);
        }
        let (mut best, mut best_d) = (0usize, f32::INFINITY);
        for (j, c) in self.centroids.iter().enumerate() {
            let d: f32 = u.iter().zip(c).map(|(a, b)| (a - b) * (a - b)).sum();
            if d < best_d {
                best_d = d;
                best = j;
            }
        }
        let c = &self.centroids[best];
        let r: Vec<f32> = u.iter().zip(c).map(|(a, b)| a - b).collect();
        let rn = l2_norm(&r).max(f32::EPSILON);
        (r.iter().map(|v| v / rn).collect(), rn, best as u32)
    }

    /// Free-start exact Viterbi over blocks. The state is the last `mem` branch
    /// bits (bitshift transition on the branch symbol), so the state chain is a
    /// suffix of the branch stream exactly as in the scalar trellis, and the
    /// backpointer stores the window `(prev << k) | branch`.
    fn encode(&self, o: &[f32]) -> CVec {
        let (r, rnorm, cidx) = self.prep(o);
        let u_r = self.rotation.apply(&r);
        let scale = std_about_mean(&u_r).max(f32::EPSILON);
        let n_blk = self.dim / 8;
        let n_states = 1usize << self.mem;
        let state_mask = (n_states - 1) as u32;
        let inf = f32::INFINITY;

        // Per block, per coset: nearest kept index and its cost. Independent of the
        // state, so computed once per block for all eight cosets.
        let mut blk_best: Vec<[(usize, f32); N_COSETS]> = Vec::with_capacity(n_blk);
        for b in 0..n_blk {
            let mut y = [0.0f32; 8];
            for j in 0..8 {
                y[j] = (u_r[b * 8 + j] / scale) * self.scale_std + self.scale_mean;
            }
            let mut row = [(0usize, 0.0f32); N_COSETS];
            for (c, slot) in row.iter_mut().enumerate() {
                *slot = self.nearest_in_coset(c, &y);
            }
            blk_best.push(row);
        }

        let mut cost = vec![0.0f32; n_states];
        let mut back = vec![0u32; n_blk * n_states];
        for b in 0..n_blk {
            let mut next = vec![inf; n_states];
            for s in 0..n_states {
                let cs = cost[s];
                if cs == inf {
                    continue;
                }
                for br in 0..N_BRANCH {
                    let c = coset_for(s, br);
                    let w = ((s as u32) << K_BITS) | br as u32;
                    let ns = (w & state_mask) as usize;
                    let total = cs + blk_best[b][c].1;
                    if total < next[ns] {
                        next[ns] = total;
                        back[b * n_states + ns] = w;
                    }
                }
            }
            cost = next;
        }
        let mut s = (0..n_states)
            .min_by(|&a, &b| cost[a].total_cmp(&cost[b]))
            .unwrap();
        let mut branches = vec![0u8; n_blk];
        for b in (0..n_blk).rev() {
            let w = back[b * n_states + s];
            branches[b] = (w & ((1u32 << K_BITS) - 1)) as u8;
            s = (w >> K_BITS) as usize;
        }
        let start = s as u32;

        // Walk forward to resolve each branch to its coset, emit, and accumulate
        // the rescale from the reconstruction actually stored.
        let mut syms: Vec<u32> = Vec::with_capacity(n_blk);
        let mut rescale = 0.0f32;
        let mut st = start as usize;
        for b in 0..n_blk {
            let br = branches[b] as usize;
            let c = coset_for(st, br);
            let (idx, _) = blk_best[b][c];
            syms.push(((br as u32) << self.m_bits) | idx as u32);
            let p = &self.books[c][idx];
            for j in 0..8 {
                let v = (p[j] as f32 / 2.0 - self.scale_mean) / self.scale_std;
                rescale += v * u_r[b * 8 + j];
            }
            st = ((((st as u32) << K_BITS) | br as u32) & state_mask) as usize;
        }
        let mut codes = pack_indices_wide(&syms, K_BITS + self.m_bits);
        let sb = (self.mem as usize).div_ceil(8);
        let mut rec = Vec::with_capacity(sb + codes.len());
        for i in 0..sb {
            rec.push(((start >> (8 * i)) & 0xFF) as u8);
        }
        rec.append(&mut codes);
        CVec {
            codes: rec,
            rescale: rescale.max(f32::EPSILON),
            rnorm,
            cidx,
        }
    }

    /// Decode a record: read the start state, walk the branch symbols forward and
    /// resolve each to its coset through the same set-partition rule the encoder
    /// used. O(D/8), no search.
    fn decode_record(&self, cv: &CVec, scratch: &mut [u32], xbar: &mut [f32]) {
        let n_blk = self.dim / 8;
        let sb = (self.mem as usize).div_ceil(8);
        let mut st = 0u32;
        for (i, &byte) in cv.codes[..sb].iter().enumerate() {
            st |= (byte as u32) << (8 * i);
        }
        let state_mask = (1u32 << self.mem) - 1;
        let idx_mask = (1u32 << self.m_bits) - 1;
        unpack_indices_wide(&cv.codes[sb..], K_BITS + self.m_bits, &mut scratch[..n_blk]);
        for b in 0..n_blk {
            let sym = scratch[b];
            let br = (sym >> self.m_bits) as usize;
            let c = coset_for(st as usize, br);
            let p = &self.books[c][(sym & idx_mask) as usize];
            for j in 0..8 {
                xbar[b * 8 + j] = (p[j] as f32 / 2.0 - self.scale_mean) / self.scale_std;
            }
            st = ((st << K_BITS) | br as u32) & state_mask;
        }
    }

    fn score(&self, q_r: &[f32], qm: f32, cv: &CVec, scratch: &mut [u32], xbar: &mut [f32]) -> f32 {
        self.decode_record(cv, scratch, xbar);
        let dot = crate::simd::dot(q_r, &xbar[..self.dim]);
        qm + cv.rnorm * (dot / cv.rescale)
    }
}

impl VectorBackend for CosetTrellis {
    fn dimensions(&self) -> usize {
        self.dim
    }
    fn len(&self) -> usize {
        self.entries.len()
    }
    fn add(&mut self, id: ItemId, embedding: &[f32]) {
        assert_eq!(embedding.len(), self.dim);
        let v = self.encode(embedding);
        self.entries.push((id, v));
    }
    fn add_batch(&mut self, embeddings: &[Vec<f32>]) {
        use rayon::prelude::*;
        let dim = self.dim;
        let this: &CosetTrellis = self;
        let mut entries: Vec<(ItemId, CVec)> = embeddings
            .par_iter()
            .enumerate()
            .map(|(i, e)| {
                assert_eq!(e.len(), dim);
                (i as ItemId, this.encode(e))
            })
            .collect();
        self.entries.append(&mut entries);
    }
    fn search(&self, query: &[f32], limit: usize) -> Vec<(ItemId, f32)> {
        assert_eq!(query.len(), self.dim);
        let qn = l2_norm(query).max(f32::EPSILON);
        let q: Vec<f32> = query.iter().map(|v| v / qn).collect();
        let q_r = self.rotation.apply(&q);
        let mut scratch = vec![0u32; self.dim / 8];
        let mut xbar = vec![0.0f32; self.dim];
        let mut scored: Vec<(ItemId, f32)> = self
            .entries
            .iter()
            .map(|(id, cv)| {
                let qm = if self.centroids.is_empty() {
                    0.0
                } else {
                    q.iter()
                        .zip(&self.centroids[cv.cidx as usize])
                        .map(|(a, b)| a * b)
                        .sum()
                };
                (*id, self.score(&q_r, qm, cv, &mut scratch, &mut xbar))
            })
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(limit);
        scored
    }
    fn mem_bytes(&self) -> usize {
        let center = crate::centering_bytes(self.centroids.len());
        self.entries
            .iter()
            .map(|(_, cv)| cv.codes.len() + 4 + center)
            .sum()
    }
    fn memory_breakdown(&self) -> MemoryBreakdown {
        MemoryBreakdown {
            code_bytes: self.mem_bytes(),
            model_bytes: self.rotation.allocated_bytes()
                + self.books.iter().map(|b| b.capacity() * 16).sum::<usize>()
                + self
                    .centroids
                    .iter()
                    .map(|v| v.capacity() * std::mem::size_of::<f32>())
                    .sum::<usize>(),
            ..Default::default()
        }
    }
    fn reconstruct_unit(&self, x: &[f32]) -> Option<Vec<f32>> {
        let cv = self.encode(x);
        let mut scratch = vec![0u32; self.dim / 8];
        let mut xbar = vec![0.0f32; self.dim];
        self.decode_record(&cv, &mut scratch, &mut xbar);
        let inv = self.rotation.apply_inverse(&xbar);
        let n = l2_norm(&inv).max(f32::EPSILON);
        Some(inv.iter().map(|v| v / n).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *state
    }

    #[test]
    fn the_partition_is_eight_balanced_cosets() {
        let raw = enumerate_e8_raw(20.0);
        let z: Vec<&[i16; 8]> = raw.iter().filter(|p| coset_of(p) == 0).collect();
        let mut state = 0x9E37_79B9u64;
        for _ in 0..2000 {
            let a = &raw[(lcg(&mut state) >> 33) as usize % raw.len()];
            let b = z[(lcg(&mut state) >> 33) as usize % z.len()];
            let mut s = [0i16; 8];
            for j in 0..8 {
                s[j] = a[j] + b[j];
            }
            assert_eq!(
                coset_of(&s),
                coset_of(a),
                "coset not closed under the sublattice"
            );
        }
        let mut counts = [0usize; N_COSETS];
        for p in &raw {
            counts[coset_of(p)] += 1;
        }
        let (lo, hi) = (*counts.iter().min().unwrap(), *counts.iter().max().unwrap());
        assert!(
            lo as f64 / hi as f64 > 0.85,
            "cosets unbalanced on the shell: {counts:?}"
        );
    }

    #[test]
    fn nearest_in_coset_matches_the_exhaustive_scan() {
        for bits in [1u8, 2] {
            let q = CosetTrellis::new(8, bits);
            let mut state = 0x1234_5678_9ABC_DEF0u64;
            let mut checked = 0usize;
            for _ in 0..500 {
                let mut y = [0.0f32; 8];
                for j in 0..8 {
                    let u = ((lcg(&mut state) >> 11) as f32 / (1u64 << 53) as f32).max(1e-9);
                    let v = (lcg(&mut state) >> 11) as f32 / (1u64 << 53) as f32;
                    let g = (-2.0f32 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos();
                    y[j] = g * q.scale_std + q.scale_mean;
                }
                for c in 0..N_COSETS {
                    let (fi, fd) = q.nearest_in_coset(c, &y);
                    let (si, sd) = q.nearest_in_coset_exhaustive(c, &y);
                    if fi != si {
                        assert!(
                            (fd - sd).abs() < 1e-5,
                            "{bits}-bit coset {c}: fast d2={fd} vs exhaustive d2={sd}"
                        );
                    }
                    checked += 1;
                }
            }
            assert!(checked >= 4000);
        }
    }

    #[test]
    fn a_record_decodes_to_the_path_the_encoder_chose() {
        // The decoder walks the state to resolve cosets; if that walk disagrees
        // with the encoder's, the stored index points into the wrong book and the
        // reconstruction is garbage. Round-trip and check correlation.
        let dim = 64;
        let q = CosetTrellis::new(dim, 2);
        let v: Vec<f32> = (0..dim)
            .map(|j| ((j * 37) % 101) as f32 / 101.0 - 0.5)
            .collect();
        let out = q.reconstruct_unit(&v).unwrap();
        let n = l2_norm(&out);
        assert!((n - 1.0).abs() < 1e-4, "reconstruction not unit: {n}");
        let vn = l2_norm(&v);
        let cos: f32 = v.iter().zip(&out).map(|(a, b)| a * b).sum::<f32>() / vn;
        assert!(
            cos > 0.5,
            "reconstruction does not correlate with the input: {cos}"
        );
    }
}
