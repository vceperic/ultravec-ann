#![allow(
    clippy::doc_lazy_continuation,
    clippy::excessive_precision,
    clippy::manual_checked_ops,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity
)]

//! `ultravec` — low-bit vector compression and approximate-nearest-neighbor
//! search with trellis-coded quantization.
//!
//! This module holds shared primitives used by the TurboQuant reference,
//! the trellis codec, alternative quantizers, and the ANN index harnesses.

pub mod baseline;
pub mod bench;
pub mod blockquant;
pub mod codebook;
pub mod coset_trellis;
pub mod datasets;
pub mod dehub;
pub mod dist;
pub mod e8;
pub mod eden;
pub mod graph;
pub mod hnsw;
pub mod ivf;
pub mod learned_rotation;
pub mod pq;
pub mod pvq;
pub mod rabitq;
pub mod residual;
pub mod simd;
pub mod trellis;
pub mod ultraquant;

/// Opaque identifier for an indexed item.
pub type ItemId = i64;

/// Resident-memory ledger for a vector backend.
///
/// `code_bytes` is the serialized per-vector representation. `model_bytes`
/// covers shared fitted or computed state, `index_bytes` covers graph/list
/// structure, and `cache_bytes` covers optional decoded accelerators. Paper
/// comparisons must use [`MemoryBreakdown::total_resident_bytes`], while the
/// individual columns make codes-only and cached deployments distinguishable.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MemoryBreakdown {
    pub code_bytes: usize,
    pub model_bytes: usize,
    pub index_bytes: usize,
    pub cache_bytes: usize,
}

impl MemoryBreakdown {
    pub fn total_resident_bytes(self) -> usize {
        self.code_bytes
            .saturating_add(self.model_bytes)
            .saturating_add(self.index_bytes)
            .saturating_add(self.cache_bytes)
    }
}

// ── Vector-search backend trait ───────────────────────────────────────────────

/// A pluggable vector-search backend, trimmed to what the benchmark needs.
/// Per-query state for scoring STORED vectors by index.
///
/// A graph search touches a few hundred scattered candidates per query rather than
/// scanning in order, so it cannot use `VectorBackend::search`. It needs to ask "score
/// entry *i*" and have the codec answer from its codes. Everything that depends only on
/// the query -- the rotation, the per-centroid dot table, any sketch -- is hoisted into
/// the prepared state, and the scratch buffers the per-candidate path needs are owned
/// here rather than by the caller, which is what lets `score_at` take `&mut self` while
/// the graph walk holds `&self` on the index.
pub trait PreparedQuery: Send {
    fn score_at(&mut self, index: usize) -> f32;
}

/// A codec that can score its stored vectors by index.
///
/// Deliberately NOT a method on [`VectorBackend`]: that trait has a dozen implementors,
/// one of which stores nothing at all, and `baseline.rs` is a frozen verbatim control
/// that must not grow methods. Only the codecs a compressed graph actually navigates
/// with need this.
pub trait CandidateScorer: Send + Sync {
    fn prepare<'a>(&'a self, query: &[f32]) -> Box<dyn PreparedQuery + 'a>;
    /// Number of stored vectors, so an index can refuse a scorer built over a
    /// different database than the graph.
    fn scored_len(&self) -> usize;
    /// Shared state the scorer keeps resident (rotation, code table, centroids).
    /// A codes-only graph drops its fp32 array but still pays this, and reporting a
    /// memory total without it would understate the configuration.
    fn scorer_model_bytes(&self) -> usize;
}

pub trait VectorBackend: Send + Sync {
    /// Dimensionality of stored vectors.
    fn dimensions(&self) -> usize;
    /// Number of stored vectors.
    fn len(&self) -> usize;
    /// Whether the index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Insert one vector under `id`.
    fn add(&mut self, id: ItemId, embedding: &[f32]);
    /// Bulk-insert `embeddings` under ids `0..n` (index = id). Default delegates
    /// to the sequential `add`; a backend with an expensive per-vector encode
    /// (the trellis Viterbi) overrides this to parallelize. Order-preserving, so
    /// the resulting index is identical to the sequential path.
    fn add_batch(&mut self, embeddings: &[Vec<f32>]) {
        for (i, v) in embeddings.iter().enumerate() {
            self.add(i as ItemId, v);
        }
    }
    /// Pre-size the entry store for `additional` incoming vectors.
    ///
    /// Indexed builds add one vector at a time to preserve global ids, so without
    /// this the entry array grows by doubling and carries ~1.44x capacity slack on
    /// average. That slack is real resident memory and it lands in the reported
    /// per-vector figure, so it is worth removing rather than explaining. Default is
    /// a no-op for backends whose accounting does not include their entry array.
    fn reserve(&mut self, _additional: usize) {}
    /// Top-`limit` ids by descending similarity to `query`.
    fn search(&self, query: &[f32], limit: usize) -> Vec<(ItemId, f32)>;
    /// Approximate resident bytes — for the A/B memory column.
    fn mem_bytes(&self) -> usize;
    /// Break memory into serialized codes, shared model/index state, and
    /// optional resident accelerators. Backends that do not override this
    /// method are conservatively treated as storing everything in `code_bytes`.
    fn memory_breakdown(&self) -> MemoryBreakdown {
        MemoryBreakdown {
            code_bytes: self.mem_bytes(),
            ..MemoryBreakdown::default()
        }
    }
    /// Whether scores are approximate (→ benefits from an exact rerank pass).
    fn is_approximate(&self) -> bool {
        false
    }
    /// Reconstruct (dequantize) the unit-normalized direction of `x`, for
    /// measuring reconstruction MSE in the rank-distortion separation study.
    /// `None` if the backend can't reconstruct (brute force / PQ).
    fn reconstruct_unit(&self, _x: &[f32]) -> Option<Vec<f32>> {
        None
    }
}

/// Mean squared reconstruction error of the unit directions over `sample`
/// (rank-distortion vs rate-distortion study). `None` if `b` can't reconstruct.
pub fn reconstruction_mse(b: &dyn VectorBackend, sample: &[Vec<f32>]) -> Option<f64> {
    use rayon::prelude::*;

    // Reconstruction can be expensive (notably exact Viterbi encoding). Compute
    // each vector independently in parallel, then reduce the order-preserving
    // collection serially so the reported floating-point result is unchanged.
    let per_vector: Vec<f64> = sample
        .par_iter()
        .map(|x| {
            let recon = b.reconstruct_unit(x)?;
            let nx = l2_norm(x).max(f32::EPSILON);
            Some(
                x.iter()
                    .zip(&recon)
                    .map(|(xi, ri)| {
                        let d = (xi / nx - ri) as f64;
                        d * d
                    })
                    .sum(),
            )
        })
        .collect::<Option<Vec<_>>>()?;
    let mut total = 0.0f64;
    for value in &per_vector {
        total += value;
    }
    let n = per_vector.len() as u64;
    (n > 0).then(|| total / n as f64)
}

// ── Math helpers ───────────────────────────────────────────────

/// L2 norm.
/// Minimum wall-clock for one throughput measurement.
///
/// A timed window has to be long enough that scheduler noise on a shared host
/// averages out. The pinned Faiss and ScaNN drivers already loop to this budget,
/// so the codec harnesses use the same one and the two are comparable; before
/// this, the codec side timed a fixed three passes, which at the fastest
/// operating points was a 30--60 ms window.
pub const MIN_TIMING_SECONDS: f64 = 0.5;

/// Repeat `pass` until it has run `min_passes` times *and* filled
/// [`MIN_TIMING_SECONDS`]. Returns the elapsed seconds and the passes performed;
/// throughput is `queries * passes / elapsed`.
pub fn timed_window(min_passes: usize, mut pass: impl FnMut()) -> (f64, usize) {
    let start = std::time::Instant::now();
    let mut passes = 0usize;
    loop {
        pass();
        passes += 1;
        let elapsed = start.elapsed().as_secs_f64();
        if passes >= min_passes.max(1) && elapsed >= MIN_TIMING_SECONDS {
            return (elapsed, passes);
        }
    }
}

pub fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Per-vector bytes a codec must store to support centering.
///
/// Centering decomposes a stored vector as `u = c + ‖u−c‖·r` and codes only the unit
/// residual `r`, so the norm is real per-vector state and has to be serialized. With
/// more than one centroid the assignment index is too. This function is what charges
/// both, and every codec's `mem_bytes` calls it. Before it existed a centered row was
/// quoted four bytes light on a 37-byte record, nearly eleven percent, which is the
/// error it exists to prevent.
///
/// Charged identically for every codec, so relative standing is unaffected; what moves
/// is the absolute byte column, which is the number a deployment budgets against.
pub fn centering_bytes(n_centroids: usize) -> usize {
    match n_centroids {
        // Uncentered: the norm is 1.0 and the index is absent.
        0 => 0,
        // One centroid: the index is a constant and costs nothing to store.
        1 => 4,
        n => 4 + ((usize::BITS - (n - 1).leading_zeros()) as usize).div_ceil(8),
    }
}

/// Exact cosine similarity (used for ground truth + the rerank baseline).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = l2_norm(a);
    let nb = l2_norm(b);
    if na < f32::EPSILON || nb < f32::EPSILON {
        return 0.0;
    }
    dot / (na * nb)
}

/// Population standard deviation about the mean.
pub fn std_about_mean(x: &[f32]) -> f32 {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    var.sqrt()
}

/// SplitMix64 — deterministic PRNG with no `rand` dependency.
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniform f64 in [0,1) from a SplitMix64 state.
pub fn next_f64(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

// ── Fast Walsh-Hadamard transform ───────────────────────────────

/// In-place normalized FWHT. `buf.len()` must be a power of two. Self-inverse.
///
/// The butterfly (`buf[j] = u+v; buf[j+h] = u−v`) is SIMD-vectorized 8-wide for
/// stride `h ≥ 8` on x86_64+AVX2; smaller strides and the closing normalization
/// stay scalar. Byte-identical to the pure-scalar transform — each output element
/// is a single f32 add/sub/mul, so the per-lane rounding matches exactly (no
/// summation is reordered). Hot on every encode and query (the rotation).
pub fn fwht(buf: &mut [f32]) {
    let n = buf.len();
    debug_assert!(n.is_power_of_two());
    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            fwht_butterfly(&mut buf[i..i + 2 * h], h);
            i += h * 2;
        }
        h *= 2;
    }
    let inv = 1.0 / (n as f32).sqrt();
    fwht_scale(buf, inv);
}

/// One butterfly pass over `pair` (length `2*h`): the low half is `u+v`, the high
/// half is `u−v`, paired index-for-index. `pair[..h]` and `pair[h..]` are disjoint.
#[inline]
fn fwht_butterfly(pair: &mut [f32], h: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if h >= 8 && is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 verified; h≥8 and pair.len()==2*h bound the accesses.
            unsafe { fwht_butterfly_avx2(pair, h) };
            return;
        }
    }
    let (lo, hi) = pair.split_at_mut(h);
    for (a, b) in lo.iter_mut().zip(hi.iter_mut()) {
        let u = *a;
        let v = *b;
        *a = u + v;
        *b = u - v;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fwht_butterfly_avx2(pair: &mut [f32], h: usize) {
    use std::arch::x86_64::*;
    let p = pair.as_mut_ptr();
    let mut k = 0usize;
    while k + 8 <= h {
        let u = _mm256_loadu_ps(p.add(k));
        let v = _mm256_loadu_ps(p.add(h + k));
        _mm256_storeu_ps(p.add(k), _mm256_add_ps(u, v));
        _mm256_storeu_ps(p.add(h + k), _mm256_sub_ps(u, v));
        k += 8;
    }
    // h is a power of two ≥ 8 ⇒ no tail; loop covers all of [0,h).
    debug_assert_eq!(k, h);
}

/// In-place scale `buf *= inv` (the FWHT's 1/√n normalization), SIMD 8-wide.
#[inline]
fn fwht_scale(buf: &mut [f32], inv: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 verified; the tail past the 8-multiple is handled below.
            unsafe {
                use std::arch::x86_64::*;
                let invv = _mm256_set1_ps(inv);
                let p = buf.as_mut_ptr();
                let n = buf.len();
                let mut i = 0usize;
                while i + 8 <= n {
                    let x = _mm256_loadu_ps(p.add(i));
                    _mm256_storeu_ps(p.add(i), _mm256_mul_ps(x, invv));
                    i += 8;
                }
                while i < n {
                    *p.add(i) *= inv;
                    i += 1;
                }
            }
            return;
        }
    }
    for x in buf.iter_mut() {
        *x *= inv;
    }
}

/// Global mean of the unit-normalized db vectors (RaBitQ-style centering).
///
/// Shared rather than owned by the benchmark harness: the reconstruction path needs
/// the identical mean to compare against an upstream build that centers, and two
/// copies of this would be two chances to center differently.
pub fn unit_mean(dim: usize, db: &[Vec<f32>]) -> Vec<f32> {
    let mut m = vec![0.0f32; dim];
    for v in db {
        let n = l2_norm(v).max(f32::EPSILON);
        for (mi, x) in m.iter_mut().zip(v) {
            *mi += x / n;
        }
    }
    let inv = 1.0 / db.len().max(1) as f32;
    m.iter_mut().for_each(|x| *x *= inv);
    m
}

/// Seed for the data-oblivious rotation, shared by every codec.
///
/// Every reported cell is one draw from the rotation distribution: the sign flip is
/// seeded, so a different seed is a different (equally valid) oblivious transform.
/// The default is 42 so that every retained result reproduces bit-exactly, and
/// `ULTRAVEC_ROTATION_SEED` overrides it. That override is what makes
/// rotation-to-rotation variability measurable instead of merely acknowledged --
/// margins of around one point cannot be interpreted without it.
///
/// Read here rather than per codec so that a sweep moves every backend together; a
/// seed that applied to only some of them would silently compare across transforms.
pub fn rotation_seed() -> u64 {
    std::env::var("ULTRAVEC_ROTATION_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(42)
}

/// Blockwise Hadamard rotation: greedy power-of-two blocks, seeded ±1 sign flip
/// then FWHT per block. Orthonormal and invertible.
///
/// Supports **R rounds** of (independent sign-flip + FWHT) — `R=1` is the
/// original single-round transform; `R>1` (env `ULTRAVEC_ROTATION_ROUNDS`)
/// composes R orthonormal rounds with fresh signs each, approximating the
/// FHT+Kac-walk rotator the official RaBitQ uses (more mixing → less residual
/// anisotropy → lower quantization error). Still orthonormal + invertible.
pub struct Rotation {
    blocks: Vec<(usize, usize)>,
    rounds: Vec<Vec<f32>>, // signs per round
}

impl Rotation {
    /// Rounds from the environment (`ULTRAVEC_ROTATION_ROUNDS`, default 3), the
    /// research-harness entry point. A deployment that must not depend on its
    /// environment calls [`Rotation::with_rounds`].
    pub fn new(dim: usize, seed: u64) -> Self {
        let n_rounds = std::env::var("ULTRAVEC_ROTATION_ROUNDS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(3);
        Self::with_rounds(dim, seed, n_rounds)
    }

    /// `n_rounds` rounds (at least one) of seeded sign flip then FWHT per block.
    pub fn with_rounds(dim: usize, seed: u64, n_rounds: usize) -> Self {
        let mut blocks = Vec::new();
        let mut remaining = dim;
        let mut offset = 0;
        while remaining > 0 {
            let mut block_size = 1;
            while block_size * 2 <= remaining {
                block_size *= 2;
            }
            blocks.push((offset, block_size));
            offset += block_size;
            remaining -= block_size;
        }
        // Three rounds, not one. More mixing leaves less residual anisotropy for
        // every codec that shares this preprocessing, and measurably so on SIFT at
        // two bits and M=12 (appendix-ablations §5b): E8 +2.20 and PVQ +1.00
        // Recall@10 points, the trellis +0.60. It is adopted because it is the
        // better transform, not because it favours us -- it narrows the trellis's
        // margin over the best comparator from +6.60 to +5.50 points. Beyond three
        // the gain reverses (four rounds is worse than three).
        let n_rounds = n_rounds.max(1);
        let mut state = seed;
        let rounds = (0..n_rounds)
            .map(|_| {
                (0..dim)
                    .map(|_| {
                        if splitmix64(&mut state) & 1 == 0 {
                            -1.0
                        } else {
                            1.0
                        }
                    })
                    .collect()
            })
            .collect();
        Self { blocks, rounds }
    }

    /// The ±1 sign vector of each round, in application order. A kernel that
    /// reimplements the rotation embeds exactly these.
    pub fn sign_rounds(&self) -> &[Vec<f32>] {
        &self.rounds
    }

    /// Forward rotation: R rounds of (sign-flip then FWHT per block).
    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        let mut out = x.to_vec();
        for signs in &self.rounds {
            for &(start, size) in &self.blocks {
                let block = &mut out[start..start + size];
                for (o, s) in block.iter_mut().zip(&signs[start..start + size]) {
                    *o *= s;
                }
                fwht(block);
            }
        }
        out
    }

    /// Inverse rotation: rounds in reverse, each (FWHT then undo sign flip).
    pub fn apply_inverse(&self, y: &[f32]) -> Vec<f32> {
        let mut out = y.to_vec();
        for signs in self.rounds.iter().rev() {
            for &(start, size) in &self.blocks {
                let block = &mut out[start..start + size];
                fwht(block);
                for (o, s) in block.iter_mut().zip(&signs[start..start + size]) {
                    *o *= s;
                }
            }
        }
        out
    }

    pub(crate) fn allocated_bytes(&self) -> usize {
        self.blocks.capacity() * std::mem::size_of::<(usize, usize)>()
            + self.rounds.capacity() * std::mem::size_of::<Vec<f32>>()
            + self
                .rounds
                .iter()
                .map(|round| round.capacity() * std::mem::size_of::<f32>())
                .sum::<usize>()
    }
}

/// A rotation that can be the oblivious blockwise-Hadamard
/// (`Rotation`, the reference) or a learned/structured dense matrix (PCA / ITQ,
/// data-dependent). Selected by `ULTRAVEC_ROTATION={hadamard|pca|itq}`; the
/// `kac4` arm is `hadamard` with `ULTRAVEC_ROTATION_ROUNDS=4`. Same two-method
/// interface (`apply`/`apply_inverse`) the quantizers already use, so swapping it
/// in is a field-type change; `baseline.rs` retains its defined `Rotation`.
pub enum Rotor {
    Hadamard(Rotation),
    /// Learned/structured orthonormal d×d matrix, row-major; `y = M·x`.
    Dense {
        mat: Vec<f32>,
        dim: usize,
    },
}

impl Rotor {
    fn kind() -> String {
        std::env::var("ULTRAVEC_ROTATION").unwrap_or_else(|_| "hadamard".into())
    }
    /// Oblivious construction (no data). `pca`/`itq` defer to a Hadamard
    /// placeholder here and are filled in by [`Rotor::fit`] once a sample exists.
    pub fn new_oblivious(dim: usize, seed: u64) -> Self {
        Rotor::Hadamard(Rotation::new(dim, seed))
    }
    /// Oblivious construction with the round count given, no environment read.
    pub fn new_oblivious_with_rounds(dim: usize, seed: u64, rounds: usize) -> Self {
        Rotor::Hadamard(Rotation::with_rounds(dim, seed, rounds))
    }
    /// The Hadamard rotation's sign rounds; `None` for a learned dense rotor.
    pub fn sign_rounds(&self) -> Option<&[Vec<f32>]> {
        match self {
            Rotor::Hadamard(r) => Some(r.sign_rounds()),
            Rotor::Dense { .. } => None,
        }
    }
    /// Data-dependent construction: builds the learned matrix when
    /// `ULTRAVEC_ROTATION` is `pca`/`itq` and `sample` is non-empty; otherwise
    /// returns the oblivious rotor unchanged.
    pub fn fit(dim: usize, seed: u64, sample: &[Vec<f32>]) -> Self {
        match Self::kind().as_str() {
            "pca" if !sample.is_empty() => Rotor::Dense {
                mat: crate::learned_rotation::pca_rotation(sample, dim),
                dim,
            },
            "itq" if !sample.is_empty() => {
                let iters = std::env::var("ULTRAVEC_ITQ_ITERS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(15);
                Rotor::Dense {
                    mat: crate::learned_rotation::itq_rotation(sample, dim, iters, seed),
                    dim,
                }
            }
            _ => Self::new_oblivious(dim, seed),
        }
    }
    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        match self {
            Rotor::Hadamard(r) => r.apply(x),
            Rotor::Dense { mat, dim } => {
                let mut y = vec![0.0f32; *dim];
                for (i, yi) in y.iter_mut().enumerate() {
                    let row = &mat[i * dim..(i + 1) * dim];
                    *yi = row.iter().zip(x).map(|(m, xj)| m * xj).sum();
                }
                y
            }
        }
    }
    pub fn apply_inverse(&self, y: &[f32]) -> Vec<f32> {
        match self {
            Rotor::Hadamard(r) => r.apply_inverse(y),
            Rotor::Dense { mat, dim } => {
                // Orthonormal ⇒ inverse is the transpose: x = Mᵀ·y.
                let mut x = vec![0.0f32; *dim];
                for (i, &yi) in y.iter().enumerate() {
                    let row = &mat[i * dim..(i + 1) * dim];
                    for (xj, m) in x.iter_mut().zip(row) {
                        *xj += m * yi;
                    }
                }
                x
            }
        }
    }

    pub(crate) fn allocated_bytes(&self) -> usize {
        match self {
            Rotor::Hadamard(rotation) => rotation.allocated_bytes(),
            Rotor::Dense { mat, .. } => mat.capacity() * std::mem::size_of::<f32>(),
        }
    }
}

// ── Bit-packing of codebook indices ─────────────────────────────

/// Pack `bits`-wide indices big-endian into bytes.
pub fn pack_indices(indices: &[u16], bits: u8) -> Vec<u8> {
    let bits = bits as u32;
    let mut out = Vec::with_capacity((indices.len() * bits as usize).div_ceil(8));
    let mut buf: u32 = 0;
    let mut in_buf: u32 = 0;
    for &idx in indices {
        buf = (buf << bits) | idx as u32;
        in_buf += bits;
        while in_buf >= 8 {
            in_buf -= 8;
            out.push(((buf >> in_buf) & 0xFF) as u8);
        }
    }
    if in_buf > 0 {
        out.push(((buf << (8 - in_buf)) & 0xFF) as u8);
    }
    out
}

/// Unpack `out.len()` `bits`-wide indices from `data`.
pub fn unpack_indices(data: &[u8], bits: u8, out: &mut [u16]) {
    let bits = bits as u32;
    let mask = (1u32 << bits) - 1;
    let mut buf: u32 = 0;
    let mut in_buf: u32 = 0;
    let mut byte = 0;
    for slot in out.iter_mut() {
        while in_buf < bits && byte < data.len() {
            buf = (buf << 8) | data[byte] as u32;
            in_buf += 8;
            byte += 1;
        }
        // A truncated or malformed buffer exhausts `data` before `out` is filled.
        // `in_buf -= bits` then underflows: a panic in debug, and in release a wrap
        // to ~4e9 followed by a nonsense shift. Stop instead, leaving the remaining
        // slots at their zero initialization.
        if in_buf < bits {
            return;
        }
        in_buf -= bits;
        *slot = ((buf >> in_buf) & mask) as u16;
    }
}

/// Pack `bits`-wide indices, `bits` up to 32, MSB-first.
///
/// Identical bit layout to [`pack_indices`], which it must stay byte-compatible
/// with: the E8 lattice codebook needs 24-bit block indices at three bits per
/// coordinate, which a `u16` index cannot hold and a `u32` accumulator would
/// overflow. The narrow path is left untouched because the frozen TurboQuant
/// control and BlockQuant both serialize through it, and a scientific control
/// must not move.
pub fn pack_indices_wide(indices: &[u32], bits: u8) -> Vec<u8> {
    let bits = bits as u32;
    let mut out = Vec::with_capacity((indices.len() * bits as usize).div_ceil(8));
    let mut buf: u64 = 0;
    let mut in_buf: u32 = 0;
    for &idx in indices {
        buf = (buf << bits) | idx as u64;
        in_buf += bits;
        while in_buf >= 8 {
            in_buf -= 8;
            out.push(((buf >> in_buf) & 0xFF) as u8);
        }
    }
    if in_buf > 0 {
        out.push(((buf << (8 - in_buf)) & 0xFF) as u8);
    }
    out
}

/// Unpack `out.len()` `bits`-wide indices from `data`, `bits` up to 32.
///
/// Stops short rather than underflowing on a truncated buffer, exactly as
/// [`unpack_indices`] does, leaving the remaining slots zeroed.
pub fn unpack_indices_wide(data: &[u8], bits: u8, out: &mut [u32]) {
    let bits = bits as u32;
    let mask = if bits >= 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    };
    let mut buf: u64 = 0;
    let mut in_buf: u32 = 0;
    let mut byte = 0;
    for slot in out.iter_mut() {
        while in_buf < bits && byte < data.len() {
            buf = (buf << 8) | data[byte] as u64;
            in_buf += 8;
            byte += 1;
        }
        if in_buf < bits {
            return;
        }
        in_buf -= bits;
        *slot = ((buf >> in_buf) as u32) & mask;
    }
}

#[cfg(test)]
mod wide_packing_tests {
    use super::*;

    /// The wide path must be byte-compatible with the narrow one at every width the
    /// narrow one supports. If it is not, moving E8 onto it would silently move the
    /// 1- and 2-bit E8 cells the manuscript already reports.
    #[test]
    fn wide_packing_matches_the_narrow_path_where_they_overlap() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for bits in 1u8..=16 {
            let max = if bits >= 16 {
                u16::MAX
            } else {
                (1u16 << bits) - 1
            };
            let narrow: Vec<u16> = (0..97)
                .map(|_| (splitmix64(&mut state) as u16) & max)
                .collect();
            let wide: Vec<u32> = narrow.iter().map(|&v| v as u32).collect();
            assert_eq!(
                pack_indices(&narrow, bits),
                pack_indices_wide(&wide, bits),
                "packed bytes differ at {bits} bits"
            );
            let packed = pack_indices_wide(&wide, bits);
            let mut back = vec![0u32; wide.len()];
            unpack_indices_wide(&packed, bits, &mut back);
            assert_eq!(back, wide, "wide round-trip failed at {bits} bits");
        }
    }

    /// 24 bits is the width E8 needs at three bits per coordinate, and the width the
    /// narrow path cannot represent.
    #[test]
    fn wide_packing_round_trips_at_the_e8_three_bit_width() {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        for bits in [17u8, 20, 24, 28, 32] {
            let mask = if bits >= 32 {
                u32::MAX
            } else {
                (1u32 << bits) - 1
            };
            let idx: Vec<u32> = (0..131)
                .map(|_| (splitmix64(&mut state) as u32) & mask)
                .collect();
            let packed = pack_indices_wide(&idx, bits);
            assert_eq!(packed.len(), (idx.len() * bits as usize).div_ceil(8));
            let mut back = vec![0u32; idx.len()];
            unpack_indices_wide(&packed, bits, &mut back);
            assert_eq!(back, idx, "round-trip failed at {bits} bits");
        }
    }

    /// A truncated buffer must stop, not underflow, exactly as the narrow path does.
    #[test]
    fn wide_unpacking_survives_a_truncated_buffer() {
        let idx: Vec<u32> = vec![0x00AB_CDEF, 0x0012_3456, 0x00FE_DCBA];
        let packed = pack_indices_wide(&idx, 24);
        let mut back = vec![0u32; idx.len()];
        unpack_indices_wide(&packed[..4], 24, &mut back);
        assert_eq!(back[0], idx[0], "the complete leading index must decode");
        assert_eq!(back[2], 0, "slots past the data must stay zeroed");
    }
}

/// Nearest centroid index for `value` in an ascending-sorted `codebook`.
pub fn nearest_index(value: f32, codebook: &[f32]) -> u16 {
    let pos = codebook.partition_point(|&c| c < value);
    if pos == 0 {
        0
    } else if pos >= codebook.len() {
        (codebook.len() - 1) as u16
    } else if (value - codebook[pos - 1]).abs() <= (codebook[pos] - value).abs() {
        (pos - 1) as u16
    } else {
        pos as u16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every retained bundle was produced at rotation seed 42, and the reported
    /// cells are one draw from that distribution. Changing the default silently
    /// invalidates all of them at once -- the numbers would still reproduce, just
    /// from a different transform -- so the default is pinned here rather than
    /// left to the env reader.
    #[test]
    fn rotation_seed_defaults_to_the_value_the_evidence_was_recorded_at() {
        // The `check` tier strips every ULTRAVEC_* var precisely so unit tests
        // cannot inherit a research knob; assert that rather than skip silently.
        assert!(
            std::env::var("ULTRAVEC_ROTATION_SEED").is_err(),
            "ULTRAVEC_ROTATION_SEED is set in this shell; run tests via the check \
             tier, which strips it, or unset it -- a research knob must not leak \
             into the correctness suite"
        );
        assert_eq!(rotation_seed(), 42);
    }

    /// `unit_mean` moved out of the benchmark harness so the reconstruction path
    /// could center on the identical vector. Two copies would be two chances to
    /// center differently, which is exactly the discrepancy the parity study
    /// existed to rule out, so the shared one is pinned to a hand-checked case.
    #[test]
    fn unit_mean_averages_the_normalized_directions() {
        // Deliberately unequal norms: the mean is over DIRECTIONS, so a long
        // vector must not outvote a short one.
        let db = vec![vec![3.0, 0.0], vec![0.0, 0.25]];
        let mean = unit_mean(2, &db);
        assert!((mean[0] - 0.5).abs() < 1e-6, "got {mean:?}");
        assert!((mean[1] - 0.5).abs() < 1e-6, "got {mean:?}");
    }

    #[test]
    fn fwht_is_self_inverse() {
        let mut buf = vec![0.3, -1.2, 0.7, 2.1, -0.5, 0.9, 1.4, -2.0];
        let orig = buf.clone();
        fwht(&mut buf);
        fwht(&mut buf);
        for (a, b) in orig.iter().zip(&buf) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn rotation_round_trips_and_preserves_norm() {
        let dim = 768;
        let rot = Rotation::new(dim, 7);
        let mut state = 11u64;
        let v: Vec<f32> = (0..dim)
            .map(|_| (next_f64(&mut state) * 2.0 - 1.0) as f32)
            .collect();
        let back = rot.apply_inverse(&rot.apply(&v));
        for (a, b) in v.iter().zip(&back) {
            assert!((a - b).abs() < 1e-4);
        }
        assert!((l2_norm(&rot.apply(&v)) - l2_norm(&v)).abs() < 1e-4);
    }

    #[test]
    fn pack_unpack_round_trips() {
        for bits in [4u8, 5, 6] {
            let n = 200usize;
            let max = (1u16 << bits) - 1;
            let mut state = 123 + bits as u64;
            let indices: Vec<u16> = (0..n)
                .map(|_| (splitmix64(&mut state) as u16) & max)
                .collect();
            let packed = pack_indices(&indices, bits);
            let mut out = vec![0u16; n];
            unpack_indices(&packed, bits, &mut out);
            assert_eq!(indices, out);
        }
    }

    #[test]
    fn unpack_indices_survives_a_truncated_buffer() {
        // Decoding must stop at the end of the data, leaving the rest of the slots
        // zeroed. Underflowing the bit counter instead is a debug panic, and in release
        // a wrap to ~4e9 followed by a nonsense shift; this pins the safe behaviour.
        let bits = 5u8;
        let n = 64usize;
        let indices: Vec<u16> = (0..n).map(|i| (i as u16) & 31).collect();
        let packed = pack_indices(&indices, bits);
        for keep in [0usize, 1, 3, packed.len() / 2, packed.len() - 1] {
            let mut out = vec![0u16; n];
            unpack_indices(&packed[..keep], bits, &mut out);
            // The prefix that the surviving bytes fully cover must still decode.
            let decodable = keep * 8 / bits as usize;
            assert_eq!(
                out[..decodable],
                indices[..decodable],
                "truncating to {keep} bytes must still decode the intact prefix"
            );
        }
    }
}
