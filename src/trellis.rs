//! Trellis-coded quantization for ANN/MIPS database vectors. The implementation
//! applies a randomized orthogonal transform, encodes the resulting coordinates
//! with a bit-shift trellis, and uses a rescaled directional estimator at query
//! time. See the accompanying manuscript for the evaluated comparisons and
//! scope of the contribution.
//!
//! The default reconstruction codebook is data-oblivious: a deterministic hash
//! maps each trellis window to `(0,1)`, then the inverse normal CDF produces a
//! Gaussian value. The complete table is normalized to zero mean and unit
//! variance. QTIP's byte-sum 1MAD construction remains available only as an
//! explicit ablation through `ULTRAVEC_TRELLIS_CODE=1mad`.
//!
//! Encode = Viterbi over a `2^M`-state bitshift trellis minimizing squared error
//! to the rotated coords (one-shot, index-build time). Scoring uses the rescaled
//! estimator introduced by RaBitQ, `⟨q_r, x̃⟩ / ⟨x̃, u_r⟩`; its unbiasedness
//! guarantee is specific to RaBitQ's randomized-codebook model, so the manuscript
//! measures UltraVec's residual bias empirically rather than inheriting that claim.

use crate::{
    dist::inv_norm_cdf, l2_norm, pack_indices, std_about_mean, unpack_indices, ItemId,
    MemoryBreakdown, Rotor, VectorBackend,
};

/// Trellis memory in bits → `2^MEM` states. More memory = more "effective
/// dimension" (joint structure) = more shape gain, at `2^MEM`× encode cost.
// State bits. The appendix ladder measures recall through M=14. At two bits the
// emission payload stays at 32 bytes; the complete cosine record is 37 bytes for
// M<=8 and 38 bytes for M>=10 because ceil(M/8) changes -- payload + start state +
// the 4-byte rescale factor, as `mem_bytes` accounts it. (MIPS records add a
// further 4-byte norm sidecar, which is where the 41/42 figures come from; the
// appendix ladder scores against exact fp32 cosine, so it does not pay that.)
// Twelve is the main operating point, balancing most of the measured gain with
// encoding cost.
const MEM: u8 = 12;

/// Memo for the oblivious reconstruction tables.
///
/// The table is a pure function of `(mem, b_step, v, code_kind)` -- a fixed hash
/// through the inverse normal CDF, then a normalization over the whole table -- so
/// two quantizers built with the same parameters hold bit-identical contents. An
/// indexed build constructs one backend per posting list, which without this memo
/// meant 1024 identical 64 KiB copies at the two-bit operating point: 67 MiB, and
/// 67 of the 290 resident bytes/vector the paper charged to the codec.
///
/// Keyed rather than global-per-process because the sweeps vary `mem` and the rate
/// within one run. Entries are never evicted; the key space is the handful of
/// configurations a campaign touches.
/// Per-thread encoder scratch.
///
/// `encode` is `&self` and runs under `par_iter`, so every buffer it needs was a fresh
/// allocation per vector. The `back` pointer array dominates: `n_steps · n_states · 4`
/// bytes was 2 MiB at `mem=12` and 32 MiB at `mem=16`, allocated ZEROED once per vector.
/// Encoding 100k vectors therefore memset ~200 GB (mem=12) or ~800 GB (mem=14) against
/// a total encode time of seconds — and every one of those bytes is overwritten before
/// it is read, because each column writes `back[ns]` for all `n_states`.
///
/// Reusing one buffer per thread removes the allocation and the zeroing. It changes no
/// arithmetic: `encode_digest` and the byte-identity test are the gate.
#[derive(Default)]
struct EncodeScratch {
    back: Vec<u8>,
    total: Vec<f32>,
    next: Vec<f32>,
    cost: Vec<f32>,
}

impl EncodeScratch {
    /// Grow to fit without clearing: `resize` zeroes only the newly added tail, and
    /// every live entry is overwritten before use, so steady state costs nothing.
    fn ensure(&mut self, n_states: usize, n_emit: usize, n_steps: usize) {
        if self.back.len() < n_steps * n_states {
            self.back.resize(n_steps * n_states, 0);
        }
        if self.total.len() < n_emit * n_states {
            self.total.resize(n_emit * n_states, 0.0);
        }
        if self.next.len() < n_states {
            self.next.resize(n_states, 0.0);
        }
        if self.cost.len() < n_states {
            self.cost.resize(n_states, 0.0);
        }
    }
}

/// Whether this CPU has AVX2, resolved once instead of twice per trellis column.
///
/// `is_x86_feature_detected!` caches its answer, but the macro still expands to an
/// atomic load and a branch, and the encoder consulted it twice for every one of the
/// `D` columns of every vector. Hoisting it costs nothing and removes 256 checks per
/// vector at `D=128`.
#[cfg(target_arch = "x86_64")]
fn has_avx2() -> bool {
    use std::sync::OnceLock;
    static AVX2: OnceLock<bool> = OnceLock::new();
    *AVX2.get_or_init(|| is_x86_feature_detected!("avx2"))
}

thread_local! {
    static ENCODE_SCRATCH: std::cell::RefCell<EncodeScratch> =
        std::cell::RefCell::new(EncodeScratch::default());
}

type CodeKey = (u8, u8, usize, bool);
static CODE_TABLES: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<CodeKey, std::sync::Arc<Vec<f32>>>>,
> = std::sync::OnceLock::new();

fn shared_code_table(key: CodeKey, build: impl FnOnce() -> Vec<f32>) -> std::sync::Arc<Vec<f32>> {
    let cache = CODE_TABLES.get_or_init(Default::default);
    let mut guard = cache.lock().expect("code-table cache poisoned");
    std::sync::Arc::clone(
        guard
            .entry(key)
            .or_insert_with(|| std::sync::Arc::new(build())),
    )
}

/// Computed-code value for an L-bit window: a high-quality integer hash
/// (`lowbias32` finalizer — universal constants, oblivious) → uniform(0,1) →
/// inverse normal CDF. Unlike the byte-sum "1MAD" (only ~1021 distinct values →
/// heavy window collisions at large L), this gives a **distinct exact-Gaussian**
/// value per window, so the trellis uses its full discriminative power. No
/// training, no calibration.
fn gaussian_code(w: u32) -> f32 {
    let mut h = w;
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846c_a68b);
    h ^= h >> 16;
    let u = (h as f64 + 0.5) / 4_294_967_296.0; // (0,1)
    inv_norm_cdf(u) as f32
}

/// QTIP's original byte-sum "1MAD" code: one multiply-add, then sum the 4 bytes of the product.
/// The sum-of-4-bytes lands in [0,1020] (only ~1021 distinct values), so at the window lengths we use
/// it collides heavily -- the ABLATION baseline showing the computed inverse-CDF code (distinct
/// exact-Gaussian per window) is the lever. `ULTRAVEC_TRELLIS_CODE=1mad`.
fn mad_code(w: u32) -> f32 {
    let p = w.wrapping_mul(0x9E37_79B1); // a multiply (the "MA" of 1MAD)
    let bytesum = (p & 0xff) + ((p >> 8) & 0xff) + ((p >> 16) & 0xff) + ((p >> 24) & 0xff);
    // center [0,1020] roughly to zero-mean before the table's global normalization.
    bytesum as f32 - 510.0
}

struct TVec {
    codes: Vec<u8>, // dim × bits emission bits, packed (the compact on-disk form)
    /// Query-time scoring cache: the reconstructed code vector `x̄[i] = code[w_i]`
    /// (`dim` f32). The trellis windows `w_i` are query-INDEPENDENT — they are
    /// fully fixed by `start` + the packed `codes` — so we decode them ONCE here at
    /// build time (folded into `finalize`, which already walks the trellis), and
    /// `score` then reduces to a dense `⟨q_r, x̄⟩` dot product with NO per-query
    /// unpack and no loop-carried `state = w & mask` chain. Identical arithmetic to
    /// codes-only decoding, while the dense dot autovectorizes or uses the AVX2-FMA
    /// path in `simd::dot`. NB: `mem_bytes()` still reports only `codes` + metadata
    /// (the true compressed footprint); `recon` is a RAM-only accelerator, exactly
    /// as PVQ keeps its decoded pulses alongside its theoretical byte budget.
    /// The three optional side tables. Each is empty in the plain codes-only
    /// configuration, where three inline `Vec` headers cost 72 bytes per vector to
    /// say "nothing here" -- more than the 22-byte record they accompany. Boxing
    /// them behind one niche-optimized pointer makes the empty case 8 bytes and
    /// leaves the populated cases one indirection dearer, which is invisible next
    /// to the work they gate.
    aux: Option<Box<TVecAux>>,
    rescale: f32, // ⟨x̃, u_r⟩  (rescaled-estimator denominator)
    start: u32,   // chosen start state (free-start; ⌈mem/8⌉ B on disk)
    // Residual control-variate (only populated when resid_m > 0):
    resid_norm: f32, // ‖u_r − x̄/rescale‖, the residual against the reconstruction
    // the scorer uses
    rnorm: f32, // ‖u − c‖ for the centered decomposition (1.0 ⇒ off)
    cidx: u32,  // assigned centroid index (0 ⇒ off / single global mean)
}

/// The side tables of a [`TVec`], allocated only when at least one is populated.
#[derive(Default)]
struct TVecAux {
    /// Query-time scoring cache: the reconstructed code vector `x̄[i] = code[w_i]`
    /// (`dim` f32), empty in codes-only mode.
    recon: Vec<f32>,
    /// Sign bits of the QJL sketch of the residual (only when `resid_m > 0`).
    resid_signs: Vec<u8>,
    /// FastScan companion: MSB-first sign bits of `u_r` (⌈dim/8⌉ B). EMPTY unless
    /// the shortlist is enabled at build. Same packer/bit-order as RaBitQ 1-bit, so a
    /// query sign sketch XOR-aligns. Counted in `mem_bytes` only when present.
    sign_code: Vec<u8>,
}

impl TVec {
    /// The side tables read as empty slices when the box was never allocated, so
    /// every consumer keeps the shape it had when these were inline fields.
    #[inline]
    fn recon(&self) -> &[f32] {
        self.aux.as_ref().map_or(&[], |aux| &aux.recon)
    }
    #[inline]
    fn resid_signs(&self) -> &[u8] {
        self.aux.as_ref().map_or(&[], |aux| &aux.resid_signs)
    }
    #[inline]
    fn sign_code(&self) -> &[u8] {
        self.aux.as_ref().map_or(&[], |aux| &aux.sign_code)
    }
    /// Heap held by the side tables, for the cache term of the memory breakdown.
    #[inline]
    fn aux_cache_bytes(&self) -> usize {
        self.aux
            .as_ref()
            .map_or(0, |aux| aux.recon.capacity() * std::mem::size_of::<f32>())
    }
}

pub struct TrellisQuantizer {
    dim: usize,
    bits: u8,   // B: emitted bits per coordinate (rate)
    mem: u8,    // M: state bits
    v: usize,   // V: coordinates jointly quantized per Viterbi step (1 = scalar TCQ)
    b_step: u8, // emission bits per step = bits * v
    state_mask: u32,
    rotation: Rotor,
    /// `code[w] = normalized gaussian_code(w)` for the `2^L` windows.
    /// Shared: the reconstruction table is data-oblivious and depends only on
    /// `(mem, b_step, v)` plus a fixed hash, so every posting list in an index
    /// computes bit-identical values. Held behind an `Arc` because copying it per
    /// list is pure waste that lands in the reported resident figure -- at
    /// `mem=12`, two bits, 1024 lists it was 64 KiB x 1024 = 67 MiB, or 67 of the
    /// 290 bytes/vector the paper attributed to the codec.
    code: std::sync::Arc<Vec<f32>>,
    /// `(code_value, window)` sorted by value — lets beam free-start seed from
    /// the windows whose code is nearest the first coordinate (only built when
    /// `beam > 0`).
    code_sorted: std::sync::Arc<Vec<(f32, u32)>>,
    /// QJL residual control-variate: dims of the sign sketch of the residual
    /// `u_r − scale·code` (0 ⇒ off, plain rescaled estimator). Set via
    /// `ULTRAVEC_TRELLIS_RESID`. Tests whether correcting the IP estimate with a
    /// cheap unbiased residual term improves ranking; the associated experiment
    /// evaluates recall rather than reconstruction MSE.
    resid_m: usize,
    qjl: Option<crate::residual::SrhtSketch>,
    /// Anisotropic Viterbi branch-metric weight λ (0 ⇒ plain squared error).
    branch_lambda: f32,
    /// Beam width. 0 ⇒ exact Viterbi (cost O(D·2^mem·2^bits) — infeasible past
    /// mem≈14). >0 ⇒ beam-search encode keeping the top-`beam` states, cost
    /// O(D·beam·2^bits) *independent of 2^mem*, so a much larger effective
    /// memory (mem 16–22) is affordable. Set via `ULTRAVEC_TRELLIS_BEAM`.
    beam: usize,
    /// Coarse centroids of the unit vectors for centering (empty ⇒ off; 1 = global
    /// mean; N = IVF-style per-cluster). Residual to the assigned centroid is
    /// quantized; the ⟨q,c⟩ term is kept exact.
    centroids: Vec<Vec<f32>>,
    /// Codes-only scoring mode (the RAM-compressed regime). When
    /// `true`, `score` re-derives each candidate's window codes from the packed
    /// `codes`+`start` via the per-query O(D) Viterbi-state walk and does NOT keep
    /// the fp32 `recon` cache — so the index is genuinely RAM-compressed (codes only,
    /// no fp32 side-table). When `false` (default, mode (a) the build-time recon
    /// cache), `score` is a bare dot against the pre-decoded fp32 `recon` (fast, but
    /// fp32 RAM). Same arithmetic in both modes; only WHEN the decode happens (per
    /// query vs once at build) and whether fp32 is resident differs. Set via
    /// `ULTRAVEC_TRELLIS_CODES_ONLY=1` or [`TrellisQuantizer::with_codes_only`].
    codes_only: bool,
    /// Ablation: force the Viterbi to START from state 0 (no free-start choice), so the first
    /// ~mem/bits coords are penalized by the forced start. `ULTRAVEC_TRELLIS_FIXEDSTART=1`. Default
    /// false for the evaluated free-start mode.
    fixed_start: bool,
    /// Tail-biting: constrain the Viterbi path to end in the state it starts from.
    /// The state is the last `mem` emitted bits, so a wrapped path's start is a
    /// suffix of its own codes -- derivable rather than stored, which removes the
    /// `ceil(mem/8)`-byte start field. Costs a second Viterbi pass.
    /// `ULTRAVEC_TRELLIS_TAILBITE=1`. Default false.
    tail_biting: bool,
    /// How many candidate wrap states the tail-biting encoder tries.
    /// `ULTRAVEC_TRELLIS_TAILBITE_K`, default 1.
    tailbite_candidates: u16,
    /// FastScan companion shortlist size C (0 ⇒ off). When >0, `finalize` emits a
    /// 1-bit sign code per vector and `search` runs a SIMD-popcount shortlist of the
    /// C nearest-by-sign candidates, then trellis-reranks only those C. Set via
    /// `ULTRAVEC_TRELLIS_SHORTLIST` or [`TrellisQuantizer::with_shortlist`].
    shortlist_c: usize,
    /// Use the asymmetric (real-query × 1-bit-data) shortlist score instead of the
    /// symmetric Hamming one (`ULTRAVEC_TRELLIS_SHORTLIST_ASYM=1`); stronger coverage,
    /// O(D)/entry (no popcount). Assumes dim % 8 == 0.
    shortlist_asym: bool,
    shortlist_rnorm: bool,
    /// Byte-budget study knob (`ULTRAVEC_TRELLIS_RESCALE_BITS`): quantize the stored
    /// rescale scalar to this many bits. Default 0 => exact f32, which is the operating
    /// point every reported byte column was measured at. Oblivious grid on
    /// rescale/sqrt(D).
    rescale_bits: u8,
    /// Byte-budget study knob (`ULTRAVEC_TRELLIS_START_BITS`): restrict the free-start
    /// set to states whose index fits in this many bits. Default `mem` => full free
    /// start, the reported operating point; 0 => fixed start 0. Honored by the exact
    /// Viterbi encoder; the beam path does not restrict its start set.
    start_bits: u8,
    entries: Vec<(ItemId, TVec)>,
}

/// Bounds of the rescale quantization grid, as multiples of sqrt(D). The stored
/// scalar is an inner product between a unit residual and a normalized
/// reconstruction, so it cannot exceed sqrt(D) by construction and 1.25 leaves
/// headroom without wasting levels. Constants, not settings: a grid a deployment
/// could tune to its corpus would no longer be data-oblivious.
const RESCALE_GRID_LO: f32 = 0.0;
const RESCALE_GRID_HI: f32 = 1.25;

impl TrellisQuantizer {
    /// Serialized bytes of the two per-vector scalars the record carries beside its
    /// codes: the rescaling factor and the free-start state.
    ///
    /// Both fields are narrowable, and until this followed the knobs the accounting
    /// silently charged the full width whatever they were set to -- so a measurement
    /// showing a narrower field costs no recall could not be turned into a byte claim,
    /// because the byte column would still have reported the wide one. A record size
    /// that does not follow the record is not an accounting.
    fn scalar_bytes(&self) -> (usize, usize) {
        let rescale_b = if self.rescale_bits > 0 {
            (self.rescale_bits as usize).div_ceil(8)
        } else {
            std::mem::size_of::<f32>()
        };
        // The start field stores whichever is smaller: the state index the encoder is
        // allowed to choose from, or the full state width.
        if self.tail_biting {
            // The wrapped start equals the code stream's own last `mem` bits, so a
            // deployment recomputes it instead of serializing it.
            return (rescale_b, 0);
        }
        let start_bits = (self.start_bits as usize).min(self.mem as usize);
        (rescale_b, start_bits.div_ceil(8))
    }

    pub fn new(dim: usize, bits: u8) -> Self {
        Self::new_with_overrides(dim, bits, None, None)
    }

    fn new_with_overrides(
        dim: usize,
        bits: u8,
        memory_bits: Option<u8>,
        anisotropic_weight: Option<f32>,
    ) -> Self {
        // Memory (state bits) is the "effective dimension" knob; sweepable via
        // env for research without a recompile. More memory = more joint
        // structure (shape gain) at 2^mem× encode cost.
        let mem = memory_bits.unwrap_or_else(|| {
            std::env::var("ULTRAVEC_TRELLIS_MEM")
                .ok()
                .and_then(|s| s.parse::<u8>().ok())
                .unwrap_or(MEM)
        });
        let beam = std::env::var("ULTRAVEC_TRELLIS_BEAM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let resid_m = std::env::var("ULTRAVEC_TRELLIS_RESID")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0)
            // `SrhtSketch` truncates to `dim` and resets its own `m`, so an unclamped
            // value here would leave `1/m` and `mem_bytes` disagreeing with the sketch
            // that actually ran.
            .min(dim);
        // Anisotropic Viterbi branch metric: weight coord i's squared error by
        // (1 + λ·x_i²), penalizing error in high-energy (IP-relevant) coordinates
        // more — the parallel-error / ranking-aligned loss. λ=0 ⇒ plain MSE
        // metric. This is the trellis's structural edge: it can optimize an
        // arbitrary additive metric that a scalar quantizer can't.
        let branch_lambda = anisotropic_weight.unwrap_or_else(|| {
            std::env::var("ULTRAVEC_TRELLIS_ANISO")
                .ok()
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(0.0)
        });
        let codes_only = std::env::var("ULTRAVEC_TRELLIS_CODES_ONLY").as_deref() == Ok("1");
        let fixed_start = std::env::var("ULTRAVEC_TRELLIS_FIXEDSTART").as_deref() == Ok("1");
        let tail_biting = std::env::var("ULTRAVEC_TRELLIS_TAILBITE").as_deref() == Ok("1");
        let tailbite_candidates = std::env::var("ULTRAVEC_TRELLIS_TAILBITE_K")
            .ok()
            .and_then(|v| v.parse::<u16>().ok())
            .unwrap_or(1);
        let shortlist_c = std::env::var("ULTRAVEC_TRELLIS_SHORTLIST")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let shortlist_asym = std::env::var("ULTRAVEC_TRELLIS_SHORTLIST_ASYM").as_deref() == Ok("1");
        // Whether stage 1 multiplies its sign-code similarity by the stored residual
        // norm. Only meaningful with centering: uncentered, every `rnorm` is 1.0.
        let shortlist_rnorm =
            std::env::var("ULTRAVEC_TRELLIS_SHORTLIST_RNORM").as_deref() != Ok("0");
        let use_mad = std::env::var("ULTRAVEC_TRELLIS_CODE").as_deref() == Ok("1mad");
        let rescale_bits = std::env::var("ULTRAVEC_TRELLIS_RESCALE_BITS")
            .ok()
            .and_then(|s| s.parse::<u8>().ok())
            .unwrap_or(0);
        let start_bits = std::env::var("ULTRAVEC_TRELLIS_START_BITS")
            .ok()
            .and_then(|s| s.parse::<u8>().ok())
            .unwrap_or(mem)
            .min(mem);
        let qjl = (resid_m > 0).then(|| crate::residual::SrhtSketch::new(dim, resid_m, 137));
        // V: coordinates jointly quantized per Viterbi step (QTIP vectorized
        // trellis). V=1 is the scalar-per-step code used by default;
        // V>1 emits a V-dim sub-vector per step from a 2^L×V computed code, raising
        // the per-step effective dimension at the SAME byte budget (total emitted
        // bits = dim*bits for any V, since b_step=bits*V over dim/V steps). V>1
        // cells tile space with rounder Voronoi regions than scalar product cells —
        // the lever that can lift the g-saturation ceiling. Beam: V=1 only.
        let v = std::env::var("ULTRAVEC_TRELLIS_V")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&x| x >= 1 && dim.is_multiple_of(x))
            .unwrap_or(1);
        let b_step = bits * v as u8;
        assert!(
            b_step <= 16,
            "bits*V = {b_step} > 16 (emission alphabet too large)"
        );
        let l = b_step + mem;
        let n_windows = 1usize << l;
        // Computed code: V iid-Gaussian values per window, flat as code[w*V+j]; the
        // flat index doubles as the hash key so each (window,j) gets a distinct
        // exact-Gaussian value (collision-free, oblivious). Normalized to N(0,1).
        let code = shared_code_table((mem, b_step, v, use_mad), || {
            let mut code: Vec<f32> = (0..n_windows * v)
                .map(|k| {
                    if use_mad {
                        mad_code(k as u32)
                    } else {
                        gaussian_code(k as u32)
                    }
                })
                .collect();
            let mean = code.iter().sum::<f32>() / code.len() as f32;
            let var = code.iter().map(|c| (c - mean) * (c - mean)).sum::<f32>() / code.len() as f32;
            let inv_std = 1.0 / var.sqrt().max(1e-9);
            for c in &mut code {
                *c = (*c - mean) * inv_std;
            }
            code
        });
        // Beam free-start (V=1 only): sorted (value, window).
        let code_sorted: Vec<(f32, u32)> = if beam > 0 && v == 1 {
            let mut cs: Vec<(f32, u32)> = code
                .iter()
                .enumerate()
                .map(|(w, &val)| (val, w as u32))
                .collect();
            cs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            cs
        } else {
            Vec::new()
        };
        Self {
            dim,
            bits,
            mem,
            v,
            b_step,
            state_mask: (1u32 << mem) - 1,
            rotation: Rotor::new_oblivious(dim, crate::rotation_seed()),
            code,
            code_sorted: std::sync::Arc::new(code_sorted),
            resid_m,
            qjl,
            branch_lambda,
            beam,
            centroids: Vec::new(),
            codes_only,
            fixed_start,
            tail_biting,
            tailbite_candidates,
            shortlist_c,
            shortlist_asym,
            shortlist_rnorm,
            rescale_bits,
            start_bits,
            entries: Vec::new(),
        }
    }

    /// Enable centering with a single global mean of the unit vectors.
    pub fn with_mean(self, mean: Vec<f32>) -> Self {
        self.with_centroids(vec![mean])
    }

    /// Select codes-only scoring (RAM-compressed): `score`
    /// re-decodes each candidate's window codes per query and the fp32 `recon`
    /// side-table is dropped. Default (`false`) is the build-time recon cache
    /// (mode (a), fast decode-free dot, fp32 RAM). See the `codes_only` field.
    /// Width of the stored rescaling scalar, in bits; 0 keeps the exact f32.
    ///
    /// Quantization is on a FIXED grid over `rescale/sqrt(D)`, so this fits no corpus
    /// statistic and leaves the codec calibration-free. Sixteen bits is measured
    /// byte-identical in Recall@10 and reconstruction MSE at every rate.
    pub fn with_rescale_bits(mut self, bits: u8) -> Self {
        self.rescale_bits = bits;
        self
    }

    /// Restrict the free-start set to states whose index fits in `bits` bits.
    ///
    /// Narrower than `mem` costs recall in the measurements retained here, so this
    /// exists for the byte-budget study rather than for deployment.
    /// Enable tail-biting: the path must end where it starts, which removes the
    /// stored start field at the cost of a second encode pass.
    pub fn with_tail_biting(mut self, on: bool) -> Self {
        self.tail_biting = on;
        self
    }

    pub fn with_start_bits(mut self, bits: u8) -> Self {
        self.start_bits = bits.min(self.mem);
        self
    }

    pub fn with_codes_only(mut self, on: bool) -> Self {
        self.codes_only = on;
        self
    }

    /// Enable the FastScan companion with shortlist size `c` (0 ⇒ off). Pair with
    /// `.with_codes_only(true)` so stage-2 is a genuine RAM-compressed decode.
    pub fn with_shortlist(mut self, c: usize) -> Self {
        self.shortlist_c = c;
        self
    }

    /// Select the stage-1 scorer: `true` keeps the query in full precision and scores
    /// `⟨q_r, sign(u_r)⟩`; `false` binarizes the query too and ranks by Hamming
    /// distance. Explicit here rather than left to the ambient
    /// `ULTRAVEC_TRELLIS_SHORTLIST_ASYM`, for the reason recorded on `with_shortlist`'s
    /// caller in `ivf.rs`: a process-global that silently reaches every arm turns
    /// separate curves into one experiment.
    pub fn with_shortlist_asym(mut self, on: bool) -> Self {
        self.shortlist_asym = on;
        self
    }

    /// Enable the QJL residual control variate with an `m`-dimensional sign sketch
    /// (0 ⇒ off, the plain rescaled estimator). This does NOT reduce the base rate, so
    /// a bit-matched comparison must drop `bits` by one and pass `m = dim`; see
    /// `scripts/residual_control_variate.py`.
    pub fn with_resid(mut self, m: usize) -> Self {
        let m = m.min(self.dim);
        self.resid_m = m;
        self.qjl = (m > 0).then(|| crate::residual::SrhtSketch::new(self.dim, m, 137));
        self
    }

    /// Whether stage 1 weights its sign-code similarity by the stored residual norm.
    /// Under centering `⟨q,x_i⟩ = ⟨q,c⟩ + ‖u_i-c‖·⟨q_r,r_i⟩`, so the weight is what
    /// makes stage 1 rank the same quantity stage 2 does -- but it also scales stage
    /// 1's estimation error per entry, which is why it is measured rather than assumed.
    pub fn with_shortlist_rnorm(mut self, on: bool) -> Self {
        self.shortlist_rnorm = on;
        self
    }

    /// Two-stage FastScan companion search: a 1-bit sign-code shortlist of the `c`
    /// nearest-by-sign candidates (SIMD popcount over all N — cheap), then the trellis
    /// rescaled-estimator rerank of only those `c` (O(c) O(D) Viterbi-state walks
    /// instead of O(N)). Requires the sign code (shortlist enabled at build).
    ///
    /// Centering is supported. Within one posting list `⟨q,x_i⟩ = ⟨q,c⟩ +
    /// ‖u_i−c‖·⟨q_r, r_i⟩` and the `⟨q,c⟩` term is identical for every candidate, so
    /// ranking on `⟨q_r, r_i⟩` -- exactly what the sign code estimates -- is the right
    /// objective; the signs do not misalign under centering.
    /// What centering does introduce is the per-vector factor `‖u_i−c‖`, which has no
    /// uncentered counterpart (there every stored direction is a unit vector) and which
    /// stage 1 must therefore apply itself or rank a near candidate with a short
    /// residual below a far one with a long residual.
    pub fn search_shortlist(&self, query: &[f32], limit: usize, c: usize) -> Vec<(ItemId, f32)> {
        let qn = l2_norm(query);
        if qn < f32::EPSILON || self.entries.is_empty() {
            return Vec::new();
        }
        debug_assert!(
            !self.entries[0].1.sign_code().is_empty(),
            "search_shortlist called but the sign code is empty — enable the shortlist at build"
        );
        let u: Vec<f32> = query.iter().map(|v| v / qn).collect();
        let qc: Vec<f32> = self
            .centroids
            .iter()
            .map(|cc| u.iter().zip(cc).map(|(a, b)| a * b).sum())
            .collect();
        let q_r = self.rotation.apply(&u);
        let q_sketch = match &self.qjl {
            Some(q) => q.project(&q_r),
            None => Vec::new(),
        };
        let c = c.max(limit).min(self.entries.len());
        // Stage 1: sign-code shortlist (the cheap O(N) prune). Both scorers produce a
        // SIMILARITY (higher is nearer) so the residual-norm weight below applies to
        // either. Hamming distance `h` becomes `dim - 2h`, which is the sign-vector
        // inner product it already measures, so the uncentered ordering is unchanged.
        let centered = self.shortlist_rnorm && !self.centroids.is_empty();
        let q_bits = (!self.shortlist_asym).then(|| {
            pack_indices(
                &q_r.iter().map(|&x| (x > 0.0) as u16).collect::<Vec<_>>(),
                1,
            )
        });
        let mut scored: Vec<(f32, u32)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, (_, tv))| {
                let sim = match &q_bits {
                    None => masked_sum(&q_r, tv.sign_code()),
                    Some(bits) => self.dim as f32 - 2.0 * hamming(bits, tv.sign_code()) as f32,
                };
                // Uncentered, `rnorm` is 1.0 for every entry, so this is a no-op and
                // the ordering is bit-for-bit the previous one.
                let sim = if centered { tv.rnorm * sim } else { sim };
                (sim, i as u32)
            })
            .collect();
        if c < scored.len() {
            scored.select_nth_unstable_by(c, |a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            scored.truncate(c);
        }
        let cand: Vec<usize> = scored.into_iter().map(|(_, i)| i as usize).collect();
        // Stage 2: trellis rerank of only the C candidates.
        let mut scratch = vec![0u16; self.dim];
        let mut xbar = vec![
            0.0f32;
            if self.codes_only && self.v > 1 {
                self.dim
            } else {
                0
            }
        ];
        let mut windows = vec![
            0u32;
            if self.codes_only && self.v == 1 {
                self.dim
            } else {
                0
            }
        ];
        let mut results: Vec<(ItemId, f32)> = cand
            .into_iter()
            .map(|i| {
                let (id, tv) = &self.entries[i];
                let qm = qc.get(tv.cidx as usize).copied().unwrap_or(0.0);
                (
                    *id,
                    self.score(
                        &q_r,
                        qm,
                        &q_sketch,
                        tv,
                        &mut scratch,
                        &mut xbar,
                        &mut windows,
                    ),
                )
            })
            .collect();
        results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }

    /// Enable IVF-style per-cluster centering: quantize the residual to the
    /// nearest centroid (by cosine), keep the `⟨q,c⟩` term exact.
    pub fn with_centroids(mut self, centroids: Vec<Vec<f32>>) -> Self {
        assert!(centroids.iter().all(|c| c.len() == self.dim));
        self.centroids = centroids;
        self
    }

    /// Replace the rotation with a learned one fit on `sample` when
    /// `ULTRAVEC_ROTATION` is pca/itq (no-op for the oblivious default).
    pub fn with_rotation_fit(mut self, sample: &[Vec<f32>]) -> Self {
        self.rotation = Rotor::fit(self.dim, crate::rotation_seed(), sample);
        self
    }

    /// Order-independent digest of the encoded output (packed codes + start
    /// state) over `vectors`, parallel like `add_batch`. Pure `&self`; used only
    /// by the encode benchmark / byte-identity test to prove a SIMD change is an
    /// arithmetic speedup, not an algorithm change. Per-vector FNV-1a folded into
    /// an XOR accumulator so the result is independent of rayon scheduling order.
    pub fn encode_digest(&self, vectors: &[Vec<f32>]) -> u64 {
        use rayon::prelude::*;
        vectors
            .par_iter()
            .map(|v| {
                let tv = self.encode(v);
                let mut h = 0xcbf2_9ce4_8422_2325u64;
                let fold = |h: &mut u64, b: u8| {
                    *h ^= b as u64;
                    *h = h.wrapping_mul(0x0000_0100_0000_01b3);
                };
                for b in tv.start.to_le_bytes() {
                    fold(&mut h, b);
                }
                for &b in &tv.codes {
                    fold(&mut h, b);
                }
                h
            })
            .reduce(|| 0u64, |a, b| a ^ b)
    }

    /// Nearest centroid index for unit vector `u` (max cosine = max dot).
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

    /// Shared finalize for both Viterbi paths: forward-reconstruct from `start`,
    /// compute the rescale dot, and (if `resid_m>0`) the residual norm + QJL sign
    /// sketch for the control-variate estimator.
    fn finalize(
        &self,
        emissions: Vec<u16>,
        start: u32,
        u_r: &[f32],
        rnorm: f32,
        cidx: u32,
    ) -> TVec {
        let v = self.v;
        let mut state = start;
        let mut rescale = 0.0f32;
        let mut resid = if self.resid_m > 0 {
            vec![0.0f32; self.dim]
        } else {
            Vec::new()
        };
        let mut resid_norm = 0.0f32;
        // Cache the reconstructed code vector x̄[i]=code[w_i] for query-time scoring.
        // This is the SAME walk that computes `rescale` — capturing each `c` here is
        // free, and it lets `score` skip the per-query unpack + state-chain entirely.
        // In codes-only mode (mode (b)) we DON'T keep this fp32 side-table — `score`
        // re-derives the codes per query — so the index is genuinely RAM-compressed;
        // `recon` stays empty and the walk below still computes `rescale`.
        let mut recon = if self.codes_only {
            Vec::new()
        } else {
            vec![0.0f32; self.dim]
        };
        for (step, &e) in emissions.iter().enumerate() {
            let w = self.window(state, e as u32);
            for j in 0..v {
                let c = self.code[w as usize * v + j];
                let i = step * v + j;
                if !self.codes_only {
                    recon[i] = c;
                }
                rescale += c * u_r[i];
                if self.resid_m > 0 {
                    // Stash the code value; the residual cannot be formed until the
                    // walk finishes, because the reconstruction the estimator actually
                    // uses is scaled by `rescale` (see below).
                    resid[i] = c;
                }
            }
            state = w & self.state_mask;
        }
        if self.resid_m > 0 {
            // The residual must be taken against the reconstruction the SCORER uses,
            // not against `scale·x̄`. Scoring returns `⟨q_r,x̄⟩/rescale`, which is
            // `⟨q_r, x̄/rescale⟩`, so the implied reconstruction is `x̄/rescale` and the
            // residual is `u_r − x̄/rescale`.
            //
            // Taking it against `scale·x̄` instead pairs the correction with the naive
            // absolute estimator rather than the rescaled one, and forces the scorer to
            // swap estimator families to use it. Measured on SIFT at two bits, that
            // swap replaced a +0.00135 bias with −0.02491 -- larger than the estimator's
            // own standard error -- and cost 21.9 points of Recall@10.
            let inv = if rescale > 1e-6 { 1.0 / rescale } else { 0.0 };
            for (r, &ur) in resid.iter_mut().zip(u_r.iter()) {
                *r = ur - *r * inv;
                resid_norm += *r * *r;
            }
        }
        let resid_signs = if let Some(q) = &self.qjl {
            let proj = q.project(&resid);
            pack_indices(
                &proj
                    .iter()
                    .map(|&val| (val >= 0.0) as u16)
                    .collect::<Vec<_>>(),
                1,
            )
        } else {
            Vec::new()
        };
        // FastScan companion by-product: the 1-bit sign code of u_r (free — we already
        // walked u_r above). Empty unless the shortlist is enabled, so it does not
        // perturb encode_digest / recon / codes-only paths when off.
        let sign_code = if self.shortlist_c > 0 {
            pack_indices(
                &u_r.iter().map(|&x| (x > 0.0) as u16).collect::<Vec<_>>(),
                1,
            )
        } else {
            Vec::new()
        };
        let aux = if recon.is_empty() && resid_signs.is_empty() && sign_code.is_empty() {
            None
        } else {
            Some(Box::new(TVecAux {
                recon,
                resid_signs,
                sign_code,
            }))
        };
        // Quantize the stored rescale onto a grid that is fixed by construction: the
        // bounds are constants times sqrt(D), so nothing here is fitted to a corpus and
        // the codec stays calibration-free. Making them tunable would quietly turn this
        // into a data-dependent parameter, which is the one thing the record may not do.
        let rescale = if self.rescale_bits > 0 && rescale > 1e-6 {
            let levels = ((1u32 << self.rescale_bits) - 1) as f32;
            let sq = (self.dim as f32).sqrt();
            let (lo, hi) = (RESCALE_GRID_LO * sq, RESCALE_GRID_HI * sq);
            let t = ((rescale - lo) / (hi - lo)).clamp(0.0, 1.0);
            lo + (t * levels).round() / levels * (hi - lo)
        } else {
            rescale
        };
        TVec {
            codes: pack_indices(&emissions, self.b_step),
            aux,
            // Same degenerate-vector guard as RaBitQ: a zero/near-zero source has
            // rescale=⟨x̃,u_r⟩≈0 but a nonzero code → dot/rescale explodes and wins
            // rank-1 for every query. Sentinel ∞ ⇒ score=dot/∞=0 (ranks low).
            rescale: if rescale > 1e-6 {
                rescale
            } else {
                f32::INFINITY
            },
            start,
            resid_norm: resid_norm.sqrt(),
            rnorm,
            cidx,
        }
    }

    #[inline]
    fn window(&self, state: u32, emission: u32) -> u32 {
        (state << self.b_step) | emission
    }

    /// Unit vector (or its UNIT residual to the assigned centroid) to quantize.
    /// Returns (vector_to_quantize, rnorm, cidx). rnorm=1.0/cidx=0 when off.
    /// We quantize the UNIT residual so dot/rescale ≈ ⟨q,r_unit⟩ and the score
    /// recomposes exactly.
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

    /// Viterbi encode: find the emission sequence minimizing Σ(u_r[i]−code[w_i])².
    /// Fixed start state 0. Returns (packed emissions, ⟨x̃,u_r⟩).
    fn encode(&self, o: &[f32]) -> TVec {
        // Beam free-start is a V=1-only optimization; V>1 uses exact Viterbi below.
        if self.beam > 0 && self.v == 1 {
            return self.encode_beam(o);
        }
        let (r, rnorm, cidx) = self.prep(o);
        let u_r = self.rotation.apply(&r);
        // Standardize for Viterbi matching: rotated unit coords have std ≈ 1/√d,
        // but the computed code is N(0,1). Match in standardized space (`u_rs`);
        // the rescale estimator below uses the ORIGINAL `u_r`, so the per-vector
        // scale cancels and the cosine estimate stays scale-invariant.
        let scale = std_about_mean(&u_r).max(f32::EPSILON);
        let u_rs: Vec<f32> = u_r.iter().map(|v| v / scale).collect();

        if self.tail_biting {
            // Pass 1 picks a good wrap state; pass 2 pins the path to start AND end
            // there, so s_0 is the code stream's own last `mem` bits and needs no
            // stored byte. Costs a second Viterbi pass.
            let mut finals: Vec<f32> = Vec::new();
            let (em, st) = self.viterbi(&u_rs, None, Some(&mut finals));
            // Rank candidate wrap states by the free pass's cost of reaching them; a
            // single candidate (the unconstrained path's own end state) is a poor
            // proxy, because it was chosen for a problem without the wrap constraint.
            let k = self.tailbite_candidates.max(1) as usize;
            let mut order: Vec<u32> = (0..finals.len() as u32).collect();
            order.sort_unstable_by(|&a, &b| finals[a as usize].total_cmp(&finals[b as usize]));
            let mut cands: Vec<u32> = order.into_iter().take(k).collect();
            let folded = self.fold_state(st, &em);
            if !cands.contains(&folded) {
                cands.push(folded);
            }
            let mut best: Option<(f32, Vec<u16>, u32)> = None;
            for c in cands {
                let (e2, s2) = self.viterbi(&u_rs, Some(c), None);
                if s2 != c {
                    continue; // unreachable wrap for this state
                }
                let cost = self.path_cost(&u_rs, s2, &e2);
                if best.as_ref().is_none_or(|(b, _, _)| cost < *b) {
                    best = Some((cost, e2, s2));
                }
            }
            if let Some((_, e2, s2)) = best {
                return self.finalize(e2, s2, &u_r, rnorm, cidx);
            }
        }
        let (emissions, start) = self.viterbi(&u_rs, None, None);
        self.finalize(emissions, start, &u_r, rnorm, cidx)
    }

    /// Squared-error cost of a concrete path, used to rank candidate wrap states.
    fn path_cost(&self, u_rs: &[f32], start: u32, emissions: &[u16]) -> f32 {
        let v = self.v;
        let mut s = start;
        let mut total = 0.0f32;
        for (step, &e) in emissions.iter().enumerate() {
            let w = self.window(s, e as u32);
            for j in 0..v {
                let xi = u_rs[step * v + j];
                let wi = 1.0 + self.branch_lambda * xi * xi;
                let d = xi - self.code[w as usize * v + j];
                total += wi * d * d;
            }
            s = w & self.state_mask;
        }
        total
    }

    /// State reached by folding `emissions` through the bitshift transition. Any
    /// vector longer than `mem`/`bits` steps shifts the start out entirely, so this
    /// is a pure function of the emitted codes -- which is what makes a tail-biting
    /// start derivable rather than stored.
    fn fold_state(&self, start: u32, emissions: &[u16]) -> u32 {
        let mut s = start;
        for &e in emissions {
            s = self.window(s, e as u32) & self.state_mask;
        }
        s
    }

    /// One exact-Viterbi pass. `forced` pins both the only legal start state and the
    /// state the backtrack begins from; `None` reproduces the free-start encoder.
    fn viterbi(
        &self,
        u_rs: &[f32],
        forced: Option<u32>,
        mut out_costs: Option<&mut Vec<f32>>,
    ) -> (Vec<u16>, u32) {
        let v = self.v;
        let n_steps = self.dim / v;
        let n_states = 1usize << self.mem;
        let n_emit = 1usize << self.b_step; // 2^(bits·V): the per-step emission alphabet
        let inf = f32::INFINITY;
        // Free start: every state is a valid start (cost 0). The chosen start
        // state is recovered by backtracking and stored (⌈mem/8⌉ bytes), so the
        // first ~mem/bits coords aren't penalized by a forced start-0. Strictly
        // ≥ fixed-start quality. The ABLATION (`fixed_start`) forces start state 0
        // (cost[0]=0, the rest +inf) -- the appendix's free-vs-fixed-start row.
        let mut cost = vec![if self.fixed_start { inf } else { 0.0f32 }; n_states];
        if self.fixed_start {
            cost[0] = 0.0;
        }
        if !self.fixed_start && (self.start_bits as usize) < self.mem as usize {
            let allowed = 1usize << self.start_bits;
            cost[allowed..n_states].fill(inf);
        }
        // Tail-biting pass two: exactly one legal start, so the backtrack can be
        // pinned to the same state and the path wraps.
        if let Some(f) = forced {
            cost.fill(inf);
            cost[f as usize] = 0.0;
        }
        // back[step*n_states + ns] = (prev_state << b_step) | emission. The V=1 path
        // takes this from per-thread scratch instead (see `EncodeScratch`); the V>1
        // research path keeps its own zeroed buffer, because it writes `back` only for
        // states it can reach and so cannot inherit a neighbour's leftovers.
        let mut back = if v == 1 && self.mem as usize >= self.b_step as usize {
            Vec::new()
        } else {
            vec![0u32; n_steps * n_states]
        };
        let mut backtracked: Option<(Vec<u16>, u32)> = None;

        // V=1 SIMD fast path. The window→code map is a single scalar per window
        // (code[w]), and the next-state stride is exactly `n_states`, so the ACS
        // becomes: total[w] = cost[w>>b_step] + wi·(xi−code[w])²  over the 2^l
        // contiguous windows, then next[ns] = min_k total[ns + k·n_states] for
        // k in 0..n_emit. The argmin window index *is* the backpointer
        // ((s<<b_step)|e == w). Byte-identical to the scalar triple loop below
        // (same float ops, same first-on-tie order: smallest contributing window
        // == smallest k == smallest source state wins, matching the strict `<`).
        // V>1 (research knob) keeps the exact scalar Viterbi.
        if v == 1 && self.mem as usize >= self.b_step as usize {
            // Per-thread scratch: `total[w]` for w in 0..2^l (= n_emit·n_states), a
            // double-buffered `next` cost column, and the `back` pointer array.
            ENCODE_SCRATCH.with(|cell| {
                let mut guard = cell.borrow_mut();
                let sc = &mut *guard;
                sc.ensure(n_states, n_emit, n_steps);
                // `cost` is the one buffer that carries state INTO the first column, so
                // it is the one that must be reinitialized per vector.
                sc.cost[..n_states].fill(if self.fixed_start { inf } else { 0.0f32 });
                if self.fixed_start {
                    sc.cost[0] = 0.0;
                }
                if !self.fixed_start && (self.start_bits as usize) < self.mem as usize {
                    let allowed = 1usize << self.start_bits;
                    sc.cost[allowed..n_states].fill(inf);
                }
                if let Some(f) = forced {
                    sc.cost[..n_states].fill(inf);
                    sc.cost[f as usize] = 0.0;
                }
                for step in 0..n_steps {
                    let xi = u_rs[step];
                    let wi = 1.0 + self.branch_lambda * xi * xi;
                    viterbi_v1_step(
                        xi,
                        wi,
                        &self.code,
                        &sc.cost[..n_states],
                        n_states,
                        n_emit,
                        &mut sc.total[..n_emit * n_states],
                        &mut sc.next[..n_states],
                        &mut sc.back[step * n_states..(step + 1) * n_states],
                    );
                    std::mem::swap(&mut sc.cost, &mut sc.next);
                }
                // Backtrack INSIDE the borrow, reading the scratch in place. Copying
                // it out would reinstate exactly the 2 MiB per vector this removes.
                if let Some(o) = out_costs.as_deref_mut() {
                    o.clear();
                    o.extend_from_slice(&sc.cost[..n_states]);
                }
                backtracked = Some(backtrack_packed_from(
                    forced,
                    &sc.cost[..n_states],
                    &sc.back[..n_steps * n_states],
                    n_states,
                    n_steps,
                    self.b_step,
                ));
            });
        } else {
            for step in 0..n_steps {
                let base = step * v;
                let mut next = vec![inf; n_states];
                for s in 0..n_states {
                    let cs = cost[s];
                    if cs == inf {
                        continue;
                    }
                    for e in 0..n_emit {
                        let w = self.window(s as u32, e as u32);
                        // V-dim sub-vector branch cost with the per-coord anisotropic
                        // weight (V=1 reduces to the old wi·d²).
                        let mut d2 = 0.0f32;
                        for j in 0..v {
                            let xi = u_rs[base + j];
                            let wi = 1.0 + self.branch_lambda * xi * xi;
                            let d = xi - self.code[w as usize * v + j];
                            d2 += wi * d * d;
                        }
                        let c = cs + d2;
                        let ns = (w & self.state_mask) as usize;
                        if c < next[ns] {
                            next[ns] = c;
                            back[step * n_states + ns] = ((s as u32) << self.b_step) | e as u32;
                        }
                    }
                }
                cost = next;
            }
        }

        if let Some(o) = out_costs {
            if o.is_empty() {
                o.extend_from_slice(&cost);
            }
        }
        backtracked
            .unwrap_or_else(|| backtrack_from(forced, &cost, &back, n_states, n_steps, self.b_step))
    }

    /// Beam-search encode (fixed start state 0): keep the top-`beam` live states
    /// per step instead of all `2^mem`, so a large effective memory is affordable.
    /// Approximate (may miss the exact-Viterbi optimum) but the beam recovers
    /// nearly all of it at modest width.
    fn encode_beam(&self, o: &[f32]) -> TVec {
        let (r, rnorm, cidx) = self.prep(o);
        let u_r = self.rotation.apply(&r);
        let scale = std_about_mean(&u_r).max(f32::EPSILON);
        let u_rs: Vec<f32> = u_r.iter().map(|v| v / scale).collect();

        let n_emit = 1u32 << self.bits;
        let emit_mask = n_emit - 1;
        let beam = self.beam.max(1);

        // Backtrack history: hist[idx] = (parent_idx, emission); root parent = MAX.
        let mut hist: Vec<(u32, u16)> = Vec::with_capacity(beam * self.dim);
        // Live hypotheses: (cost, state, hist_idx, start_state).
        let mut live: Vec<(f32, u32, u32, u32)> = Vec::with_capacity(beam);

        // Step 0 — FREE START: seed with the `beam` windows whose code value is
        // nearest u_rs[0] (any prefix allowed). Each window W0 implies a start
        // state W0>>bits and a first emission W0&emit_mask.
        let t0 = u_rs[0];
        let cs = &self.code_sorted;
        let p = cs.partition_point(|&(v, _)| v < t0);
        let (mut lo, mut hi) = (p as isize - 1, p as isize);
        while live.len() < beam && (lo >= 0 || (hi as usize) < cs.len()) {
            let take_hi = if lo < 0 {
                true
            } else if (hi as usize) >= cs.len() {
                false
            } else {
                (t0 - cs[lo as usize].0) > (cs[hi as usize].0 - t0)
            };
            let (val, w0) = if take_hi {
                let x = cs[hi as usize];
                hi += 1;
                x
            } else {
                let x = cs[lo as usize];
                lo -= 1;
                x
            };
            let d = t0 - val;
            let w0w = 1.0 + self.branch_lambda * t0 * t0;
            let e0 = (w0 & emit_mask) as u16;
            let start_state = w0 >> self.bits;
            let hidx = hist.len() as u32;
            hist.push((u32::MAX, e0));
            live.push((w0w * d * d, w0 & self.state_mask, hidx, start_state));
        }

        let mut cand: Vec<(f32, u32, u32, u16, u32)> = Vec::with_capacity(beam * n_emit as usize);
        for &xi in u_rs.iter().skip(1) {
            cand.clear();
            let wi = 1.0 + self.branch_lambda * xi * xi;
            for &(c, state, hidx, st) in &live {
                for e in 0..n_emit {
                    let w = (state << self.bits) | e;
                    let d = xi - self.code[w as usize];
                    cand.push((c + wi * d * d, w & self.state_mask, hidx, e as u16, st));
                }
            }
            if cand.len() > beam {
                cand.select_nth_unstable_by(beam, |a, b| a.0.partial_cmp(&b.0).unwrap());
                cand.truncate(beam);
            }
            live.clear();
            for &(c, ns, parent, e, st) in &cand {
                let hidx = hist.len() as u32;
                hist.push((parent, e));
                live.push((c, ns, hidx, st));
            }
        }

        // Backtrack from the best final hypothesis; its start_state is carried.
        let best = live
            .iter()
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .unwrap();
        let start = best.3;
        let mut emissions = vec![0u16; self.dim];
        let mut idx = best.2;
        for i in (0..self.dim).rev() {
            let (parent, e) = hist[idx as usize];
            emissions[i] = e;
            if parent == u32::MAX {
                break;
            }
            idx = parent;
        }

        self.finalize(emissions, start, &u_r, rnorm, cidx)
    }

    fn score(
        &self,
        q_r: &[f32],
        qm: f32,
        q_sketch: &[f32],
        tv: &TVec,
        scratch: &mut [u16],
        xbar: &mut [f32],
        windows: &mut [u32],
    ) -> f32 {
        // The inner product `Σ_i q_r[i]·code[w_i]` has two deployment modes:
        //   (a) recon cache (default): a dense dot against the pre-decoded fp32
        //       `tv.recon` (built once per DB vector in `finalize`). Codes-only
        //       decoding derives each `code[w_i]` through a loop-carried `state = w & mask`
        //       chain that serialized the loop; the window `w_i` is query-independent,
        //       so decoding at build time removes the chain + the per-query unpack.
        //   (b) codes-only (RAM-compressed): re-derive x̄ from the packed codes+start
        //       into the caller's `xbar` scratch via that same state-chain walk, then
        //       dot. No fp32 side-table resides; the cost is one O(D) walk per
        //       candidate. The products are identical; only decode timing differs.
        let dot = if !self.codes_only {
            crate::simd::dot(q_r, tv.recon())
        } else if self.v == 1 {
            self.dot_codes_only(q_r, tv, windows)
        } else {
            // A window carries `v` interleaved values, which the stride-1 gather
            // cannot address; the vector trellis keeps the chain walk until its own
            // scan exists.
            self.decode_into(tv, scratch, xbar);
            crate::simd::dot(q_r, &xbar[..self.dim])
        };
        if self.resid_m == 0 {
            // Plain RaBitQ-style rescaled estimator, recomposed with the exact
            // mean term when centering is on (qm=0, rnorm=1 ⇒ uncentered).
            return qm + tv.rnorm * (dot / tv.rescale);
        }
        // Residual control-variate: scale·⟨q_r,x̄⟩ + √(π/2)/m·‖r‖·⟨q_sketch, signs⟩
        // estimates ⟨q_r, u_r⟩ for the STORED direction, because encode defines
        // r = u_r − scale·x̄. Reuses `scratch` rather than allocating: `resid_m ≤ dim`
        // and the emission unpack that owns this buffer is finished once `dot` exists.
        let signs = &mut scratch[..self.resid_m];
        unpack_indices(tv.resid_signs(), 1, signs);
        let mut s = 0.0f32;
        for (sq, &b) in q_sketch.iter().zip(signs.iter()) {
            s += sq * (b as f32 * 2.0 - 1.0);
        }
        let corr = (std::f32::consts::PI / 2.0).sqrt() / self.resid_m as f32 * tv.resid_norm * s;
        // Same rescaled base as the plain branch, plus the correction -- the two are
        // now the same estimator family, which is what makes the correction additive
        // rather than a substitution. Under centering the stored direction is the unit
        // residual to the centroid, so the whole thing is recomposed exactly as above;
        // uncentered `qm` is 0 and `rnorm` is 1.
        qm + tv.rnorm * (dot / tv.rescale + corr)
    }

    /// Codes-only decode of the rotated reconstruction `x̄[i]=code[w_i]` into `xbar`
    /// (caller-owned, len ≥ dim) via the state-chain walk from `tv.start` over the
    /// unpacked emissions (into `scratch`, len ≥ dim/V). This is the pre-inverse-
    /// rotation vector `score` dots against `q_r` — identical to `tv.recon` in mode
    /// (a), but recomputed per query so no fp32 side-table is resident (mode (b)).
    fn decode_into(&self, tv: &TVec, scratch: &mut [u16], xbar: &mut [f32]) {
        let v = self.v;
        let n_steps = self.dim / v;
        unpack_indices(&tv.codes, self.b_step, &mut scratch[..n_steps]);
        let mut state = tv.start;
        for step in 0..n_steps {
            let w = self.window(state, scratch[step] as u32);
            for j in 0..v {
                xbar[step * v + j] = self.code[w as usize * v + j];
            }
            state = w & self.state_mask;
        }
    }

    /// Write the window sequence `w_0..w_{n-1}` straight out of the packed code
    /// stream, without walking the state chain.
    ///
    /// The chain in [`Self::decode_into`] looks like a dependency and is not one.
    /// `state` holds exactly the last `mem` emitted bits, so with
    /// `w_i = (state_i << b) | e_i` and `state_{i+1} = w_i & (2^mem - 1)`,
    /// `w_i` is the `(mem + b)`-bit field at bit offset `i·b` of the stream
    /// `[start ++ e_0 ++ e_1 ++ …]`. Each window is thus a pure function of the
    /// packed bits at a known offset, and the values it indexes can be gathered in
    /// any order.
    ///
    /// That matters because the reconstruction table is 64 KiB at the two-bit
    /// operating point and larger above it. Walking the chain issues one dependent
    /// L2 load at a time; emitting the windows first lets the scan gather eight
    /// independent lanes and overlap that latency.
    ///
    /// Bit order matches `pack_indices`/`unpack_indices`: most-significant first,
    /// so `acc` holds the pending stream in its low `nbits` bits.
    #[inline]
    fn windows_into(&self, tv: &TVec, out: &mut [u32]) {
        let width = self.mem as u32 + self.b_step as u32;
        let wmask = (1u64 << width) - 1;
        let mut acc = tv.start as u64;
        let mut nbits = self.mem as u32;
        let mut byte = 0usize;
        for slot in out.iter_mut() {
            while nbits < width && byte < tv.codes.len() {
                acc = (acc << 8) | tv.codes[byte] as u64;
                nbits += 8;
                byte += 1;
            }
            // A truncated stream leaves the remaining windows at whatever the chain
            // walk would have produced from zero emissions, matching
            // `unpack_indices`'s documented early return rather than inventing bits.
            if nbits < width {
                let shortfall = width - nbits;
                acc <<= shortfall;
                nbits = width;
            }
            *slot = ((acc >> (nbits - width)) & wmask) as u32;
            nbits -= self.b_step as u32;
            acc &= (1u64 << nbits) - 1;
        }
    }

    /// `⟨q_r, x̄⟩` for a codes-only entry, fused: windows out of the bit stream, then
    /// one vectorized gather-and-FMA against the reconstruction table.
    ///
    /// Replaces unpack → chain-walk-writing-`xbar` → separate dot with two passes and
    /// no `xbar` at all, so the `4·dim` bytes that path wrote and re-read per
    /// candidate per query are gone. Scalar-`v` only; `v > 1` interleaves `v` values
    /// per window and keeps the general path.
    /// Windows are always kept 32-bit: the gather instruction takes 32-bit lanes
    /// regardless, so `u32` skips the widening step a `u16` buffer would need, and
    /// the extra 2 bytes per coordinate stay well inside L1. It also means one code
    /// path covers every `mem + b`, including the appendix sweeps above 16 bits.
    #[inline]
    fn dot_codes_only(&self, q_r: &[f32], tv: &TVec, windows: &mut [u32]) -> f32 {
        let n = self.dim;
        self.windows_into(tv, &mut windows[..n]);
        crate::simd::lut_dot_u32(&q_r[..n], &windows[..n], &self.code)
    }

    /// Decode-only: walk a stored `TVec` to its reconstructed unit direction WITHOUT
    /// re-encoding. This is the deploy-time decode path — the Viterbi is an *encode*-only
    /// cost (QTIP's inference kernel runs no Viterbi). `reconstruct_unit` = `encode` then
    /// this; a quantized index encodes once and calls this per read.
    fn reconstruct_from(&self, tv: &TVec) -> Vec<f32> {
        let v = self.v;
        let n_steps = self.dim / v;
        let mut scratch = vec![0u16; n_steps];
        unpack_indices(&tv.codes, self.b_step, &mut scratch);
        let mut state = tv.start;
        let mut xbar = vec![0.0f32; self.dim];
        for step in 0..n_steps {
            let w = self.window(state, scratch[step] as u32);
            for j in 0..v {
                xbar[step * v + j] = self.code[w as usize * v + j];
            }
            state = w & self.state_mask;
        }
        let mut recon = self.rotation.apply_inverse(&xbar);
        let n = l2_norm(&recon).max(f32::EPSILON);
        recon.iter_mut().for_each(|x| *x /= n);
        if self.centroids.is_empty() {
            recon
        } else {
            let c = &self.centroids[tv.cidx as usize];
            let mut full: Vec<f32> = c
                .iter()
                .zip(&recon)
                .map(|(cc, rr)| cc + tv.rnorm * rr)
                .collect();
            let nf = l2_norm(&full).max(f32::EPSILON);
            full.iter_mut().for_each(|x| *x /= nf);
            full
        }
    }

    /// Decode-only throughput probe: reconstruct every stored vector from its `TVec`
    /// (no re-encode), `rounds` times. Returns the number of reconstructions performed.
    /// Measures the deploy-time decode cost, isolated from the (one-time) Viterbi encode.
    pub fn decode_bench(&self, rounds: usize) -> usize {
        let mut acc = 0.0f32;
        for _ in 0..rounds {
            for (_, tv) in &self.entries {
                acc += self.reconstruct_from(tv)[0];
            }
        }
        std::hint::black_box(acc);
        self.entries.len() * rounds
    }
}

// ── SIMD-accelerated V=1 Viterbi ACS step ─────────────────────────
//
// One column of the V=1 trellis. The window→code map is a single scalar per
// window, so the ACS factors into:
//   Phase A  total[w] = (wi·d)·d,  d = xi − code[w]   (the squared-distance
//            metric over all 2^l = n_emit·n_states windows — the 2^MEM-scaling
//            bottleneck; pure contiguous, SIMD-vectorized)
//   Phase B  next[ns] = min_{k<n_emit} ( cost[(ns+k·n_states)>>b_step]
//                                        + total[ns+k·n_states] )
//            with the argmin window stored as the backpointer.
//
// Byte-identical to the scalar triple loop: the metric is computed as `(wi*d)*d`
// (same left-assoc, no FMA) and the cost add is the same `cs + d2`; the min scans
// k = 0..n_emit ascending (= source state ascending), so the strict-`<` first-on-
// tie winner matches exactly. `total` and `next` are caller-owned scratch reused
// across steps. `n_emit = 2^b_step`, `b_step = log2(n_emit)`.
#[inline]
fn viterbi_v1_step(
    xi: f32,
    wi: f32,
    code: &[f32],
    cost: &[f32],
    n_states: usize,
    n_emit: usize,
    total: &mut [f32],
    next: &mut [f32],
    back: &mut [u8],
) {
    let n_win = n_emit * n_states;
    let b_step = n_emit.trailing_zeros();
    // ── Phase A: metric over all windows (the hot, vectorizable part) ──
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            // SAFETY: avx2 verified above; slices are >= n_win (asserted by len).
            unsafe { metric_avx2(xi, wi, &code[..n_win], &mut total[..n_win]) };
        } else {
            metric_scalar(xi, wi, &code[..n_win], &mut total[..n_win]);
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        metric_scalar(xi, wi, &code[..n_win], &mut total[..n_win]);
    }
    // ── Phase B: add cost + min-reduce over the n_emit predecessors of each ns ──
    // k=0 seeds (cost of the lowest source state for this ns), k=1..n_emit refine
    // with strict `<` so the smallest-k (smallest source state) wins ties.
    //
    // This is the add-compare-select recursion, and it is the encoder's scalar
    // remainder: Phase A above is vectorized, so a ~6-op scalar body here running
    // once per window dominated a ~0.5-op-per-element vector body. The min-reduce is
    // across *states*, which are independent within a column -- the serial part of
    // Viterbi is the column-to-column chain, not this -- so it vectorizes.
    #[cfg(target_arch = "x86_64")]
    {
        if n_states >= 8 && has_avx2() {
            // SAFETY: avx2 verified; every load/store below stays inside the
            // `n_win`-length slices, and gathered `cost` indices are `w >> b_step`
            // with `w < n_win`, so they are `< n_states`.
            unsafe { acs_avx2(cost, total, n_states, n_emit, b_step, next, back) };
            return;
        }
    }
    for ns in 0..n_states {
        let s0 = ns >> b_step; // == (ns + 0*n_states) >> b_step
        let mut best = cost[s0] + total[ns];
        // The emission index, not the window: `w = ns + k*n_states` is recoverable.
        let mut bestk = 0u8;
        for k in 1..n_emit {
            let w = ns + k * n_states;
            let s = w >> b_step;
            let c = cost[s] + total[w];
            if c < best {
                best = c;
                bestk = k as u8;
            }
        }
        next[ns] = best;
        back[ns] = bestk;
    }
}

/// Backtrack a V=1 encode, whose `back` holds one emission index per state.
///
/// The full window is recoverable rather than stored: `w = ns + k*n_states`, and the
/// walk already knows `ns`. Only `k` is new information, so the array is a `u8` per
/// state instead of a `u32` per state -- 8 MiB rather than 32 MiB per vector at
/// `mem=16`, which is the difference between streaming past L3 and staying near it.
/// Backtrack the packed scratch from `forced` when given, else the cheapest state.
fn backtrack_packed_from(
    forced: Option<u32>,
    cost: &[f32],
    back: &[u8],
    n_states: usize,
    n_steps: usize,
    b_step: u8,
) -> (Vec<u16>, u32) {
    let Some(f) = forced else {
        return backtrack_packed(cost, back, n_states, n_steps, b_step);
    };
    let mut s = f as usize;
    let emit_mask = (1u32 << b_step) - 1;
    let mut emissions = vec![0u16; n_steps];
    for step in (0..n_steps).rev() {
        let k = back[step * n_states + s] as usize;
        let w = (s + k * n_states) as u32;
        emissions[step] = (w & emit_mask) as u16;
        s = (w >> b_step) as usize;
    }
    (emissions, s as u32)
}

fn backtrack_packed(
    cost: &[f32],
    back: &[u8],
    n_states: usize,
    n_steps: usize,
    b_step: u8,
) -> (Vec<u16>, u32) {
    let mut s = (0..n_states)
        .min_by(|&a, &b| cost[a].total_cmp(&cost[b]))
        .unwrap();
    let emit_mask = (1u32 << b_step) - 1;
    let mut emissions = vec![0u16; n_steps];
    for step in (0..n_steps).rev() {
        let k = back[step * n_states + s] as usize;
        let w = (s + k * n_states) as u32;
        emissions[step] = (w & emit_mask) as u16;
        s = (w >> b_step) as usize;
    }
    (emissions, s as u32)
}

/// Backtrack from the best final state; the returned `u32` is the chosen start state.
///
/// A NaN cost (from a NaN/degenerate input row) makes `partial_cmp` return None;
/// `total_cmp` keeps it total so a bad row yields a (garbage-but-valid) code instead of
/// panicking the whole batch.
/// Backtrack from `forced` when given, else from the cheapest final state.
fn backtrack_from(
    forced: Option<u32>,
    cost: &[f32],
    back: &[u32],
    n_states: usize,
    n_steps: usize,
    b_step: u8,
) -> (Vec<u16>, u32) {
    match forced {
        Some(f) => backtrack_at(f as usize, back, n_states, n_steps, b_step),
        None => backtrack(cost, back, n_states, n_steps, b_step),
    }
}

/// Walk the backpointers from an explicit final state.
fn backtrack_at(
    mut s: usize,
    back: &[u32],
    n_states: usize,
    n_steps: usize,
    b_step: u8,
) -> (Vec<u16>, u32) {
    let emit_mask = (1u32 << b_step) - 1;
    let mut emissions = vec![0u16; n_steps];
    for step in (0..n_steps).rev() {
        let packed = back[step * n_states + s];
        emissions[step] = (packed & emit_mask) as u16;
        s = (packed >> b_step) as usize;
    }
    (emissions, s as u32)
}

fn backtrack(
    cost: &[f32],
    back: &[u32],
    n_states: usize,
    n_steps: usize,
    b_step: u8,
) -> (Vec<u16>, u32) {
    let mut s = (0..n_states)
        .min_by(|&a, &b| cost[a].total_cmp(&cost[b]))
        .unwrap();
    let emit_mask = (1u32 << b_step) - 1;
    let mut emissions = vec![0u16; n_steps];
    for step in (0..n_steps).rev() {
        let packed = back[step * n_states + s];
        let prev = (packed >> b_step) as usize;
        let e = (packed & emit_mask) as u16;
        emissions[step] = e;
        s = prev;
    }
    (emissions, s as u32)
}

/// AVX2 add-compare-select for one trellis column.
///
/// Eight destination states per iteration. For each candidate emission `k` the
/// window index is `w = ns + k·n_states`, so `total[w]` is a contiguous load across
/// the eight lanes while the predecessor cost needs `cost[w >> b_step]`, which the
/// lanes share in runs and which is gathered.
///
/// Byte-identical to the scalar loop by construction: the summand is the same
/// `cost[s] + total[w]` with the same operand order, the comparison is the same
/// strict `<`, and `k` still ascends, so the smallest-`k` candidate still wins a tie.
/// `simd_v1_encode_is_byte_identical_to_scalar` is the gate on that, and it asserts
/// identity rather than tolerance.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn acs_avx2(
    cost: &[f32],
    total: &[f32],
    n_states: usize,
    n_emit: usize,
    b_step: u32,
    next: &mut [f32],
    back: &mut [u8],
) {
    use std::arch::x86_64::*;
    let lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    // log2(n_states): the shift that turns a window into its emission index.
    let mem_sh = _mm_cvtsi32_si128(n_states.trailing_zeros() as i32);
    // `_mm256_srli_epi32` needs a literal shift; `b_step` is a runtime value, so use
    // the variable-count form, whose amount lives in the low 64 bits of an xmm.
    let shift = _mm_cvtsi32_si128(b_step as i32);
    let pc = cost.as_ptr();
    let pt = total.as_ptr();
    let mut ns = 0usize;
    while ns + 8 <= n_states {
        // k = 0 seeds the running minimum.
        let mut wv = _mm256_add_epi32(_mm256_set1_epi32(ns as i32), lanes);
        let mut best = _mm256_add_ps(
            _mm256_i32gather_ps(pc, _mm256_srl_epi32(wv, shift), 4),
            _mm256_loadu_ps(pt.add(ns)),
        );
        let mut bestw = wv;
        for k in 1..n_emit {
            let w0 = ns + k * n_states;
            wv = _mm256_add_epi32(_mm256_set1_epi32(w0 as i32), lanes);
            let c = _mm256_add_ps(
                _mm256_i32gather_ps(pc, _mm256_srl_epi32(wv, shift), 4),
                _mm256_loadu_ps(pt.add(w0)),
            );
            // Strict less-than keeps the earlier (smaller) k on equality, matching
            // the scalar tie-break.
            let take = _mm256_cmp_ps(c, best, _CMP_LT_OS);
            best = _mm256_blendv_ps(best, c, take);
            bestw = _mm256_castps_si256(_mm256_blendv_ps(
                _mm256_castsi256_ps(bestw),
                _mm256_castsi256_ps(wv),
                take,
            ));
        }
        _mm256_storeu_ps(next.as_mut_ptr().add(ns), best);
        // Store the emission index, one byte per state, rather than the 32-bit
        // window: `w = ns + k*n_states`, so `k = w >> log2(n_states)` and the walk
        // recovers `w` from the state it is already standing on. Two packs turn eight
        // u32 lanes into an eight-byte store.
        let k32 = _mm256_srl_epi32(bestw, mem_sh);
        let p16 = _mm_packus_epi32(
            _mm256_castsi256_si128(k32),
            _mm256_extracti128_si256(k32, 1),
        );
        let p8 = _mm_packus_epi16(p16, p16);
        _mm_storel_epi64(back.as_mut_ptr().add(ns) as *mut __m128i, p8);
        ns += 8;
    }
    // Tail: identical scalar body for a state count that is not a multiple of eight.
    for ns in ns..n_states {
        let mut b = *pc.add(ns >> b_step) + *pt.add(ns);
        let mut bk = 0u8;
        for k in 1..n_emit {
            let w = ns + k * n_states;
            let c = *pc.add(w >> b_step) + *pt.add(w);
            if c < b {
                b = c;
                bk = k as u8;
            }
        }
        next[ns] = b;
        back[ns] = bk;
    }
}

/// Scalar reference metric: `total[w] = (wi·d)·d`, `d = xi − code[w]`. The
/// association `(wi*d)*d` matches the original `wi * d * d` exactly (no FMA).
fn metric_scalar(xi: f32, wi: f32, code: &[f32], total: &mut [f32]) {
    for (t, &c) in total.iter_mut().zip(code.iter()) {
        let d = xi - c;
        *t = (wi * d) * d;
    }
}

/// AVX2 metric kernel — 8 windows/iter, mirroring `metric_scalar` bit-for-bit
/// (sub, mul, mul; no fused multiply-add). `code`/`total` have equal length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn metric_avx2(xi: f32, wi: f32, code: &[f32], total: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = code.len();
    let xiv = _mm256_set1_ps(xi);
    let wiv = _mm256_set1_ps(wi);
    let cp = code.as_ptr();
    let tp = total.as_mut_ptr();
    let mut i = 0usize;
    // The anisotropic weight is 1.0 whenever lambda is 0, which is the default and
    // every configuration this paper reports. Specializing that case drops one of the
    // three arithmetic ops in the innermost loop of the encoder; the weighted path is
    // unchanged and still exercised by the anisotropy ablation.
    if wi == 1.0 {
        while i + 8 <= n {
            let cv = _mm256_loadu_ps(cp.add(i));
            let d = _mm256_sub_ps(xiv, cv);
            _mm256_storeu_ps(tp.add(i), _mm256_mul_ps(d, d));
            i += 8;
        }
    } else {
        while i + 8 <= n {
            let cv = _mm256_loadu_ps(cp.add(i));
            let d = _mm256_sub_ps(xiv, cv); // xi − code
            let wd = _mm256_mul_ps(wiv, d); // wi·d
            let d2 = _mm256_mul_ps(wd, d); // (wi·d)·d
            _mm256_storeu_ps(tp.add(i), d2);
            i += 8;
        }
    }
    // Scalar tail (n is a power of two ≥ 4; only the n=4 case has a tail here).
    while i < n {
        let d = xi - *cp.add(i);
        *tp.add(i) = (wi * d) * d;
        i += 1;
    }
}

/// Prepared query for [`TrellisQuantizer`]: the same hoists `search` performs once per
/// query, held so a graph walk can score scattered candidates.
pub struct TrellisPrepared<'a> {
    codec: &'a TrellisQuantizer,
    qc: Vec<f32>,
    q_r: Vec<f32>,
    q_sketch: Vec<f32>,
    scratch: Vec<u16>,
    xbar: Vec<f32>,
    windows: Vec<u32>,
    degenerate: bool,
}

impl crate::PreparedQuery for TrellisPrepared<'_> {
    #[inline]
    fn score_at(&mut self, index: usize) -> f32 {
        if self.degenerate {
            return 0.0;
        }
        let (_, tv) = &self.codec.entries[index];
        let qm = self.qc.get(tv.cidx as usize).copied().unwrap_or(0.0);
        self.codec.score(
            &self.q_r,
            qm,
            &self.q_sketch,
            tv,
            &mut self.scratch,
            &mut self.xbar,
            &mut self.windows,
        )
    }
}

impl crate::CandidateScorer for TrellisQuantizer {
    fn prepare<'a>(&'a self, query: &[f32]) -> Box<dyn crate::PreparedQuery + 'a> {
        let qn = l2_norm(query);
        let degenerate = qn < f32::EPSILON;
        let u: Vec<f32> = if degenerate {
            vec![0.0; self.dim]
        } else {
            query.iter().map(|v| v / qn).collect()
        };
        let qc: Vec<f32> = self
            .centroids
            .iter()
            .map(|c| u.iter().zip(c).map(|(a, b)| a * b).sum())
            .collect();
        let q_r = self.rotation.apply(&u);
        let q_sketch = match &self.qjl {
            Some(q) => q.project(&q_r),
            None => Vec::new(),
        };
        // Buffer widths mirror `search` exactly; getting them wrong here would be a
        // silent out-of-bounds rather than a wrong number.
        let xbar_len = if self.codes_only && self.v > 1 {
            self.dim
        } else {
            0
        };
        let windows_len = if self.codes_only && self.v == 1 {
            self.dim
        } else {
            0
        };
        Box::new(TrellisPrepared {
            codec: self,
            qc,
            q_r,
            q_sketch,
            scratch: vec![0u16; self.dim],
            xbar: vec![0.0f32; xbar_len],
            windows: vec![0u32; windows_len],
            degenerate,
        })
    }

    fn scored_len(&self) -> usize {
        self.entries.len()
    }

    fn scorer_model_bytes(&self) -> usize {
        self.memory_breakdown().model_bytes
    }
}

impl VectorBackend for TrellisQuantizer {
    fn dimensions(&self) -> usize {
        self.dim
    }
    fn len(&self) -> usize {
        self.entries.len()
    }
    fn add(&mut self, id: ItemId, embedding: &[f32]) {
        assert_eq!(embedding.len(), self.dim);
        let tv = self.encode(embedding);
        self.entries.push((id, tv));
    }
    fn reserve(&mut self, additional: usize) {
        self.entries.reserve_exact(additional);
    }
    fn add_batch(&mut self, embeddings: &[Vec<f32>]) {
        // The Viterbi encode dominates build time and is pure (`&self`), so
        // parallelize across vectors. Order-preserving collect ⇒ byte-identical
        // index to the sequential `add` loop (validated against it).
        use rayon::prelude::*;
        let dim = self.dim;
        let this: &TrellisQuantizer = self;
        let mut tvs: Vec<(ItemId, TVec)> = embeddings
            .par_iter()
            .enumerate()
            .map(|(i, v)| {
                assert_eq!(v.len(), dim);
                (i as ItemId, this.encode(v))
            })
            .collect();
        self.entries.append(&mut tvs);
    }
    fn search(&self, query: &[f32], limit: usize) -> Vec<(ItemId, f32)> {
        if self.shortlist_c > 0 {
            return self.search_shortlist(query, limit, self.shortlist_c);
        }
        let qn = l2_norm(query);
        if qn < f32::EPSILON {
            return Vec::new();
        }
        let u: Vec<f32> = query.iter().map(|v| v / qn).collect();
        // Query stays UNcentered; exact ⟨q,c⟩ for each vector's centroid added per
        // entry. Precompute ⟨q,c_j⟩ for every centroid once.
        let qc: Vec<f32> = self
            .centroids
            .iter()
            .map(|c| u.iter().zip(c).map(|(a, b)| a * b).sum())
            .collect();
        let q_r = self.rotation.apply(&u);
        // QJL query sketch (once per query) for the residual control-variate.
        let q_sketch = match &self.qjl {
            Some(q) => q.project(&q_r),
            None => Vec::new(),
        };
        let mut scratch = vec![0u16; self.dim];
        // Codes-only (mode b) re-decodes x̄ per candidate into this fp32 scratch; a
        // no-op buffer in the recon-cache path (mode a dots `tv.recon` directly).
        let mut xbar = vec![
            0.0f32;
            if self.codes_only && self.v > 1 {
                self.dim
            } else {
                0
            }
        ];
        let mut windows = vec![
            0u32;
            if self.codes_only && self.v == 1 {
                self.dim
            } else {
                0
            }
        ];
        let mut results: Vec<(ItemId, f32)> = self
            .entries
            .iter()
            .map(|(id, tv)| {
                let qm = qc.get(tv.cidx as usize).copied().unwrap_or(0.0);
                (
                    *id,
                    self.score(
                        &q_r,
                        qm,
                        &q_sketch,
                        tv,
                        &mut scratch,
                        &mut xbar,
                        &mut windows,
                    ),
                )
            })
            .collect();
        results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }
    fn mem_bytes(&self) -> usize {
        // codes + rescale + start + (resid: norm(4) + m sign bits).
        let (rescale_b, start_b) = self.scalar_bytes();
        let center = crate::centering_bytes(self.centroids.len());
        let resid_b = if self.resid_m > 0 {
            4 + self.resid_m.div_ceil(8)
        } else {
            0
        };
        self.entries
            .iter()
            .map(|(_, tv)| {
                tv.codes.len() + rescale_b + start_b + resid_b + tv.sign_code().len() + center
            })
            .sum()
    }
    fn memory_breakdown(&self) -> MemoryBreakdown {
        // The reconstruction table is shared across every backend built with the same
        // parameters, so charging each one the full table would multiply a single
        // allocation by the number of posting lists -- exactly the error this
        // accounting had before. Attribute it across its holders instead, so the sum
        // an index reports converges on one copy. The memo itself holds one reference,
        // so the attribution is short by one part in the number of lists.
        let shared = std::sync::Arc::strong_count(&self.code).max(1);
        let model_bytes = self.code.capacity() * std::mem::size_of::<f32>() / shared
            + self.code_sorted.capacity() * std::mem::size_of::<(f32, u32)>()
            + self.rotation.allocated_bytes()
            + self
                .qjl
                .as_ref()
                .map_or(0, crate::residual::SrhtSketch::allocated_bytes)
            + self
                .centroids
                .iter()
                .map(|centroid| centroid.capacity() * std::mem::size_of::<f32>())
                .sum::<usize>();
        let cache_bytes = self
            .entries
            .iter()
            .map(|(_, tv)| tv.aux_cache_bytes())
            .sum();
        MemoryBreakdown {
            code_bytes: self.mem_bytes(),
            model_bytes,
            index_bytes: self.entries.capacity() * std::mem::size_of::<(ItemId, TVec)>(),
            cache_bytes,
        }
    }
    fn is_approximate(&self) -> bool {
        true
    }
    /// Reconstruct the unit direction `ō` of `x`. Walk the trellis from the stored
    /// `start` state to rebuild the code vector `x̄` (same loop as `score`), then
    /// inverse-rotate + normalize. `g = ⟨ō, o⟩` measures how close the trellis
    /// reconstruction is to the truth — the variance-theory mediator (raising g
    /// shrinks the estimator std at 0 extra bytes; that is what memory M buys).
    fn reconstruct_unit(&self, x: &[f32]) -> Option<Vec<f32>> {
        // Encode (the one-time index-build / quantize cost) then the cheap O(D) decode
        // walk. Callers that already hold the TVec (a quantized index read) should
        // call `reconstruct_from` directly to skip the re-encode.
        Some(self.reconstruct_from(&self.encode(x)))
    }
}

/// Hamming distance between two equal-length packed bit slices (hardware POPCNT,
/// 8 bytes at a time). Both are ⌈dim/8⌉ bytes.
fn hamming(a: &[u8], b: &[u8]) -> u32 {
    let n = a.len().min(b.len());
    let mut d = 0u32;
    let mut i = 0;
    while i + 8 <= n {
        let x = u64::from_le_bytes(a[i..i + 8].try_into().unwrap())
            ^ u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        d += x.count_ones();
        i += 8;
    }
    while i < n {
        d += (a[i] ^ b[i]).count_ones();
        i += 1;
    }
    d
}

/// Asymmetric shortlist score Σ_i s_i·q_r[i], where s_i ∈ {−1,+1} is the sign bit of
/// `u_r[i]` unpacked MSB-first from the packed 1-bit `sign_code` (matching
/// `pack_indices(...,1)`). Higher ⇒ more aligned. Assumes dim % 8 == 0.
fn masked_sum(q_r: &[f32], sign_code: &[u8]) -> f32 {
    let mut s = 0.0f32;
    for (i, &qi) in q_r.iter().enumerate() {
        let bit = (sign_code[i >> 3] >> (7 - (i & 7))) & 1;
        s += if bit == 1 { qi } else { -qi };
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{cosine, next_f64};

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
    fn entry_size_stays_within_the_reported_index_footprint() {
        // `index_bytes` charges `entries.capacity()` times this, so it is literally
        // the resident per-vector cost the systems section reports -- and it is paid
        // whether or not the side tables hold anything. Pinned because the regression
        // is invisible: adding one `Vec` field to `TVec` costs 24 bytes per vector
        // across the whole index while every recall number stays identical.
        assert_eq!(std::mem::size_of::<(ItemId, TVec)>(), 64);
    }

    #[test]
    fn computed_code_is_approximately_gaussian() {
        // The normalized code table should be ~zero-mean unit-var with a
        // Bell-shaped spread: sanity-check the default inverse-CDF table.
        let q = TrellisQuantizer::new(64, 2);
        let n = q.code.len() as f32;
        let mean = q.code.iter().sum::<f32>() / n;
        let var = q.code.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        assert!(mean.abs() < 1e-4, "code mean {mean}");
        assert!((var - 1.0).abs() < 1e-3, "code var {var}");
        let maxv = q.code.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
        // The bound scales with the table: the extreme of `n` standard-normal draws
        // grows like sqrt(2 ln n), so larger tables reach farther into the tail.
        let expected_extreme = (2.0 * n.ln()).sqrt();
        assert!(
            maxv > 0.5 * expected_extreme && maxv < 2.0 * expected_extreme,
            "code range {maxv} not bell-like for {n} entries (expected around {expected_extreme})"
        );
    }

    #[test]
    fn trellis_self_query_top() {
        let dim = 128;
        let mut q = TrellisQuantizer::new(dim, 4);
        let db = rand_unit(300, dim, 11);
        for (i, v) in db.iter().enumerate() {
            q.add(i as ItemId, v);
        }
        assert_eq!(
            q.search(&db[0], 1)[0].0,
            0,
            "self-query should retrieve self"
        );
    }

    #[test]
    fn trellis_reconstruction_correlates_with_truth() {
        // ADC self-score should be high (good reconstruction of a unit vector).
        let dim = 256;
        let q = TrellisQuantizer::new(dim, 4);
        let db = rand_unit(50, dim, 5);
        let mut scratch = vec![0u16; dim];
        let mut xbar = vec![0.0f32; dim];
        let mut windows = vec![0u32; dim];
        let mut tot = 0.0f32;
        for v in &db {
            let tv = q.encode(v);
            let qn = l2_norm(v);
            let u: Vec<f32> = v.iter().map(|x| x / qn).collect();
            let q_r = q.rotation.apply(&u);
            tot += q.score(&q_r, 0.0, &[], &tv, &mut scratch, &mut xbar, &mut windows);
            // ≈ self cosine ≈ 1
        }
        let avg = tot / db.len() as f32;
        assert!(avg > 0.9, "trellis self-score {avg} too low");
        // (cosine imported for parity with sibling tests)
        let _ = cosine(&db[0], &db[0]);
    }

    /// Independent scalar Viterbi forward pass (the ORIGINAL triple loop), used
    /// only to prove the SIMD V=1 fast path is byte-identical. Returns the packed
    /// codes + chosen start state — exactly what `encode` stores.
    fn encode_scalar_ref(q: &TrellisQuantizer, o: &[f32]) -> (Vec<u8>, u32) {
        let (r, _rnorm, _cidx) = q.prep(o);
        let u_r = q.rotation.apply(&r);
        let scale = std_about_mean(&u_r).max(f32::EPSILON);
        let u_rs: Vec<f32> = u_r.iter().map(|v| v / scale).collect();
        let v = q.v;
        let n_steps = q.dim / v;
        let n_states = 1usize << q.mem;
        let n_emit = 1usize << q.b_step;
        let inf = f32::INFINITY;
        let mut cost = vec![0.0f32; n_states];
        let mut back = vec![0u32; n_steps * n_states];
        for step in 0..n_steps {
            let base = step * v;
            let mut next = vec![inf; n_states];
            for s in 0..n_states {
                let cs = cost[s];
                if cs == inf {
                    continue;
                }
                for e in 0..n_emit {
                    let w = (s as u32) << q.b_step | e as u32;
                    let mut d2 = 0.0f32;
                    for j in 0..v {
                        let xi = u_rs[base + j];
                        let wi = 1.0 + q.branch_lambda * xi * xi;
                        let d = xi - q.code[w as usize * v + j];
                        d2 += wi * d * d;
                    }
                    let c = cs + d2;
                    let ns = (w & q.state_mask) as usize;
                    if c < next[ns] {
                        next[ns] = c;
                        back[step * n_states + ns] = ((s as u32) << q.b_step) | e as u32;
                    }
                }
            }
            cost = next;
        }
        // total_cmp (not partial_cmp().unwrap()): a NaN cost from a degenerate row must not panic.
        let mut s = (0..n_states)
            .min_by(|&a, &b| cost[a].total_cmp(&cost[b]))
            .unwrap();
        let emit_mask = (1u32 << q.b_step) - 1;
        let mut emissions = vec![0u16; n_steps];
        for step in (0..n_steps).rev() {
            let packed = back[step * n_states + s];
            emissions[step] = (packed & emit_mask) as u16;
            s = (packed >> q.b_step) as usize;
        }
        (pack_indices(&emissions, q.b_step), s as u32)
    }

    /// The SIMD V=1 fast path must reproduce the scalar Viterbi BYTE-for-BYTE
    /// (codes + start), across the memory/rate regimes the encode benchmark uses
    /// and the anisotropic-metric knob. This is the correctness guardrail for the
    /// "SIMD is an arithmetic speedup, not an algorithm change" claim.
    /// The ascending-`k`, strict-`<` tie-break, tested where it actually applies.
    ///
    /// `simd_v1_encode_is_byte_identical_to_scalar` covers the arithmetic but not
    /// this: exact `f32` ties essentially never arise from random data, and that test
    /// keeps passing with the comparison inverted to `<=`. So the contract the kernel
    /// comment leans on was unverified. Coarse integer-valued cost and metric arrays
    /// make ties abundant, and then the backpointer -- which is what a wrong tie-break
    /// corrupts -- has to agree exactly.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn acs_tie_break_matches_scalar_on_deliberate_ties() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        for &(n_states, n_emit) in &[(8usize, 2usize), (8, 4), (16, 4), (32, 16), (64, 8)] {
            let b_step = n_emit.trailing_zeros();
            let n_win = n_states * n_emit;
            // Small integer values ⇒ many exactly-equal candidate sums.
            let cost: Vec<f32> = (0..n_states).map(|s| (s % 3) as f32).collect();
            let total: Vec<f32> = (0..n_win).map(|w| (w % 5) as f32).collect();

            let mut next_scalar = vec![0.0f32; n_states];
            // The backpointer is the EMISSION INDEX, one byte per state; the window
            // `w = ns + k*n_states` is recovered by the walk that already knows `ns`.
            let mut back_scalar = vec![0u8; n_states];
            for ns in 0..n_states {
                let mut best = cost[ns >> b_step] + total[ns];
                let mut bestk = 0u8;
                for k in 1..n_emit {
                    let w = ns + k * n_states;
                    let c = cost[w >> b_step] + total[w];
                    if c < best {
                        best = c;
                        bestk = k as u8;
                    }
                }
                next_scalar[ns] = best;
                back_scalar[ns] = bestk;
            }

            let mut next_simd = vec![0.0f32; n_states];
            let mut back_simd = vec![0u8; n_states];
            // SAFETY: avx2 checked above; slices are exactly the documented lengths.
            unsafe {
                acs_avx2(
                    &cost,
                    &total,
                    n_states,
                    n_emit,
                    b_step,
                    &mut next_simd,
                    &mut back_simd,
                )
            };
            assert_eq!(
                back_simd, back_scalar,
                "backpointer tie-break diverges (n_states={n_states}, n_emit={n_emit})"
            );
            assert_eq!(
                next_simd, next_scalar,
                "minimum diverges (n_states={n_states}, n_emit={n_emit})"
            );
        }
    }

    #[test]
    fn simd_v1_encode_is_byte_identical_to_scalar() {
        // Use process-local overrides because Rust runs sibling tests concurrently.
        let build = |dim: usize, bits: u8, mem: u8, aniso: Option<f32>| {
            TrellisQuantizer::new_with_overrides(dim, bits, Some(mem), aniso)
        };
        let check = |q: &TrellisQuantizer, label: &str| {
            assert_eq!(q.v, 1, "fast path is V=1 only");
            let db = rand_unit(40, q.dim, 4242 + q.mem as u64);
            for v in &db {
                let tv = q.encode(v); // SIMD fast path
                let (ref_codes, ref_start) = encode_scalar_ref(q, v);
                assert_eq!(tv.codes, ref_codes, "codes differ ({label})");
                assert_eq!(tv.start, ref_start, "start differs ({label})");
            }
        };
        for &(dim, bits, mem) in &[
            (128usize, 2u8, 6u8),
            (128, 2, 8),
            (96, 2, 10),
            (256, 4, 6),
            (128, 3, 7),
        ] {
            let q = build(dim, bits, mem, None);
            assert_eq!(q.mem, mem);
            check(&q, &format!("dim={dim} bits={bits} mem={mem}"));
        }
        // Also exercise the anisotropic branch metric (λ≠0): still byte-identical.
        let q = build(128, 2, 8, Some(0.5));
        check(&q, "aniso λ=0.5");
    }

    #[test]
    fn recon_cache_matches_state_chain_walk() {
        // The optimization's correctness invariant: the per-vector `recon` cache that
        // `score` uses must hold exactly the window codes the codes-only
        // state-chain walk produces. (The cache is filled by that
        // same walk in `finalize`; this pins the equivalence so a future refactor of
        // both paths remain equivalent.) V=1 is the default evaluated configuration; no
        // env mutation, so it's parallel-safe with the rest of the suite.
        for &(dim, bits) in &[(128usize, 2u8), (96, 2), (256, 4)] {
            let q = TrellisQuantizer::new(dim, bits);
            assert_eq!(q.v, 1, "test assumes default V=1");
            let db = rand_unit(8, dim, 7);
            let n_steps = dim;
            let mut scratch = vec![0u16; n_steps];
            for x in &db {
                let tv = q.encode(x);
                // Reproduce the window codes via the serial state-chain walk from the
                // stored `start` and the unpacked emissions.
                unpack_indices(&tv.codes, q.b_step, &mut scratch);
                let mut state = tv.start;
                let mut walk = vec![0.0f32; dim];
                for step in 0..n_steps {
                    let w = q.window(state, scratch[step] as u32);
                    walk[step] = q.code[w as usize];
                    state = w & q.state_mask;
                }
                assert_eq!(
                    tv.recon(),
                    walk,
                    "cache != state-chain walk (dim={dim},bits={bits})"
                );
            }
        }
    }

    /// The fused scan's whole premise: emitting windows straight out of the bit
    /// stream gives exactly what the state-chain walk gives.
    ///
    /// This is the sharpest available guard, because it compares the windows
    /// themselves rather than a downstream dot product where a discrepancy could
    /// hide under float tolerance. It covers windows narrower and wider than 16 bits
    /// (`mem + bits > 16` is reachable in the appendix sweeps) and a `bits` value
    /// that does not divide a byte, which is where a bit-order bug would surface.
    #[test]
    fn the_rescale_grid_is_fixed_not_configurable() {
        // The saving this codec claims rests on the grid being data-oblivious. If the
        // bounds ever become settable the claim quietly stops being true, so pin both
        // the constants and the absence of an environment read in the encode path.
        assert_eq!(RESCALE_GRID_LO, 0.0);
        assert_eq!(RESCALE_GRID_HI, 1.25);
        let source = include_str!("trellis.rs");
        let finalize = source
            .split("fn finalize(")
            .nth(1)
            .expect("finalize must exist");
        let body = &finalize[..finalize.find("\n    fn ").unwrap_or(finalize.len())];
        assert!(
            !body.contains("env::var"),
            "finalize runs once per vector; an environment read here is a per-vector \
             lock, the same defect Appendix B.3 records removing from the encoder"
        );
    }

    /// Which half of the codes-only scan actually costs: extracting the window
    /// stream, or gathering from the reconstruction table?
    ///
    /// Sections 8.1 and 10.6 attribute the scan's throughput to the gather. This
    /// separates the two halves, which have opposite fixes: extraction is a scalar walk
    /// with a loop-carried accumulator, bounded by dependency latency and fixed by
    /// vectorizing the bit-field reads; the gather is bounded by the table leaving
    /// cache, and fixed by computing the value instead of fetching it.
    ///
    /// Measured on an idle campaign machine, three consecutive runs: the gather is 33%
    /// of the fused scan at M=12 and 41% at M=16. Extraction is therefore the larger
    /// half at both memories and is flat in M; the gather is the term that grows when
    /// the 2^(M+b) table leaves cache.
    ///
    /// Run: `cargo test --release scan_cost_split -- --ignored --nocapture`
    ///
    /// RUN THIS ON A QUIET MACHINE. The split does not survive contention: the gather
    /// is memory-bound and a competing process evicts the very table whose residency is
    /// the question, so three back-to-back runs under load put its share at 44%, 56%
    /// and 67% against the 41% an idle box reports. The DIRECTION survives either way
    /// -- the M=12 arm is the control, its table 64 KiB and cache-resident where the
    /// M=16 table is 1 MiB and is not, and the gather's share rises between them in
    /// every run -- but only a quiet machine settles the magnitude.
    #[test]
    #[ignore]
    fn scan_cost_split() {
        use std::hint::black_box;
        use std::time::Instant;

        let dim = 128usize;
        let bits = 2u8;
        let vectors: Vec<Vec<f32>> = (0..2048)
            .map(|i| {
                (0..dim)
                    .map(|j| (((i * 37 + j * 101) % 1013) as f32 / 1013.0) - 0.5)
                    .collect()
            })
            .collect();
        let query: Vec<f32> = (0..dim)
            .map(|j| ((j * 17 % 97) as f32 / 97.0) - 0.5)
            .collect();

        println!("\nscan cost split, dim {dim}, {bits} bits, 2048 candidates per pass");
        println!(
            "{:>4} {:>10} {:>12} {:>12} {:>12} {:>11} {:>9}",
            "M", "table", "extract*", "gather ns", "fused ns", "extract ns", "gather%"
        );
        for mem in [12u8, 16u8] {
            // The reconstruction table is built inside `new` from this variable, so it
            // has to be set before construction. Run with `--test-threads=1`.
            std::env::set_var("ULTRAVEC_TRELLIS_MEM", mem.to_string());
            let mut q = TrellisQuantizer::new(dim, bits);
            assert_eq!(q.mem, mem, "the memory knob did not take");
            q.reserve(vectors.len());
            q.add_batch(&vectors);
            let entries: Vec<&TVec> = q.entries.iter().map(|(_, tv)| tv).collect();

            let mut windows = vec![0u32; dim];
            // Pre-extracted windows for the gather-only arm, so it measures the
            // gather and the FMA and nothing else.
            let mut pre: Vec<Vec<u32>> = Vec::with_capacity(entries.len());
            for tv in &entries {
                let mut w = vec![0u32; dim];
                q.windows_into(tv, &mut w);
                pre.push(w);
            }

            let reps = 40usize;
            for _ in 0..4 {
                for tv in &entries {
                    black_box(q.dot_codes_only(&query, tv, &mut windows));
                }
            }
            let t0 = Instant::now();
            for _ in 0..reps {
                for tv in &entries {
                    q.windows_into(tv, &mut windows);
                    black_box(windows[0]);
                }
            }
            let extract = t0.elapsed().as_secs_f64() / (reps * entries.len()) as f64 * 1e9;

            let t1 = Instant::now();
            for _ in 0..reps {
                for w in &pre {
                    black_box(crate::simd::lut_dot_u32(&query, w, &q.code));
                }
            }
            let gather = t1.elapsed().as_secs_f64() / (reps * pre.len()) as f64 * 1e9;

            let t2 = Instant::now();
            for _ in 0..reps {
                for tv in &entries {
                    black_box(q.dot_codes_only(&query, tv, &mut windows));
                }
            }
            let fused = t2.elapsed().as_secs_f64() / (reps * entries.len()) as f64 * 1e9;

            let table_kib = (q.code.len() * 4) / 1024;
            // Decompose by subtraction, not by the standalone extraction timing: called
            // on its own `windows_into` is a real call, while inside `dot_codes_only` it
            // is inlined and interleaves with the gather, so the standalone figure reads
            // higher than the fused path that contains it. `fused - gather` is what the
            // extraction actually costs where it runs.
            let in_situ = fused - gather;
            let share = gather / fused * 100.0;
            println!(
                "{mem:>4} {:>8} KiB {extract:>12.0} {gather:>12.0} {fused:>12.0}                  {in_situ:>11.0} {share:>8.0}%",
                table_kib
            );
        }
        println!();
    }

    #[test]
    fn a_tail_biting_path_actually_wraps() {
        // The claim that buys the two bytes: the encoder's chosen start state equals
        // the state its own emissions fold to. If this fails the start is NOT
        // derivable from the codes and the byte saving is fiction.
        let dim = 128;
        let base: Vec<Vec<f32>> = (0..32)
            .map(|i| {
                (0..dim)
                    .map(|j| ((i * 37 + j * 11) % 101) as f32 / 101.0 - 0.5)
                    .collect()
            })
            .collect();
        for bits in [1u8, 2, 3] {
            let mut q = TrellisQuantizer::new(dim, bits).with_tail_biting(true);
            q.reserve(base.len());
            q.add_batch(&base);
            for (_, tv) in &q.entries {
                let mut scratch = vec![0u16; dim / q.v];
                crate::unpack_indices(&tv.codes, q.b_step, &mut scratch);
                assert_eq!(
                    q.fold_state(tv.start, &scratch),
                    tv.start,
                    "{bits}-bit path does not wrap: start {} != folded state",
                    tv.start
                );
            }
        }
    }

    #[test]
    fn tail_biting_drops_the_start_field_from_the_record() {
        let dim = 128;
        let base: Vec<Vec<f32>> = (0..32)
            .map(|i| {
                (0..dim)
                    .map(|j| ((i * 53 + j * 7) % 97) as f32 / 97.0 - 0.5)
                    .collect()
            })
            .collect();
        let record = |q: TrellisQuantizer| {
            let mut q = q;
            q.reserve(base.len());
            q.add_batch(&base);
            q.mem_bytes() / base.len()
        };
        let free = record(TrellisQuantizer::new(dim, 2));
        let wrapped = record(TrellisQuantizer::new(dim, 2).with_tail_biting(true));
        let mem = TrellisQuantizer::new(dim, 2).mem as usize;
        assert_eq!(
            free - wrapped,
            mem.div_ceil(8),
            "tail-biting must save exactly the start field"
        );
    }

    #[test]
    fn record_bytes_follow_the_scalar_widths() {
        // The default must be unchanged: every byte column in the manuscript was
        // measured with an exact f32 rescale and a full-width start, so this test
        // failing means a reported record moved.
        let dim = 128;
        let base: Vec<Vec<f32>> = (0..64)
            .map(|i| {
                (0..dim)
                    .map(|j| ((i * 31 + j * 17) % 97) as f32 / 97.0 - 0.5)
                    .collect()
            })
            .collect();
        let record = |q: TrellisQuantizer| {
            let mut q = q;
            q.reserve(base.len());
            q.add_batch(&base);
            q.mem_bytes() / base.len()
        };
        let payload = dim * 2 / 8; // two bits per dimension
        let mem = TrellisQuantizer::new(dim, 2).mem as usize;
        let start_b = mem.div_ceil(8);
        let default = record(TrellisQuantizer::new(dim, 2));
        assert_eq!(
            default,
            payload + 4 + start_b,
            "default record is payload + an exact f32 + a full-width start"
        );

        // A 16-bit rescale is two bytes, which is the saving the appendix reports.
        let narrow = record(TrellisQuantizer::new(dim, 2).with_rescale_bits(16));
        assert_eq!(
            narrow,
            payload + 2 + start_b,
            "a 16-bit rescale must be charged two bytes"
        );
        assert_eq!(
            default - narrow,
            2,
            "the saving is exactly the two bytes measured"
        );

        // And the start field follows its own knob rather than the state width.
        let short_start = record(TrellisQuantizer::new(dim, 2).with_start_bits(8));
        assert_eq!(
            short_start,
            payload + 4 + 1,
            "an 8-bit start must be charged one byte"
        );
    }

    #[test]
    fn windows_match_the_state_chain() {
        for &(dim, bits, mem) in &[
            (128usize, 2u8, 12u8),
            (128, 3, 12), // 15-bit window, emissions straddle byte boundaries
            (128, 4, 12), // 16-bit window, exactly u16::MAX
            (64, 1, 8),
            (64, 4, 14), // 18-bit window: past u16
            (32, 1, 16), // 17-bit window, as rate_memory_sweep.py uses
        ] {
            let q = TrellisQuantizer::new_with_overrides(dim, bits, Some(mem), None);
            let db = rand_unit(8, dim, 31 + mem as u64);
            for vector in &db {
                let tv = q.encode(vector);

                // Reference: the chain walk, exactly as decode_into performs it.
                let mut emissions = vec![0u16; dim / q.v];
                unpack_indices(&tv.codes, q.b_step, &mut emissions);
                let mut expected = Vec::with_capacity(dim / q.v);
                let mut state = tv.start;
                for &e in &emissions {
                    let w = q.window(state, e as u32);
                    expected.push(w);
                    state = w & q.state_mask;
                }

                let mut got = vec![0u32; dim / q.v];
                q.windows_into(&tv, &mut got);
                assert_eq!(
                    got,
                    expected,
                    "window stream diverges from the chain walk \
                     (dim={dim}, bits={bits}, mem={mem}, width={})",
                    mem as u32 + q.b_step as u32
                );
            }
        }
    }

    #[test]
    fn codes_only_matches_recon_cache_ranking() {
        // Codes-only must return the same top-k as reconstruction-cache mode:
        // it re-derives the same x̄ per query instead of reading the fp32 cache, so
        // the scores (hence ranking) are identical up to simd::dot's benign lane
        // reassociation. This test pins the equivalence.
        let dim = 128;
        let db = rand_unit(200, dim, 17);
        let q_recon = {
            let mut q = TrellisQuantizer::new(dim, 2);
            for (i, v) in db.iter().enumerate() {
                q.add(i as ItemId, v);
            }
            q
        };
        let q_codes = {
            let mut q = TrellisQuantizer::new(dim, 2).with_codes_only(true);
            for (i, v) in db.iter().enumerate() {
                q.add(i as ItemId, v);
            }
            q
        };
        // codes-only drops the fp32 recon side-table (the RAM-compressed claim).
        assert!(q_codes.entries.iter().all(|(_, tv)| tv.recon().is_empty()));
        assert!(q_recon
            .entries
            .iter()
            .all(|(_, tv)| tv.recon().len() == dim));
        let codes_memory = q_codes.memory_breakdown();
        let cached_memory = q_recon.memory_breakdown();
        assert_eq!(codes_memory.cache_bytes, 0);
        assert_eq!(
            cached_memory.cache_bytes,
            db.len() * dim * std::mem::size_of::<f32>()
        );
        assert!(cached_memory.total_resident_bytes() > codes_memory.total_resident_bytes());
        let queries = rand_unit(20, dim, 23);
        for qv in &queries {
            let a: Vec<ItemId> = q_recon
                .search(qv, 10)
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            let b: Vec<ItemId> = q_codes
                .search(qv, 10)
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            assert_eq!(a, b, "codes-only top-10 differs from recon-cache");
        }
    }

    #[test]
    fn residual_control_variate_composes_with_centering() {
        // The control variate estimates the inner product against the STORED
        // direction, which under centering is the unit residual to the entry's
        // centroid. Recomposing with the exact centroid term and the residual norm is
        // what puts entries from DIFFERENT centroids on one scale; without it the arm
        // ranks a different quantity and is silently incompatible with every centered
        // configuration, the systems tier included.
        //
        // TWO separated clusters are essential. With one centroid `qm` is the same
        // constant for every candidate and cancels out of the ranking, so a
        // single-cluster version of this test passes whether or not the recomposition
        // is there -- verified by mutation.
        let dim = 64;
        let mut a = vec![0.0f32; dim];
        let mut b = vec![0.0f32; dim];
        a[0] = 1.0;
        b[1] = 1.0;
        let cluster = |seed: u64, centre: &[f32]| -> Vec<Vec<f32>> {
            rand_unit(200, dim, seed)
                .iter()
                .map(|r| {
                    let v: Vec<f32> = centre
                        .iter()
                        .zip(r)
                        .map(|(c, x)| 0.85 * c + 0.5 * x)
                        .collect();
                    let n = l2_norm(&v).max(f32::EPSILON);
                    v.iter().map(|x| x / n).collect()
                })
                .collect()
        };
        let mut db = cluster(17, &a);
        db.extend(cluster(23, &b));
        let mean = |vs: &[Vec<f32>]| -> Vec<f32> {
            let mut c = vec![0.0f32; dim];
            for v in vs {
                for (acc, x) in c.iter_mut().zip(v) {
                    *acc += *x;
                }
            }
            let n = vs.len() as f32;
            c.iter().map(|x| x / n).collect()
        };
        let centroids = vec![mean(&db[..200]), mean(&db[200..])];
        let mut q = TrellisQuantizer::new(dim, 2)
            .with_codes_only(true)
            .with_resid(dim)
            .with_centroids(centroids);
        for (i, v) in db.iter().enumerate() {
            q.add(i as ItemId, v);
        }
        assert!(
            q.entries.iter().any(|(_, tv)| tv.cidx == 0)
                && q.entries.iter().any(|(_, tv)| tv.cidx == 1),
            "both centroids must own entries, or the cross-centroid scale is untested"
        );
        let queries = rand_unit(150, dim, 19);
        let mut hits = 0usize;
        for qv in &queries {
            let ip = |v: &Vec<f32>| v.iter().zip(qv).map(|(x, y)| x * y).sum::<f32>();
            let exact = db
                .iter()
                .enumerate()
                .max_by(|(_, x), (_, y)| {
                    ip(x)
                        .partial_cmp(&ip(y))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i as ItemId)
                .unwrap();
            if q.search(qv, 1)[0].0 == exact {
                hits += 1;
            }
        }
        assert!(
            hits * 2 > queries.len(),
            "centered control-variate search agreed with exact top-1 only {hits}/{} \
             times: the estimate is not being recomposed across centroids",
            queries.len()
        );
    }

    #[test]
    fn probe_resid_estimator_error() {
        let dim = 128;
        let db = rand_unit(300, dim, 5);
        let queries = rand_unit(60, dim, 7);
        for &m in &[0usize, dim] {
            let mut q = TrellisQuantizer::new(dim, 2);
            if m > 0 {
                q = q.with_resid(m);
            }
            for (i, v) in db.iter().enumerate() {
                q.add(i as ItemId, v);
            }
            let mut err = 0.0f64;
            let mut n = 0usize;
            let mut spearman_hits = 0usize;
            for qv in &queries {
                let got = q.search(qv, 300);
                let est: std::collections::HashMap<ItemId, f32> = got.iter().cloned().collect();
                for (i, v) in db.iter().enumerate() {
                    let truth: f32 = v.iter().zip(qv).map(|(a, b)| a * b).sum();
                    if let Some(e) = est.get(&(i as ItemId)) {
                        err += ((*e - truth) as f64).abs();
                        n += 1;
                    }
                }
                let exact = db
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        let ip = |v: &Vec<f32>| v.iter().zip(qv).map(|(x, y)| x * y).sum::<f32>();
                        ip(a).partial_cmp(&ip(b)).unwrap()
                    })
                    .map(|(i, _)| i as ItemId)
                    .unwrap();
                if got[0].0 == exact {
                    spearman_hits += 1;
                }
            }
            eprintln!(
                "resid_m={m:4}  mean|est-true|={:.5}  top1={spearman_hits}/{}",
                err / n as f64,
                queries.len()
            );
        }
    }

    #[test]
    fn asymmetric_stage1_beats_binarizing_the_query() {
        // Stage 1 can compare the query to a sign code two ways: keep the query in
        // full precision (`⟨q_r, sign(u_r)⟩`) or binarize it too and count bit
        // disagreements. The second throws away every query magnitude, so it should
        // retain the true nearest neighbour less often at a small shortlist.
        //
        // The cluster must be TIGHT for this to model an IVF posting list: with
        // vectors spread over the whole sphere the centroid collapses toward the
        // origin, every residual norm is ~1, and the geometry stops resembling the
        // thing under test.
        let dim = 128;
        let mut base = vec![0.0f32; dim];
        base[0] = 1.0;
        let db: Vec<Vec<f32>> = rand_unit(600, dim, 71)
            .iter()
            .map(|r| {
                let v: Vec<f32> = base.iter().zip(r).map(|(b, x)| 0.8 * b + 0.6 * x).collect();
                let n = l2_norm(&v).max(f32::EPSILON);
                v.iter().map(|x| x / n).collect()
            })
            .collect();
        let centroid = {
            let mut c = vec![0.0f32; dim];
            for v in &db {
                for (acc, x) in c.iter_mut().zip(v) {
                    *acc += *x;
                }
            }
            let n = db.len() as f32;
            c.iter().map(|x| x / n).collect::<Vec<f32>>()
        };
        let build = |asym: bool| {
            let mut q = TrellisQuantizer::new(dim, 2)
                .with_codes_only(true)
                .with_shortlist(16)
                .with_shortlist_asym(asym)
                .with_centroids(vec![centroid.clone()]);
            for (i, v) in db.iter().enumerate() {
                q.add(i as ItemId, v);
            }
            q
        };
        let (asym, ham) = (build(true), build(false));
        let queries = rand_unit(150, dim, 73);
        let (mut asym_hits, mut ham_hits) = (0usize, 0usize);
        for qv in &queries {
            // C=N reranks the whole list, so this is the exhaustive answer stage 1
            // is trying to keep.
            let exact = asym.search_shortlist(qv, 1, db.len())[0].0;
            if asym.search_shortlist(qv, 1, 16)[0].0 == exact {
                asym_hits += 1;
            }
            if ham.search_shortlist(qv, 1, 16)[0].0 == exact {
                ham_hits += 1;
            }
        }
        assert!(
            asym_hits > ham_hits,
            "asymmetric stage 1 did not beat the binarized one \
             (asym {asym_hits}, hamming {ham_hits} of {})",
            queries.len()
        );
    }

    #[test]
    fn companion_c_eq_n_matches_full_scan() {
        // At C=N the sign-code shortlist prunes nothing, so the two-stage search reranks
        // the whole DB and must return exactly the plain codes-only top-k.
        let dim = 128;
        let db = rand_unit(200, dim, 31);
        let plain = {
            let mut q = TrellisQuantizer::new(dim, 2).with_codes_only(true);
            for (i, v) in db.iter().enumerate() {
                q.add(i as ItemId, v);
            }
            q
        };
        let comp = {
            let mut q = TrellisQuantizer::new(dim, 2)
                .with_codes_only(true)
                .with_shortlist(1);
            for (i, v) in db.iter().enumerate() {
                q.add(i as ItemId, v);
            }
            q
        };
        assert!(
            comp.entries
                .iter()
                .all(|(_, tv)| tv.sign_code().len() == dim / 8),
            "companion must emit a dim/8-byte sign code per vector"
        );
        assert!(
            plain
                .entries
                .iter()
                .all(|(_, tv)| tv.sign_code().is_empty()),
            "plain trellis must not emit a sign code (field inert when off)"
        );
        for qv in &rand_unit(20, dim, 37) {
            let a: Vec<ItemId> = plain.search(qv, 10).into_iter().map(|(id, _)| id).collect();
            let b: Vec<ItemId> = comp
                .search_shortlist(qv, 10, db.len())
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            assert_eq!(
                a, b,
                "companion at C=N differs from plain codes-only top-10"
            );
        }
    }

    #[test]
    fn companion_recall_monotone_in_c() {
        // Recall@10 of the two-stage search vs the full codes-only scan is non-decreasing
        // in the shortlist size C; C=N is exact against itself.
        let dim = 128;
        let db = rand_unit(500, dim, 41);
        let comp = {
            let mut q = TrellisQuantizer::new(dim, 2)
                .with_codes_only(true)
                .with_shortlist(1);
            for (i, v) in db.iter().enumerate() {
                q.add(i as ItemId, v);
            }
            q
        };
        let queries = rand_unit(30, dim, 43);
        let recall_at = |c: usize| -> f64 {
            let (mut hit, mut tot) = (0usize, 0usize);
            for qv in &queries {
                let gold: std::collections::HashSet<ItemId> = comp
                    .search_shortlist(qv, 10, db.len())
                    .into_iter()
                    .map(|(id, _)| id)
                    .collect();
                let got: std::collections::HashSet<ItemId> = comp
                    .search_shortlist(qv, 10, c)
                    .into_iter()
                    .map(|(id, _)| id)
                    .collect();
                hit += gold.intersection(&got).count();
                tot += gold.len();
            }
            hit as f64 / tot as f64
        };
        let mut prev = 0.0;
        for &c in &[50usize, 100, 200, db.len()] {
            let r = recall_at(c);
            assert!(
                r >= prev - 1e-9,
                "recall@10 dropped as C grew: {prev} -> {r} at C={c}"
            );
            prev = r;
        }
        assert!(
            (recall_at(db.len()) - 1.0).abs() < 1e-9,
            "C=N must be exact vs itself"
        );
    }
}
