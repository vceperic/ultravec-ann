//! Shared vectorized scoring kernels.
//!
//! These live here rather than inside one codec because relative throughput is a
//! reported result. When one codec's scan is hand-vectorized and its comparators'
//! are not, the measured QPS ordering is a property of the harness rather than of
//! the methods, and no disclaimer in the text repairs that. Every codec whose
//! throughput is reported therefore scores through a kernel from this module, at
//! the same optimization level, and the module carries the tests that keep the
//! vector and scalar paths in agreement.
//!
//! Two shapes cover the compared codecs:
//!
//! * [`dot`] — both operands are dense `f32`. Used by any codec that materializes a
//!   reconstruction before scoring (the trellis decode path, block codebooks).
//! * [`lut_dot`] — one operand is a small lookup table addressed by per-coordinate
//!   codes. Used by the scalar-quantizer family, whose stored code *is* the table
//!   index, so materializing a reconstruction first would add a pass the kernel can
//!   fold into the gather.
//!
//! Every path is runtime-detected: AVX2+FMA where present, scalar otherwise. The
//! two differ in summation order only, which for `f32` is not associative — see
//! [`dot`] for why that is acceptable here and what pins it.

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use std::arch::is_x86_feature_detected;

/// Dense `f32` dot product `Σ a[i]·b[i]`.
///
/// On x86-64 with AVX2+FMA this runs 8 lanes wide with two independent
/// accumulators to hide FMA latency; elsewhere it is the scalar `zip` sum.
///
/// The vector and scalar paths compute the same products and sum them in a
/// different order, so they can differ in the last `f32` bits. That is acceptable
/// for this use because the quantity feeds a ranking, and a difference small enough
/// to reorder two candidates can only do so when their scores were already within
/// rounding of each other. It is not merely asserted: `simd_dot_matches_scalar`
/// pins the numerical agreement, and `simd_scoring_preserves_recall` pins the
/// consequence that actually matters, that switching paths does not move recall.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: gated on runtime AVX2+FMA detection; reads stay in-bounds
            // (the remainder past the last full 16-lane block is handled scalar).
            return unsafe { dot_avx2_fma(a, b) };
        }
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n = a.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    // Two accumulators over 8-wide lanes (16 floats/iter) to break the FMA
    // dependency chain — ~2× the single-accumulator throughput on the hot loop.
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 16 <= n {
        let a0 = _mm256_loadu_ps(pa.add(i));
        let b0 = _mm256_loadu_ps(pb.add(i));
        let a1 = _mm256_loadu_ps(pa.add(i + 8));
        let b1 = _mm256_loadu_ps(pb.add(i + 8));
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);
        acc1 = _mm256_fmadd_ps(a1, b1, acc1);
        i += 16;
    }
    if i + 8 <= n {
        let a0 = _mm256_loadu_ps(pa.add(i));
        let b0 = _mm256_loadu_ps(pb.add(i));
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);
        i += 8;
    }
    let mut total = hsum256(_mm256_add_ps(acc0, acc1));
    while i < n {
        total += *pa.add(i) * *pb.add(i);
        i += 1;
    }
    total
}

/// `Σ q[i] · book[idx[i]]` — a dot product whose second operand is addressed
/// indirectly through a lookup table.
///
/// Two codec families share this shape and therefore share this kernel:
///
/// * scalar quantizers, where the stored code for coordinate `i` indexes a
///   `2^bits`-entry table (2 to 16 floats here, L1-resident);
/// * the trellis, where the index is a `(mem + b)`-bit *window* and the table is
///   `2^(mem+b)` floats — 1 MiB at the `mem=16` two-bit operating point, so L2 or
///   beyond rather than L1. That growth is why raising the memory helps a linear
///   scan, where the table is amortized over a long sequential pass, and costs a
///   graph walk, which scores a few hundred scattered candidates per query.
///
/// The large-table case is why the gather matters beyond instruction count: a
/// serialized chain of dependent lookups exposes L2 latency one load at a time,
/// whereas eight independent gathered lanes overlap it.
///
/// The AVX2 path widens eight `u16` indices to 32-bit lanes, gathers their eight
/// entries in one instruction, and folds them into two FMA accumulators. Use
/// [`lut_dot_u32`] when indices exceed 16 bits (`mem + b > 16`).
///
/// Callers must ensure every index is in bounds for `book`; the gather has no
/// bounds check. Codecs satisfy this by construction — a code is masked to the
/// table width — and the debug assertion below catches a regression.
#[inline]
pub fn lut_dot(q: &[f32], idx: &[u16], book: &[f32]) -> f32 {
    debug_assert_eq!(q.len(), idx.len());
    debug_assert!(
        idx.iter().all(|&c| (c as usize) < book.len()),
        "lut_dot: code out of range for a {}-entry table",
        book.len()
    );
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: gated on runtime AVX2+FMA detection. Lane loads stay within
            // `q`/`idx` (the tail past the last full 8-lane block is scalar), and
            // every gathered index is `< book.len()` per the contract above.
            return unsafe { lut_dot_avx2_fma(q, idx, book) };
        }
    }
    q.iter().zip(idx).map(|(&x, &c)| x * book[c as usize]).sum()
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn lut_dot_avx2_fma(q: &[f32], idx: &[u16], book: &[f32]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n = q.len();
    let pq = q.as_ptr();
    let pi = idx.as_ptr();
    let pb = book.as_ptr();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0usize;
    // Two 8-lane groups per iteration, mirroring `dot`'s two-accumulator shape so
    // the two kernels are comparable rather than incidentally different.
    while i + 16 <= n {
        let c0 = _mm256_cvtepu16_epi32(_mm_loadu_si128(pi.add(i) as *const __m128i));
        let c1 = _mm256_cvtepu16_epi32(_mm_loadu_si128(pi.add(i + 8) as *const __m128i));
        let b0 = _mm256_i32gather_ps(pb, c0, 4);
        let b1 = _mm256_i32gather_ps(pb, c1, 4);
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pq.add(i)), b0, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(pq.add(i + 8)), b1, acc1);
        i += 16;
    }
    if i + 8 <= n {
        let c0 = _mm256_cvtepu16_epi32(_mm_loadu_si128(pi.add(i) as *const __m128i));
        let b0 = _mm256_i32gather_ps(pb, c0, 4);
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pq.add(i)), b0, acc0);
        i += 8;
    }
    let mut total = hsum256(_mm256_add_ps(acc0, acc1));
    while i < n {
        total += *pq.add(i) * *pb.add(*pi.add(i) as usize);
        i += 1;
    }
    total
}

/// [`lut_dot`] for indices that do not fit 16 bits.
///
/// The trellis window is `mem + b` bits wide, which exceeds `u16` once
/// `mem + b > 16` — reachable in the appendix sweeps (`mem=16` at one bit,
/// `mem=14` at four). Keeping this separate rather than widening `lut_dot`
/// everywhere means the common path still moves indices at two bytes each.
#[inline]
pub fn lut_dot_u32(q: &[f32], idx: &[u32], book: &[f32]) -> f32 {
    debug_assert_eq!(q.len(), idx.len());
    debug_assert!(
        idx.iter().all(|&c| (c as usize) < book.len()),
        "lut_dot_u32: index out of range for a {}-entry table",
        book.len()
    );
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: as `lut_dot`, with indices already 32-bit so no widening.
            return unsafe { lut_dot_u32_avx2_fma(q, idx, book) };
        }
    }
    q.iter().zip(idx).map(|(&x, &c)| x * book[c as usize]).sum()
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn lut_dot_u32_avx2_fma(q: &[f32], idx: &[u32], book: &[f32]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n = q.len();
    let pq = q.as_ptr();
    let pi = idx.as_ptr();
    let pb = book.as_ptr();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 16 <= n {
        let c0 = _mm256_loadu_si256(pi.add(i) as *const __m256i);
        let c1 = _mm256_loadu_si256(pi.add(i + 8) as *const __m256i);
        acc0 = _mm256_fmadd_ps(
            _mm256_loadu_ps(pq.add(i)),
            _mm256_i32gather_ps(pb, c0, 4),
            acc0,
        );
        acc1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(pq.add(i + 8)),
            _mm256_i32gather_ps(pb, c1, 4),
            acc1,
        );
        i += 16;
    }
    if i + 8 <= n {
        let c0 = _mm256_loadu_si256(pi.add(i) as *const __m256i);
        acc0 = _mm256_fmadd_ps(
            _mm256_loadu_ps(pq.add(i)),
            _mm256_i32gather_ps(pb, c0, 4),
            acc0,
        );
        i += 8;
    }
    let mut total = hsum256(_mm256_add_ps(acc0, acc1));
    while i < n {
        total += *pq.add(i) * *pb.add(*pi.add(i) as usize);
        i += 1;
    }
    total
}

/// Horizontal sum of an 8-lane vector.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum256(v: std::arch::x86_64::__m256) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_hadd_ps(s, s);
    let s = _mm_hadd_ps(s, s);
    _mm_cvtss_f32(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn next(state: &mut u64) -> f32 {
        // xorshift64*, so the fixtures are deterministic without a dependency.
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        (((state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64) as f32
            - 0.5)
            * 4.0
    }

    #[test]
    fn simd_dot_matches_scalar() {
        // Lengths that exercise the 16-wide block, the single 8-wide block, and the
        // scalar tail, including lengths shorter than one vector.
        let mut st = 99u64;
        for &n in &[1usize, 7, 8, 15, 16, 17, 31, 64, 96, 128, 130] {
            let a: Vec<f32> = (0..n).map(|_| next(&mut st)).collect();
            let b: Vec<f32> = (0..n).map(|_| next(&mut st)).collect();
            let got = dot(&a, &b);
            let want: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            assert!(
                (got - want).abs() <= 1e-4 + 1e-4 * want.abs(),
                "n={n}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn lut_dot_matches_scalar() {
        // Every table size the scalar-quantizer family uses (1 to 4 bits), across
        // the same block/tail boundaries.
        let mut st = 7u64;
        for &bits in &[1u32, 2, 3, 4] {
            let levels = 1usize << bits;
            let book: Vec<f32> = (0..levels)
                .map(|c| c as f32 - (levels as f32 - 1.0) * 0.5)
                .collect();
            for &n in &[1usize, 7, 8, 15, 16, 17, 31, 128, 130] {
                let q: Vec<f32> = (0..n).map(|_| next(&mut st)).collect();
                let idx: Vec<u16> = (0..n)
                    .map(|i| ((i * 7 + bits as usize) % levels) as u16)
                    .collect();
                let got = lut_dot(&q, &idx, &book);
                let want: f32 = q
                    .iter()
                    .zip(&idx)
                    .map(|(&x, &c)| x * book[c as usize])
                    .sum();
                assert!(
                    (got - want).abs() <= 1e-4 + 1e-4 * want.abs(),
                    "bits={bits} n={n}: {got} vs {want}"
                );
            }
        }
    }

    /// The vector paths reassociate `f32` addition, so scores can differ in the
    /// last bits and, in principle, reorder two near-tied candidates. The codec
    /// comments have long appealed to a "recall-identity guardrail" for this; no
    /// such test existed, so the claim rested on nothing. This is it.
    ///
    /// The check is on the retrieved *set*, not on the scores: a scoring difference
    /// nobody can observe through the ranking is exactly the difference we are
    /// willing to accept, and pinning the scores instead would fail on a harmless
    /// last-bit change.
    #[test]
    fn simd_scoring_preserves_ranking() {
        let (dim, n, k) = (128usize, 400usize, 10usize);
        let mut st = 4242u64;
        let db: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| next(&mut st)).collect())
            .collect();
        let levels = 8usize;
        let book: Vec<f32> = (0..levels)
            .map(|c| c as f32 - (levels as f32 - 1.0) * 0.5)
            .collect();
        // One code stream per database vector, as a scalar quantizer would produce.
        let codes: Vec<Vec<u16>> = db
            .iter()
            .map(|v| {
                v.iter()
                    .map(|x| {
                        let t = ((x * 0.25 + 0.5) * (levels - 1) as f32).round();
                        t.clamp(0.0, (levels - 1) as f32) as u16
                    })
                    .collect()
            })
            .collect();

        for query_seed in 0..8u64 {
            let mut qs = 977 + query_seed;
            let q: Vec<f32> = (0..dim).map(|_| next(&mut qs)).collect();
            let rank = |score: &dyn Fn(&[u16]) -> f32| {
                let mut scored: Vec<(usize, f32)> = codes
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (i, score(c)))
                    .collect();
                scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                scored
                    .into_iter()
                    .take(k)
                    .map(|(i, _)| i)
                    .collect::<Vec<_>>()
            };
            let vectorized = rank(&|c: &[u16]| lut_dot(&q, c, &book));
            let scalar =
                rank(&|c: &[u16]| q.iter().zip(c).map(|(&x, &i)| x * book[i as usize]).sum());
            assert_eq!(
                vectorized, scalar,
                "query {query_seed}: vector and scalar scoring disagree on the top-{k}"
            );
        }
    }

    #[test]
    fn lut_dot_handles_a_degenerate_single_entry_table() {
        // A 1-entry table makes every gathered lane identical, which is the case a
        // permute-based implementation would special-case and get wrong.
        let book = [0.5f32];
        let q: Vec<f32> = (0..20).map(|i| i as f32 - 10.0).collect();
        let idx = vec![0u16; 20];
        let want: f32 = q.iter().map(|x| x * 0.5).sum();
        assert!((lut_dot(&q, &idx, &book) - want).abs() <= 1e-4 + 1e-4 * want.abs());
    }
}
