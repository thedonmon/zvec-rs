//! AVX2 + FMA SIMD distance kernels for x86_64.
//!
//! These are called via runtime feature detection in the parent module.
//! Each function processes 8 floats per iteration (256-bit AVX2 registers).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AVX2 + FMA squared L2 distance.
///
/// Processes 8 floats per iteration using 256-bit SIMD registers.
/// Uses FMA (fused multiply-add) for better precision and throughput.
///
/// # Safety
/// Caller must ensure AVX2 and FMA are supported (checked via `is_x86_feature_detected!`).
#[target_feature(enable = "avx2,fma")]
pub unsafe fn l2_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let mut sum = _mm256_setzero_ps();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a_ptr.add(offset));
        let vb = _mm256_loadu_ps(b_ptr.add(offset));
        let diff = _mm256_sub_ps(va, vb);
        // FMA: sum = diff * diff + sum
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    // Horizontal sum of 8 floats
    let mut result = hsum_avx(sum);

    // Handle remainder
    let start = chunks * 8;
    for i in 0..remainder {
        let d = a[start + i] - b[start + i];
        result += d * d;
    }

    result
}

/// AVX2 + FMA inner product.
///
/// # Safety
/// Caller must ensure AVX2 and FMA are supported.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn inner_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let mut sum = _mm256_setzero_ps();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a_ptr.add(offset));
        let vb = _mm256_loadu_ps(b_ptr.add(offset));
        // FMA: sum = va * vb + sum
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    let mut result = hsum_avx(sum);

    let start = chunks * 8;
    for i in 0..remainder {
        result += a[start + i] * b[start + i];
    }

    result
}

/// AVX2 + FMA cosine similarity.
///
/// Computes dot(a, b) / (|a| * |b|) using three accumulators.
///
/// # Safety
/// Caller must ensure AVX2 and FMA are supported.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn cosine_similarity_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let mut dot = _mm256_setzero_ps();
    let mut norm_a = _mm256_setzero_ps();
    let mut norm_b = _mm256_setzero_ps();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a_ptr.add(offset));
        let vb = _mm256_loadu_ps(b_ptr.add(offset));
        // FMA: dot = va * vb + dot
        dot = _mm256_fmadd_ps(va, vb, dot);
        // FMA: norm_a = va * va + norm_a
        norm_a = _mm256_fmadd_ps(va, va, norm_a);
        // FMA: norm_b = vb * vb + norm_b
        norm_b = _mm256_fmadd_ps(vb, vb, norm_b);
    }

    let mut dot_sum = hsum_avx(dot);
    let mut norm_a_sum = hsum_avx(norm_a);
    let mut norm_b_sum = hsum_avx(norm_b);

    // Handle remainder
    let start = chunks * 8;
    for i in 0..remainder {
        let ai = a[start + i];
        let bi = b[start + i];
        dot_sum += ai * bi;
        norm_a_sum += ai * ai;
        norm_b_sum += bi * bi;
    }

    let denom = (norm_a_sum * norm_b_sum).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot_sum / denom
    }
}

/// Horizontal sum of 8 f32 values in a __m256.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_avx(v: __m256) -> f32 {
    // Add high 128 bits to low 128 bits
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);
    // Horizontal add pairs
    let shuf = _mm_movehdup_ps(sum128);
    let sum64 = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sum64, sum64);
    let sum32 = _mm_add_ss(sum64, shuf2);
    _mm_cvtss_f32(sum32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avx2_l2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return; // Skip on CPUs without AVX2+FMA
        }

        let mut rng = rand::thread_rng();
        use rand::Rng;
        for dims in [8, 16, 32, 128, 384, 768, 1536, 1537] {
            let a: Vec<f32> = (0..dims).map(|_| rng.gen::<f32>() - 0.5).collect();
            let b: Vec<f32> = (0..dims).map(|_| rng.gen::<f32>() - 0.5).collect();

            let scalar = super::super::scalar::l2_squared(&a, &b);
            let simd = unsafe { l2_squared_avx2(&a, &b) };
            assert!(
                (scalar - simd).abs() < 1e-3,
                "L2 mismatch at dims={dims}: scalar={scalar}, avx2={simd}"
            );
        }
    }

    #[test]
    fn test_avx2_cosine_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }

        let mut rng = rand::thread_rng();
        use rand::Rng;
        for dims in [8, 16, 32, 128, 384, 768, 1536, 1537] {
            let a: Vec<f32> = (0..dims).map(|_| rng.gen::<f32>() - 0.5).collect();
            let b: Vec<f32> = (0..dims).map(|_| rng.gen::<f32>() - 0.5).collect();

            let scalar = super::super::scalar::cosine_similarity(&a, &b);
            let simd = unsafe { cosine_similarity_avx2(&a, &b) };
            assert!(
                (scalar - simd).abs() < 1e-3,
                "Cosine mismatch at dims={dims}: scalar={scalar}, avx2={simd}"
            );
        }
    }

    #[test]
    fn test_avx2_ip_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }

        let mut rng = rand::thread_rng();
        use rand::Rng;
        for dims in [8, 16, 32, 128, 384, 768, 1536, 1537] {
            let a: Vec<f32> = (0..dims).map(|_| rng.gen::<f32>() - 0.5).collect();
            let b: Vec<f32> = (0..dims).map(|_| rng.gen::<f32>() - 0.5).collect();

            let scalar = super::super::scalar::inner_product(&a, &b);
            let simd = unsafe { inner_product_avx2(&a, &b) };
            assert!(
                (scalar - simd).abs() < 1e-3,
                "IP mismatch at dims={dims}: scalar={scalar}, avx2={simd}"
            );
        }
    }
}
