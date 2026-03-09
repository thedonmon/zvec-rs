//! NEON SIMD distance kernels for aarch64.
//!
//! Processes 4 floats per iteration (128-bit NEON registers).
//! On Apple Silicon, NEON is always available.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON squared L2 distance.
///
/// # Safety
/// Caller must ensure NEON is supported (always true on aarch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn l2_squared_neon(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum = vdupq_n_f32(0.0);

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a_ptr.add(offset));
        let vb = vld1q_f32(b_ptr.add(offset));
        let diff = vsubq_f32(va, vb);
        // FMA: sum += diff * diff
        sum = vfmaq_f32(sum, diff, diff);
    }

    let mut result = vaddvq_f32(sum);

    let start = chunks * 4;
    for i in 0..remainder {
        let d = a[start + i] - b[start + i];
        result += d * d;
    }

    result
}

/// NEON inner product.
///
/// # Safety
/// Caller must ensure NEON is supported.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn inner_product_neon(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum = vdupq_n_f32(0.0);

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a_ptr.add(offset));
        let vb = vld1q_f32(b_ptr.add(offset));
        sum = vfmaq_f32(sum, va, vb);
    }

    let mut result = vaddvq_f32(sum);

    let start = chunks * 4;
    for i in 0..remainder {
        result += a[start + i] * b[start + i];
    }

    result
}

/// NEON cosine similarity = dot(a,b) / (|a| * |b|).
///
/// # Safety
/// Caller must ensure NEON is supported.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn cosine_similarity_neon(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut dot_sum = vdupq_n_f32(0.0);
    let mut norm_a_sum = vdupq_n_f32(0.0);
    let mut norm_b_sum = vdupq_n_f32(0.0);

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a_ptr.add(offset));
        let vb = vld1q_f32(b_ptr.add(offset));
        dot_sum = vfmaq_f32(dot_sum, va, vb);
        norm_a_sum = vfmaq_f32(norm_a_sum, va, va);
        norm_b_sum = vfmaq_f32(norm_b_sum, vb, vb);
    }

    let mut dot = vaddvq_f32(dot_sum);
    let mut norm_a = vaddvq_f32(norm_a_sum);
    let mut norm_b = vaddvq_f32(norm_b_sum);

    let start = chunks * 4;
    for i in 0..remainder {
        let ai = a[start + i];
        let bi = b[start + i];
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "aarch64")]
    use super::*;

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_l2_matches_scalar() {
        let mut rng = rand::thread_rng();
        use rand::Rng;
        for dims in [4, 8, 16, 32, 128, 384, 768, 1536, 1537] {
            let a: Vec<f32> = (0..dims).map(|_| rng.gen::<f32>() - 0.5).collect();
            let b: Vec<f32> = (0..dims).map(|_| rng.gen::<f32>() - 0.5).collect();

            let scalar = super::super::scalar::l2_squared(&a, &b);
            let simd = unsafe { l2_squared_neon(&a, &b) };
            assert!(
                (scalar - simd).abs() < 1e-3,
                "L2 mismatch at dims={dims}: scalar={scalar}, neon={simd}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_cosine_matches_scalar() {
        let mut rng = rand::thread_rng();
        use rand::Rng;
        for dims in [4, 8, 16, 32, 128, 384, 768, 1536, 1537] {
            let a: Vec<f32> = (0..dims).map(|_| rng.gen::<f32>() - 0.5).collect();
            let b: Vec<f32> = (0..dims).map(|_| rng.gen::<f32>() - 0.5).collect();

            let scalar = super::super::scalar::cosine_similarity(&a, &b);
            let simd = unsafe { cosine_similarity_neon(&a, &b) };
            assert!(
                (scalar - simd).abs() < 1e-3,
                "Cosine mismatch at dims={dims}: scalar={scalar}, neon={simd}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_ip_matches_scalar() {
        let mut rng = rand::thread_rng();
        use rand::Rng;
        for dims in [4, 8, 16, 32, 128, 384, 768, 1536, 1537] {
            let a: Vec<f32> = (0..dims).map(|_| rng.gen::<f32>() - 0.5).collect();
            let b: Vec<f32> = (0..dims).map(|_| rng.gen::<f32>() - 0.5).collect();

            let scalar = super::super::scalar::inner_product(&a, &b);
            let simd = unsafe { inner_product_neon(&a, &b) };
            assert!(
                (scalar - simd).abs() < 1e-3,
                "IP mismatch at dims={dims}: scalar={scalar}, neon={simd}"
            );
        }
    }
}
