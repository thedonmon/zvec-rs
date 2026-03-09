//! Scalar (auto-vectorizable) distance implementations.
//!
//! These use 4-wide accumulator unrolling which the compiler will
//! auto-vectorize to SIMD on most platforms with -O2 or higher.

/// Squared L2 distance — scalar with 4-wide unrolling.
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum0: f32 = 0.0;
    let mut sum1: f32 = 0.0;
    let mut sum2: f32 = 0.0;
    let mut sum3: f32 = 0.0;

    let mut i = 0;
    for _ in 0..chunks {
        let d0 = a[i] - b[i];
        let d1 = a[i + 1] - b[i + 1];
        let d2 = a[i + 2] - b[i + 2];
        let d3 = a[i + 3] - b[i + 3];
        sum0 += d0 * d0;
        sum1 += d1 * d1;
        sum2 += d2 * d2;
        sum3 += d3 * d3;
        i += 4;
    }

    for _ in 0..remainder {
        let d = a[i] - b[i];
        sum0 += d * d;
        i += 1;
    }

    sum0 + sum1 + sum2 + sum3
}

/// Inner product — scalar with 4-wide unrolling.
#[inline]
pub fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum0: f32 = 0.0;
    let mut sum1: f32 = 0.0;
    let mut sum2: f32 = 0.0;
    let mut sum3: f32 = 0.0;

    let mut i = 0;
    for _ in 0..chunks {
        sum0 += a[i] * b[i];
        sum1 += a[i + 1] * b[i + 1];
        sum2 += a[i + 2] * b[i + 2];
        sum3 += a[i + 3] * b[i + 3];
        i += 4;
    }

    for _ in 0..remainder {
        sum0 += a[i] * b[i];
        i += 1;
    }

    sum0 + sum1 + sum2 + sum3
}

/// Cosine similarity = dot(a, b) / (|a| * |b|).
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut dot0: f32 = 0.0;
    let mut dot1: f32 = 0.0;
    let mut dot2: f32 = 0.0;
    let mut dot3: f32 = 0.0;
    let mut norm_a: f32 = 0.0;
    let mut norm_b: f32 = 0.0;

    let mut i = 0;
    for _ in 0..chunks {
        dot0 += a[i] * b[i];
        dot1 += a[i + 1] * b[i + 1];
        dot2 += a[i + 2] * b[i + 2];
        dot3 += a[i + 3] * b[i + 3];
        norm_a += a[i] * a[i] + a[i + 1] * a[i + 1] + a[i + 2] * a[i + 2] + a[i + 3] * a[i + 3];
        norm_b += b[i] * b[i] + b[i + 1] * b[i + 1] + b[i + 2] * b[i + 2] + b[i + 3] * b[i + 3];
        i += 4;
    }

    for _ in 0..remainder {
        dot0 += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
        i += 1;
    }

    let dot = dot0 + dot1 + dot2 + dot3;
    let denom = (norm_a * norm_b).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// L2-normalize a vector in place.
#[inline]
pub fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}
