//! Distance/similarity kernels for vector comparison.
//!
//! All functions operate on `&[f32]` slices and return `f32`.
//! Lower values = more similar for L2; higher values = more similar for IP/Cosine.
//!
//! The implementations use manual loop unrolling (4-wide accumulators) which
//! allows the compiler to auto-vectorize to SIMD (SSE/AVX on x86, NEON on ARM).
//! On x86_64, compile with `RUSTFLAGS="-C target-cpu=native"` or
//! `-C target-feature=+avx2,+fma` for best performance.

#[cfg(target_arch = "x86_64")]
mod x86;

#[cfg(target_arch = "aarch64")]
mod neon;

mod scalar;

/// Metric type for vector similarity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricType {
    /// L2 (Euclidean) squared distance. Lower = more similar.
    L2,
    /// Inner product (dot product). Higher = more similar.
    /// With L2-normalized vectors, equivalent to cosine similarity.
    IP,
    /// Cosine similarity. Higher = more similar.
    /// Normalizes vectors before computing dot product.
    Cosine,
    /// Maximum Inner Product Search. Higher = more similar.
    /// Uses inner product directly — equivalent to IP but with MIPS semantics.
    MIPS,
}

impl MetricType {
    /// Returns true if higher scores mean more similar.
    #[inline]
    pub fn is_similarity(self) -> bool {
        matches!(self, MetricType::IP | MetricType::Cosine | MetricType::MIPS)
    }

    /// Compare two distances. Returns true if `a` is closer/better than `b`.
    #[inline]
    pub fn is_better(self, a: f32, b: f32) -> bool {
        if self.is_similarity() {
            a > b
        } else {
            a < b
        }
    }

    /// The worst possible distance value (used for initialization).
    #[inline]
    pub fn worst_distance(self) -> f32 {
        if self.is_similarity() {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        }
    }

    /// Compute distance between two vectors using this metric.
    /// Automatically dispatches to the best available implementation.
    #[inline]
    pub fn distance(self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            MetricType::L2 => l2_squared(a, b),
            MetricType::IP => inner_product(a, b),
            MetricType::Cosine => cosine_similarity(a, b),
            MetricType::MIPS => inner_product(a, b),
        }
    }
}

/// Squared L2 (Euclidean) distance.
/// Dispatches to SIMD implementation when available.
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { x86::l2_squared_avx2(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on aarch64
        return unsafe { neon::l2_squared_neon(a, b) };
    }
    #[allow(unreachable_code)]
    scalar::l2_squared(a, b)
}

/// Inner product (dot product).
/// Dispatches to SIMD implementation when available.
#[inline]
pub fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { x86::inner_product_avx2(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { neon::inner_product_neon(a, b) };
    }
    #[allow(unreachable_code)]
    scalar::inner_product(a, b)
}

/// Cosine similarity = dot(a, b) / (|a| * |b|).
/// Dispatches to SIMD implementation when available.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { x86::cosine_similarity_avx2(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { neon::cosine_similarity_neon(a, b) };
    }
    #[allow(unreachable_code)]
    scalar::cosine_similarity(a, b)
}

/// L2-normalize a vector in place.
#[inline]
pub fn l2_normalize(v: &mut [f32]) {
    scalar::l2_normalize(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_squared_identical() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        assert!((l2_squared(&v, &v) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_squared_known() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!((l2_squared(&a, &b) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_inner_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((inner_product(&a, &b) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_identical() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_ip_equals_cosine_for_normalized() {
        let mut a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut b = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        l2_normalize(&mut a);
        l2_normalize(&mut b);
        let ip = inner_product(&a, &b);
        let cos = cosine_similarity(&a, &b);
        assert!((ip - cos).abs() < 1e-5, "IP={ip}, Cosine={cos}");
    }

    #[test]
    fn test_odd_dimensions() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b = vec![7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum();
        assert!((l2_squared(&a, &b) - expected).abs() < 1e-6);
    }

    #[test]
    fn test_metric_type_is_better() {
        assert!(MetricType::L2.is_better(1.0, 2.0));
        assert!(MetricType::IP.is_better(2.0, 1.0));
        assert!(MetricType::Cosine.is_better(0.9, 0.5));
    }

    #[test]
    fn test_mips_is_similarity() {
        assert!(MetricType::MIPS.is_similarity());
        assert!(MetricType::MIPS.is_better(2.0, 1.0));
        assert!(!MetricType::MIPS.is_better(1.0, 2.0));
        assert_eq!(MetricType::MIPS.worst_distance(), f32::NEG_INFINITY);
    }

    #[test]
    fn test_mips_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let mips_dist = MetricType::MIPS.distance(&a, &b);
        let ip_dist = MetricType::IP.distance(&a, &b);
        assert!((mips_dist - ip_dist).abs() < 1e-6, "MIPS should use inner product");
        assert!((mips_dist - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_mips_search() {
        use crate::hnsw::{HnswIndex, HnswParams};

        let params = HnswParams::new(16, 100);
        let index = HnswIndex::new(3, MetricType::MIPS, params);

        // Insert vectors with varying inner products with query [1, 1, 1]
        index.insert(1, &[0.1, 0.1, 0.1]); // IP = 0.3
        index.insert(2, &[1.0, 1.0, 1.0]); // IP = 3.0
        index.insert(3, &[0.5, 0.5, 0.5]); // IP = 1.5
        index.insert(4, &[2.0, 2.0, 2.0]); // IP = 6.0

        let query = vec![1.0, 1.0, 1.0];
        let results = index.search(&query, 4);

        // Highest IP should be first
        assert_eq!(results[0].id, 4);
        assert_eq!(results[1].id, 2);
    }

    #[test]
    fn test_large_vectors() {
        // Test with typical embedding dimensions
        let mut rng = rand::thread_rng();
        use rand::Rng;
        for dims in [128, 384, 768, 1536] {
            let a: Vec<f32> = (0..dims).map(|_| rng.gen::<f32>() - 0.5).collect();
            let b: Vec<f32> = (0..dims).map(|_| rng.gen::<f32>() - 0.5).collect();

            // Compare SIMD result against scalar
            let l2_result = l2_squared(&a, &b);
            let l2_scalar = scalar::l2_squared(&a, &b);
            assert!(
                (l2_result - l2_scalar).abs() < 1e-3,
                "L2 mismatch at dims={dims}: {l2_result} vs {l2_scalar}"
            );

            let ip_result = inner_product(&a, &b);
            let ip_scalar = scalar::inner_product(&a, &b);
            assert!(
                (ip_result - ip_scalar).abs() < 1e-3,
                "IP mismatch at dims={dims}: {ip_result} vs {ip_scalar}"
            );
        }
    }
}
