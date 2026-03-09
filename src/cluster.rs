//! Standalone clustering algorithms for vector data preprocessing.
//!
//! Provides k-means, mini-batch k-means, and the elbow method for
//! choosing the optimal number of clusters.

use std::ops::Range;

use rand::seq::SliceRandom;
use rand::Rng;

use crate::distance::MetricType;

/// Result of a k-means clustering run.
#[derive(Debug, Clone)]
pub struct KMeansResult {
    /// Centroid vectors, one per cluster (length = k).
    pub centroids: Vec<Vec<f32>>,
    /// Cluster assignment for each input vector (length = data.len()).
    pub assignments: Vec<usize>,
    /// Sum of distances from each point to its assigned centroid.
    pub inertia: f32,
}

/// Assign a single vector to the nearest centroid, returning (cluster_index, distance).
fn assign(vector: &[f32], centroids: &[Vec<f32>], metric: MetricType) -> (usize, f32) {
    let mut best_idx = 0;
    let mut best_dist = metric.worst_distance();
    for (i, c) in centroids.iter().enumerate() {
        let d = metric.distance(vector, c);
        if metric.is_better(d, best_dist) {
            best_dist = d;
            best_idx = i;
        }
    }
    (best_idx, best_dist)
}

/// Compute inertia: for distance metrics (L2), sum of distances.
/// For similarity metrics (IP/Cosine), we negate so that higher similarity
/// gives lower inertia (better clustering).
fn inertia_contribution(dist: f32, metric: MetricType) -> f32 {
    if metric.is_similarity() {
        // For similarity metrics, inertia = sum of (1 - similarity) or just -similarity
        // so that lower inertia = tighter clusters.
        -dist
    } else {
        dist
    }
}

/// Initialize centroids using k-means++ for better convergence.
fn kmeans_plus_plus(data: &[Vec<f32>], k: usize, metric: MetricType) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);

    // Pick the first centroid uniformly at random.
    let first = rng.gen_range(0..data.len());
    centroids.push(data[first].clone());

    // For each subsequent centroid, pick with probability proportional to distance squared.
    for _ in 1..k {
        let mut dists: Vec<f32> = data
            .iter()
            .map(|v| {
                let (_, d) = assign(v, &centroids, metric);
                // For similarity metrics, convert to a "distance" for weighting
                if metric.is_similarity() {
                    // Use (1 - similarity) as distance; clamp to 0
                    (1.0 - d).max(0.0)
                } else {
                    d
                }
            })
            .collect();

        // Square the distances for k-means++ weighting
        for d in &mut dists {
            *d = *d * *d;
        }

        let total: f32 = dists.iter().sum();
        if total <= 0.0 {
            // All points are identical to existing centroids; just pick random
            let idx = rng.gen_range(0..data.len());
            centroids.push(data[idx].clone());
            continue;
        }

        // Weighted random selection
        let threshold = rng.gen::<f32>() * total;
        let mut cumulative = 0.0;
        let mut chosen = data.len() - 1;
        for (i, d) in dists.iter().enumerate() {
            cumulative += d;
            if cumulative >= threshold {
                chosen = i;
                break;
            }
        }
        centroids.push(data[chosen].clone());
    }

    centroids
}

/// Recompute centroids as the mean of assigned vectors.
fn recompute_centroids(
    data: &[Vec<f32>],
    assignments: &[usize],
    k: usize,
    dims: usize,
    old_centroids: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    let mut sums = vec![vec![0.0f32; dims]; k];
    let mut counts = vec![0usize; k];

    for (i, v) in data.iter().enumerate() {
        let c = assignments[i];
        counts[c] += 1;
        for (j, val) in v.iter().enumerate() {
            sums[c][j] += val;
        }
    }

    let mut centroids = Vec::with_capacity(k);
    for c_idx in 0..k {
        if counts[c_idx] > 0 {
            let n = counts[c_idx] as f32;
            centroids.push(sums[c_idx].iter().map(|s| s / n).collect());
        } else {
            // Keep old centroid if no points assigned
            centroids.push(old_centroids[c_idx].clone());
        }
    }
    centroids
}

/// Standard k-means clustering.
///
/// Partitions `data` into `k` clusters using Lloyd's algorithm with k-means++
/// initialization.
///
/// # Arguments
/// - `data`: slice of vectors, all with the same dimensionality.
/// - `k`: number of clusters.
/// - `max_iters`: maximum number of Lloyd iterations.
/// - `metric`: distance metric to use.
///
/// # Panics
/// Panics if `data` is empty, `k` is 0, or vectors have inconsistent dimensions.
pub fn kmeans(data: &[Vec<f32>], k: usize, max_iters: usize, metric: MetricType) -> KMeansResult {
    assert!(!data.is_empty(), "cannot cluster empty data");
    assert!(k > 0, "k must be at least 1");

    let dims = data[0].len();
    let k = k.min(data.len());

    let mut centroids = kmeans_plus_plus(data, k, metric);
    let mut assignments = vec![0usize; data.len()];

    for _ in 0..max_iters {
        // Assignment step
        let mut changed = false;
        for (i, v) in data.iter().enumerate() {
            let (best, _) = assign(v, &centroids, metric);
            if assignments[i] != best {
                assignments[i] = best;
                changed = true;
            }
        }

        // Recompute centroids
        centroids = recompute_centroids(data, &assignments, k, dims, &centroids);

        if !changed {
            break;
        }
    }

    // Compute final inertia
    let inertia: f32 = data
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let d = metric.distance(v, &centroids[assignments[i]]);
            inertia_contribution(d, metric)
        })
        .sum();

    KMeansResult {
        centroids,
        assignments,
        inertia,
    }
}

/// Mini-batch k-means clustering.
///
/// A faster approximation of k-means that uses random subsets (mini-batches)
/// of the data for each centroid update step. Well-suited for large datasets.
///
/// # Arguments
/// - `data`: slice of vectors.
/// - `k`: number of clusters.
/// - `batch_size`: number of samples per mini-batch.
/// - `max_iters`: maximum number of iterations.
/// - `metric`: distance metric to use.
///
/// # Panics
/// Panics if `data` is empty or `k` is 0.
pub fn mini_batch_kmeans(
    data: &[Vec<f32>],
    k: usize,
    batch_size: usize,
    max_iters: usize,
    metric: MetricType,
) -> KMeansResult {
    assert!(!data.is_empty(), "cannot cluster empty data");
    assert!(k > 0, "k must be at least 1");

    let dims = data[0].len();
    let k = k.min(data.len());
    let batch_size = batch_size.min(data.len());

    let mut centroids = kmeans_plus_plus(data, k, metric);
    let mut counts = vec![0usize; k]; // per-centroid sample counts for running mean

    let mut rng = rand::thread_rng();
    let mut indices: Vec<usize> = (0..data.len()).collect();

    for _ in 0..max_iters {
        // Sample a mini-batch
        indices.shuffle(&mut rng);
        let batch: Vec<usize> = indices[..batch_size].to_vec();

        // Assign batch points to nearest centroids
        let batch_assignments: Vec<(usize, usize)> = batch
            .iter()
            .map(|&idx| {
                let (c, _) = assign(&data[idx], &centroids, metric);
                (idx, c)
            })
            .collect();

        // Update centroids with running mean
        for (idx, c) in &batch_assignments {
            counts[*c] += 1;
            let lr = 1.0 / counts[*c] as f32;
            for j in 0..dims {
                centroids[*c][j] += lr * (data[*idx][j] - centroids[*c][j]);
            }
        }
    }

    // Final assignment of all points
    let mut assignments = vec![0usize; data.len()];
    let mut inertia = 0.0f32;
    for (i, v) in data.iter().enumerate() {
        let (best, d) = assign(v, &centroids, metric);
        assignments[i] = best;
        inertia += inertia_contribution(d, metric);
    }

    KMeansResult {
        centroids,
        assignments,
        inertia,
    }
}

/// Elbow method for choosing the optimal number of clusters.
///
/// Runs k-means for each value of k in `k_range` and returns the (k, inertia)
/// pairs. The optimal k is typically at the "elbow" — the point where inertia
/// starts decreasing more slowly.
///
/// # Arguments
/// - `data`: slice of vectors.
/// - `k_range`: range of k values to test (e.g., `2..10`).
/// - `max_iters`: maximum k-means iterations per run.
///
/// # Returns
/// A vector of `(k, inertia)` pairs, one per k in the range.
pub fn elbow_method(
    data: &[Vec<f32>],
    k_range: Range<usize>,
    max_iters: usize,
) -> Vec<(usize, f32)> {
    k_range
        .filter(|&k| k > 0)
        .map(|k| {
            let result = kmeans(data, k, max_iters, MetricType::L2);
            (k, result.inertia)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_clusters(
        centers: &[Vec<f32>],
        points_per_cluster: usize,
        spread: f32,
    ) -> Vec<Vec<f32>> {
        let mut rng = rand::thread_rng();
        let mut data = Vec::new();
        for center in centers {
            for _ in 0..points_per_cluster {
                let point: Vec<f32> = center
                    .iter()
                    .map(|&c| c + (rng.gen::<f32>() - 0.5) * spread)
                    .collect();
                data.push(point);
            }
        }
        data
    }

    #[test]
    fn test_kmeans_basic() {
        let data = make_clusters(
            &[vec![0.0, 0.0], vec![10.0, 10.0], vec![20.0, 0.0]],
            30,
            0.5,
        );

        let result = kmeans(&data, 3, 50, MetricType::L2);

        assert_eq!(result.centroids.len(), 3);
        assert_eq!(result.assignments.len(), data.len());
        assert!(result.inertia >= 0.0);

        // Check that each cluster has roughly 30 points
        let mut counts = vec![0; 3];
        for &a in &result.assignments {
            counts[a] += 1;
        }
        for c in &counts {
            assert!(*c >= 20, "cluster count too low: {}", c);
        }
    }

    #[test]
    fn test_kmeans_single_cluster() {
        let data = make_clusters(&[vec![5.0, 5.0]], 20, 0.1);
        let result = kmeans(&data, 1, 10, MetricType::L2);

        assert_eq!(result.centroids.len(), 1);
        assert!(result.assignments.iter().all(|&a| a == 0));
        // Centroid should be near (5, 5)
        assert!((result.centroids[0][0] - 5.0).abs() < 1.0);
        assert!((result.centroids[0][1] - 5.0).abs() < 1.0);
    }

    #[test]
    fn test_kmeans_k_exceeds_data() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = kmeans(&data, 10, 10, MetricType::L2);
        // k should be clamped to data.len()
        assert_eq!(result.centroids.len(), 2);
    }

    #[test]
    fn test_kmeans_ip_metric() {
        let data = make_clusters(
            &[vec![1.0, 0.0], vec![0.0, 1.0]],
            20,
            0.1,
        );

        let result = kmeans(&data, 2, 30, MetricType::IP);

        assert_eq!(result.centroids.len(), 2);
        assert_eq!(result.assignments.len(), data.len());
    }

    #[test]
    fn test_kmeans_cosine_metric() {
        let data = make_clusters(
            &[vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]],
            20,
            0.1,
        );

        let result = kmeans(&data, 3, 30, MetricType::Cosine);
        assert_eq!(result.centroids.len(), 3);
    }

    #[test]
    fn test_mini_batch_kmeans_basic() {
        let data = make_clusters(
            &[vec![0.0, 0.0], vec![10.0, 10.0], vec![20.0, 0.0]],
            30,
            0.5,
        );

        let result = mini_batch_kmeans(&data, 3, 20, 50, MetricType::L2);

        assert_eq!(result.centroids.len(), 3);
        assert_eq!(result.assignments.len(), data.len());
        assert!(result.inertia >= 0.0);

        // Check all clusters are populated
        let mut counts = vec![0; 3];
        for &a in &result.assignments {
            counts[a] += 1;
        }
        for c in &counts {
            assert!(*c > 0, "cluster should not be empty");
        }
    }

    #[test]
    fn test_mini_batch_kmeans_large_batch() {
        // batch_size > data.len() should be clamped
        let data = vec![vec![1.0], vec![2.0], vec![3.0]];
        let result = mini_batch_kmeans(&data, 2, 1000, 10, MetricType::L2);
        assert_eq!(result.centroids.len(), 2);
    }

    #[test]
    fn test_elbow_method() {
        let data = make_clusters(
            &[vec![0.0, 0.0], vec![10.0, 10.0], vec![20.0, 0.0]],
            30,
            0.5,
        );

        let results = elbow_method(&data, 1..6, 20);

        assert_eq!(results.len(), 5);
        // k values should be 1..6
        for (i, &(k, _)) in results.iter().enumerate() {
            assert_eq!(k, i + 1);
        }

        // Inertia should generally decrease as k increases
        // (not strictly guaranteed, but with well-separated clusters it should)
        assert!(
            results[0].1 > results[2].1,
            "inertia for k=1 ({}) should be > inertia for k=3 ({})",
            results[0].1,
            results[2].1
        );
    }

    #[test]
    fn test_elbow_shows_improvement() {
        let data = make_clusters(
            &[vec![0.0, 0.0], vec![100.0, 100.0]],
            50,
            1.0,
        );

        let results = elbow_method(&data, 1..5, 30);

        // Going from k=1 to k=2 should dramatically reduce inertia
        let inertia_1 = results[0].1;
        let inertia_2 = results[1].1;
        assert!(
            inertia_2 < inertia_1 * 0.5,
            "k=2 inertia ({}) should be much less than k=1 inertia ({})",
            inertia_2,
            inertia_1
        );
    }

    #[test]
    fn test_kmeans_deterministic_convergence() {
        // With identical points, should converge immediately
        let data = vec![vec![1.0, 1.0]; 10];
        let result = kmeans(&data, 1, 100, MetricType::L2);
        assert!((result.centroids[0][0] - 1.0).abs() < 1e-6);
        assert!((result.centroids[0][1] - 1.0).abs() < 1e-6);
        assert!(result.inertia < 1e-6);
    }

    #[test]
    #[should_panic(expected = "cannot cluster empty data")]
    fn test_kmeans_empty_data() {
        let data: Vec<Vec<f32>> = vec![];
        kmeans(&data, 3, 10, MetricType::L2);
    }

    #[test]
    #[should_panic(expected = "k must be at least 1")]
    fn test_kmeans_k_zero() {
        let data = vec![vec![1.0]];
        kmeans(&data, 0, 10, MetricType::L2);
    }
}
