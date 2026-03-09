//! IVF (Inverted File) index with flat/exhaustive search within clusters.
//!
//! Partitions vectors into clusters using k-means, then searches only the
//! nearest clusters for approximate nearest neighbor queries.

use std::collections::HashMap;

use ordered_float::OrderedFloat;
use rand::Rng;

use crate::distance::MetricType;
use crate::hnsw::SearchResult;

/// Parameters for IVF index construction and search.
#[derive(Debug, Clone)]
pub struct IvfParams {
    /// Number of clusters (Voronoi cells). Default: 100.
    pub n_list: usize,
    /// Number of clusters to probe during search. Default: 10.
    pub n_probe: usize,
    /// Number of k-means iterations during training. Default: 10.
    pub n_iters: usize,
}

impl Default for IvfParams {
    fn default() -> Self {
        Self {
            n_list: 100,
            n_probe: 10,
            n_iters: 10,
        }
    }
}

/// IVF (Inverted File) index.
///
/// Vectors are assigned to the nearest centroid (cluster). At search time,
/// only the `n_probe` nearest clusters are exhaustively scanned.
pub struct IvfIndex {
    /// Centroid vectors, one per cluster.
    centroids: Vec<Vec<f32>>,
    /// Per-cluster storage: bucket[i] = [(external_id, vector), ...]
    buckets: Vec<Vec<(u64, Vec<f32>)>>,
    metric: MetricType,
    dims: usize,
    n_list: usize,
    /// Reverse map: external_id -> bucket index (for fast removal).
    id_map: HashMap<u64, usize>,
    /// Default n_probe for searches.
    default_n_probe: usize,
    /// K-means iterations for training.
    n_iters: usize,
}

impl IvfIndex {
    /// Create a new empty IVF index.
    ///
    /// Centroids are not initialized until `train()` is called.
    pub fn new(dims: usize, metric: MetricType, params: IvfParams) -> Self {
        Self {
            centroids: Vec::new(),
            buckets: Vec::new(),
            metric,
            dims,
            n_list: params.n_list,
            id_map: HashMap::new(),
            default_n_probe: params.n_probe,
            n_iters: params.n_iters,
        }
    }

    /// Train the index by running k-means clustering on the provided vectors.
    ///
    /// This initializes centroids and creates empty buckets. Any previously
    /// inserted vectors are lost — call `train` before inserting.
    ///
    /// # Panics
    /// Panics if `vectors` is empty or any vector has the wrong dimensionality.
    pub fn train(&mut self, vectors: &[Vec<f32>]) {
        assert!(!vectors.is_empty(), "cannot train on empty vector set");
        for v in vectors {
            assert_eq!(v.len(), self.dims, "vector dimension mismatch");
        }

        let k = self.n_list.min(vectors.len());
        self.n_list = k;

        // Initialize centroids by sampling k distinct vectors randomly.
        let mut rng = rand::thread_rng();
        let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
        let mut used = std::collections::HashSet::new();
        while centroids.len() < k {
            let idx = rng.gen_range(0..vectors.len());
            if used.insert(idx) {
                centroids.push(vectors[idx].clone());
            }
        }

        // K-means iterations.
        let mut assignments = vec![0usize; vectors.len()];

        for _ in 0..self.n_iters {
            // Assign each vector to the nearest centroid.
            for (i, v) in vectors.iter().enumerate() {
                let mut best_idx = 0;
                let mut best_dist = self.metric.worst_distance();
                for (c_idx, c) in centroids.iter().enumerate() {
                    let d = self.metric.distance(v, c);
                    if self.metric.is_better(d, best_dist) {
                        best_dist = d;
                        best_idx = c_idx;
                    }
                }
                assignments[i] = best_idx;
            }

            // Recompute centroids as the mean of assigned vectors.
            let mut sums = vec![vec![0.0f32; self.dims]; k];
            let mut counts = vec![0usize; k];
            for (i, v) in vectors.iter().enumerate() {
                let c = assignments[i];
                counts[c] += 1;
                for (j, val) in v.iter().enumerate() {
                    sums[c][j] += val;
                }
            }
            for c_idx in 0..k {
                if counts[c_idx] > 0 {
                    let n = counts[c_idx] as f32;
                    for j in 0..self.dims {
                        centroids[c_idx][j] = sums[c_idx][j] / n;
                    }
                }
                // If a centroid has no assignments, keep it unchanged.
            }
        }

        self.centroids = centroids;
        self.buckets = (0..k).map(|_| Vec::new()).collect();
        self.id_map.clear();
    }

    /// Insert a vector with the given external ID.
    ///
    /// The vector is assigned to the nearest centroid's bucket.
    /// If centroids haven't been trained yet, this will panic.
    pub fn insert(&mut self, external_id: u64, vector: Vec<f32>) {
        assert_eq!(vector.len(), self.dims, "vector dimension mismatch");
        assert!(
            !self.centroids.is_empty(),
            "index not trained — call train() first"
        );

        // Remove old entry if exists (for upsert behavior).
        self.remove(external_id);

        let bucket_idx = self.nearest_centroid(&vector);
        self.buckets[bucket_idx].push((external_id, vector));
        self.id_map.insert(external_id, bucket_idx);
    }

    /// Remove a vector by external ID. Returns true if found and removed.
    pub fn remove(&mut self, external_id: u64) -> bool {
        if let Some(bucket_idx) = self.id_map.remove(&external_id) {
            self.buckets[bucket_idx].retain(|(id, _)| *id != external_id);
            true
        } else {
            false
        }
    }

    /// Search for the `top_k` nearest neighbors to `query`.
    ///
    /// Probes `n_probe` nearest clusters (or the default if `n_probe` is 0).
    pub fn search(&self, query: &[f32], top_k: usize, n_probe: usize) -> Vec<SearchResult> {
        assert_eq!(query.len(), self.dims, "query dimension mismatch");

        if self.centroids.is_empty() || self.is_empty() {
            return Vec::new();
        }

        let probe = if n_probe == 0 {
            self.default_n_probe
        } else {
            n_probe
        };
        let probe = probe.min(self.n_list);

        // Find the `probe` nearest centroids.
        let mut centroid_dists: Vec<(usize, OrderedFloat<f32>)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, OrderedFloat(self.metric.distance(query, c))))
            .collect();

        // Sort: best (closest/most similar) first.
        if self.metric.is_similarity() {
            centroid_dists.sort_by(|a, b| b.1.cmp(&a.1));
        } else {
            centroid_dists.sort_by(|a, b| a.1.cmp(&b.1));
        }
        centroid_dists.truncate(probe);

        // Exhaustive search within the selected buckets.
        let mut candidates: Vec<(u64, OrderedFloat<f32>)> = Vec::new();
        for (bucket_idx, _) in &centroid_dists {
            for (ext_id, vec) in &self.buckets[*bucket_idx] {
                let d = self.metric.distance(query, vec);
                candidates.push((*ext_id, OrderedFloat(d)));
            }
        }

        // Sort by best score.
        if self.metric.is_similarity() {
            candidates.sort_by(|a, b| b.1.cmp(&a.1));
        } else {
            candidates.sort_by(|a, b| a.1.cmp(&b.1));
        }
        candidates.truncate(top_k);

        candidates
            .into_iter()
            .map(|(id, d)| SearchResult::new(id, d.into_inner()))
            .collect()
    }

    /// Total number of vectors in the index.
    pub fn len(&self) -> usize {
        self.id_map.len()
    }

    /// Whether the index contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.id_map.is_empty()
    }

    /// Find the nearest centroid for a vector, returning the bucket index.
    fn nearest_centroid(&self, vector: &[f32]) -> usize {
        let mut best_idx = 0;
        let mut best_dist = self.metric.worst_distance();
        for (i, c) in self.centroids.iter().enumerate() {
            let d = self.metric.distance(vector, c);
            if self.metric.is_better(d, best_dist) {
                best_dist = d;
                best_idx = i;
            }
        }
        best_idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vectors(n: usize, dims: usize) -> Vec<Vec<f32>> {
        let mut rng = rand::thread_rng();
        (0..n)
            .map(|_| (0..dims).map(|_| rng.gen::<f32>()).collect())
            .collect()
    }

    #[test]
    fn test_insert_and_search_basic() {
        let dims = 8;
        let mut index = IvfIndex::new(
            dims,
            MetricType::L2,
            IvfParams {
                n_list: 4,
                n_probe: 4,
                n_iters: 5,
            },
        );

        let vectors = make_vectors(50, dims);
        index.train(&vectors);

        for (i, v) in vectors.iter().enumerate() {
            index.insert(i as u64, v.clone());
        }

        assert_eq!(index.len(), 50);

        // Search for the first vector — it should be the top result.
        let results = index.search(&vectors[0], 5, 0);
        assert!(!results.is_empty());
        assert_eq!(results[0].id, 0);
        assert!(results[0].score < 1e-6); // L2 distance to itself should be ~0
    }

    #[test]
    fn test_search_quality() {
        // With n_probe == n_list (exhaustive), IVF should find exact neighbors.
        let dims = 4;
        let mut index = IvfIndex::new(
            dims,
            MetricType::L2,
            IvfParams {
                n_list: 3,
                n_probe: 3, // probe all clusters = exhaustive
                n_iters: 10,
            },
        );

        let vectors = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        index.train(&vectors);
        for (i, v) in vectors.iter().enumerate() {
            index.insert(i as u64, v.clone());
        }

        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 2, 0);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 0); // exact match
        assert_eq!(results[1].id, 1); // closest neighbor
    }

    #[test]
    fn test_remove() {
        let dims = 4;
        let mut index = IvfIndex::new(
            dims,
            MetricType::L2,
            IvfParams {
                n_list: 2,
                n_probe: 2,
                n_iters: 5,
            },
        );

        let vectors = make_vectors(20, dims);
        index.train(&vectors);
        for (i, v) in vectors.iter().enumerate() {
            index.insert(i as u64, v.clone());
        }

        assert_eq!(index.len(), 20);
        assert!(index.remove(5));
        assert_eq!(index.len(), 19);
        assert!(!index.remove(5)); // already removed

        // Search should not return removed ID.
        let results = index.search(&vectors[5], 20, 0);
        for r in &results {
            assert_ne!(r.id, 5);
        }
    }

    #[test]
    fn test_train_synthetic_clusters() {
        let dims = 2;
        let mut index = IvfIndex::new(
            dims,
            MetricType::L2,
            IvfParams {
                n_list: 3,
                n_probe: 1,
                n_iters: 20,
            },
        );

        // Create 3 well-separated clusters.
        let mut vectors = Vec::new();
        let mut rng = rand::thread_rng();
        // Cluster 0 near (0, 0)
        for _ in 0..30 {
            vectors.push(vec![rng.gen::<f32>() * 0.1, rng.gen::<f32>() * 0.1]);
        }
        // Cluster 1 near (10, 0)
        for _ in 0..30 {
            vectors.push(vec![
                10.0 + rng.gen::<f32>() * 0.1,
                rng.gen::<f32>() * 0.1,
            ]);
        }
        // Cluster 2 near (0, 10)
        for _ in 0..30 {
            vectors.push(vec![
                rng.gen::<f32>() * 0.1,
                10.0 + rng.gen::<f32>() * 0.1,
            ]);
        }

        index.train(&vectors);
        for (i, v) in vectors.iter().enumerate() {
            index.insert(i as u64, v.clone());
        }

        // Searching near cluster 0 with n_probe=1 should only find vectors from cluster 0.
        let results = index.search(&[0.05, 0.05], 5, 1);
        assert_eq!(results.len(), 5);
        for r in &results {
            assert!(r.id < 30, "expected cluster 0 vector, got id={}", r.id);
        }
    }

    #[test]
    fn test_empty_index() {
        let index = IvfIndex::new(4, MetricType::L2, IvfParams::default());
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 5, 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_ip_metric() {
        let dims = 4;
        let mut index = IvfIndex::new(
            dims,
            MetricType::IP,
            IvfParams {
                n_list: 2,
                n_probe: 2,
                n_iters: 5,
            },
        );

        let vectors = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.5, 0.5, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        index.train(&vectors);
        for (i, v) in vectors.iter().enumerate() {
            index.insert(i as u64, v.clone());
        }

        // Query [1, 0, 0, 0]: highest IP with vector 0.
        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 3, 0);
        assert_eq!(results[0].id, 0);
    }
}
