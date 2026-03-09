//! IVF (Inverted File) index with flat/exhaustive search within clusters.
//!
//! Partitions vectors into clusters using k-means, then searches only the
//! nearest clusters for approximate nearest neighbor queries.

use std::collections::HashMap;

use ordered_float::OrderedFloat;
use rand::Rng;

use crate::distance::MetricType;
use crate::hnsw::SearchResult;
use crate::quantize::PqCodebook;

/// Parameters for IVF index construction and search.
#[derive(Debug, Clone)]
pub struct IvfParams {
    /// Number of clusters (Voronoi cells). Default: 100.
    pub n_list: usize,
    /// Number of clusters to probe during search. Default: 10.
    pub n_probe: usize,
    /// Number of k-means iterations during training. Default: 10.
    pub n_iters: usize,
    /// Whether to use Product Quantization for compressed residual storage.
    pub use_pq: bool,
    /// Number of sub-quantizers for PQ. dims must be divisible by this. Default: 8.
    pub pq_m: usize,
    /// Number of bits per PQ code (only 8 is supported). Default: 8.
    pub pq_bits: usize,
}

impl Default for IvfParams {
    fn default() -> Self {
        Self {
            n_list: 100,
            n_probe: 10,
            n_iters: 10,
            use_pq: false,
            pq_m: 8,
            pq_bits: 8,
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

// ---------------------------------------------------------------------------
// IVF-PQ: IVF with Product Quantization for compressed residual storage
// ---------------------------------------------------------------------------

use crate::quantize::PqCode;

/// IVF index with Product Quantization for compressed storage.
///
/// Vectors are assigned to the nearest centroid. The residual (vector - centroid)
/// is encoded using PQ. At search time, Asymmetric Distance Computation (ADC)
/// is used on the residuals for fast approximate distance calculation.
pub struct IvfPqIndex {
    /// Centroid vectors, one per cluster.
    centroids: Vec<Vec<f32>>,
    /// Per-cluster storage: bucket[i] = [(external_id, pq_code), ...]
    buckets: Vec<Vec<(u64, PqCode)>>,
    /// PQ codebook trained on residual vectors.
    codebook: Option<PqCodebook>,
    _metric: MetricType,
    dims: usize,
    n_list: usize,
    /// Reverse map: external_id -> bucket index.
    id_map: HashMap<u64, usize>,
    /// Default n_probe for searches.
    default_n_probe: usize,
    /// K-means iterations for training.
    n_iters: usize,
    /// Number of PQ sub-quantizers.
    pq_m: usize,
}

impl IvfPqIndex {
    /// Create a new empty IVF-PQ index.
    ///
    /// Centroids and PQ codebook are not initialized until `train()` is called.
    pub fn new(dims: usize, metric: MetricType, params: IvfParams) -> Self {
        assert!(
            dims % params.pq_m == 0,
            "dims ({}) must be divisible by pq_m ({})",
            dims,
            params.pq_m
        );
        Self {
            centroids: Vec::new(),
            buckets: Vec::new(),
            codebook: None,
            _metric: metric,
            dims,
            n_list: params.n_list,
            id_map: HashMap::new(),
            default_n_probe: params.n_probe,
            n_iters: params.n_iters,
            pq_m: params.pq_m,
        }
    }

    /// Train the index: run k-means to find centroids, then train PQ on residuals.
    ///
    /// # Panics
    /// Panics if `vectors` is empty or any vector has wrong dimensionality.
    pub fn train(&mut self, vectors: &[Vec<f32>]) {
        assert!(!vectors.is_empty(), "cannot train on empty vector set");
        for v in vectors {
            assert_eq!(v.len(), self.dims, "vector dimension mismatch");
        }

        let k = self.n_list.min(vectors.len());
        self.n_list = k;

        // --- Step 1: K-means clustering for centroids ---
        let mut rng = rand::thread_rng();
        let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
        let mut used = std::collections::HashSet::new();
        while centroids.len() < k {
            let idx = rng.gen_range(0..vectors.len());
            if used.insert(idx) {
                centroids.push(vectors[idx].clone());
            }
        }

        let mut assignments = vec![0usize; vectors.len()];

        for _ in 0..self.n_iters {
            // Assign each vector to nearest centroid.
            for (i, v) in vectors.iter().enumerate() {
                let mut best_idx = 0;
                let mut best_dist = f32::INFINITY;
                for (c_idx, c) in centroids.iter().enumerate() {
                    let d = l2_distance(v, c);
                    if d < best_dist {
                        best_dist = d;
                        best_idx = c_idx;
                    }
                }
                assignments[i] = best_idx;
            }

            // Recompute centroids.
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
            }
        }

        // Final assignment pass.
        for (i, v) in vectors.iter().enumerate() {
            let mut best_idx = 0;
            let mut best_dist = f32::INFINITY;
            for (c_idx, c) in centroids.iter().enumerate() {
                let d = l2_distance(v, c);
                if d < best_dist {
                    best_dist = d;
                    best_idx = c_idx;
                }
            }
            assignments[i] = best_idx;
        }

        // --- Step 2: Compute residuals and train PQ ---
        let residuals: Vec<Vec<f32>> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let c = &centroids[assignments[i]];
                v.iter().zip(c.iter()).map(|(a, b)| a - b).collect()
            })
            .collect();

        let residual_refs: Vec<&[f32]> = residuals.iter().map(|v| v.as_slice()).collect();
        let pq_k = 256.min(residuals.len()); // max 256 centroids for 8-bit codes
        let codebook = PqCodebook::train(&residual_refs, self.dims, self.pq_m, pq_k, self.n_iters);

        self.centroids = centroids;
        self.codebook = Some(codebook);
        self.buckets = (0..k).map(|_| Vec::new()).collect();
        self.id_map.clear();
    }

    /// Insert a vector with the given external ID.
    ///
    /// The vector is assigned to the nearest centroid, and the residual is PQ-encoded.
    pub fn insert(&mut self, external_id: u64, vector: Vec<f32>) {
        assert_eq!(vector.len(), self.dims, "vector dimension mismatch");
        assert!(
            self.codebook.is_some(),
            "index not trained — call train() first"
        );

        self.remove(external_id);

        let bucket_idx = self.nearest_centroid(&vector);
        let residual: Vec<f32> = vector
            .iter()
            .zip(self.centroids[bucket_idx].iter())
            .map(|(a, b)| a - b)
            .collect();

        let code = self.codebook.as_ref().unwrap().encode(&residual);
        self.buckets[bucket_idx].push((external_id, code));
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
    /// Uses Asymmetric Distance Computation (ADC) on PQ-encoded residuals.
    pub fn search(&self, query: &[f32], top_k: usize, n_probe: usize) -> Vec<SearchResult> {
        assert_eq!(query.len(), self.dims, "query dimension mismatch");

        if self.centroids.is_empty() || self.is_empty() || self.codebook.is_none() {
            return Vec::new();
        }

        let codebook = self.codebook.as_ref().unwrap();
        let probe = if n_probe == 0 {
            self.default_n_probe
        } else {
            n_probe
        };
        let probe = probe.min(self.n_list);

        // Find the `probe` nearest centroids (always use L2 for centroid assignment).
        let mut centroid_dists: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, l2_distance(query, c)))
            .collect();
        centroid_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        centroid_dists.truncate(probe);

        // For each probed cluster, compute residual query and use ADC.
        let mut candidates: Vec<(u64, f32)> = Vec::new();

        for (bucket_idx, _) in &centroid_dists {
            // Residual of query with respect to this centroid.
            let query_residual: Vec<f32> = query
                .iter()
                .zip(self.centroids[*bucket_idx].iter())
                .map(|(a, b)| a - b)
                .collect();

            // Build distance table once per centroid.
            let table = codebook.build_distance_table(&query_residual);

            for (ext_id, code) in &self.buckets[*bucket_idx] {
                let dist = codebook.distance_with_table(&table, code);
                candidates.push((*ext_id, dist));
            }
        }

        // Sort by L2 distance (lower = better).
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(top_k);

        candidates
            .into_iter()
            .map(|(id, d)| SearchResult::new(id, d))
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

    /// Find the nearest centroid for a vector (using L2), returning the bucket index.
    fn nearest_centroid(&self, vector: &[f32]) -> usize {
        let mut best_idx = 0;
        let mut best_dist = f32::INFINITY;
        for (i, c) in self.centroids.iter().enumerate() {
            let d = l2_distance(vector, c);
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        best_idx
    }
}

/// Squared L2 distance helper (used internally for centroid assignment in IVF-PQ).
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum()
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
                ..Default::default()
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
                ..Default::default()
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
                ..Default::default()
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
                ..Default::default()
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
                ..Default::default()
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

    // -----------------------------------------------------------------------
    // IVF-PQ tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ivfpq_insert_and_search() {
        let dims = 8;
        let mut index = IvfPqIndex::new(
            dims,
            MetricType::L2,
            IvfParams {
                n_list: 4,
                n_probe: 4,
                n_iters: 10,
                use_pq: true,
                pq_m: 4,
                pq_bits: 8,
            },
        );

        let vectors = make_vectors(60, dims);
        index.train(&vectors);

        for (i, v) in vectors.iter().enumerate() {
            index.insert(i as u64, v.clone());
        }

        assert_eq!(index.len(), 60);

        // Search for the first vector — it should be among top results.
        let results = index.search(&vectors[0], 5, 0);
        assert!(!results.is_empty());
        // With PQ compression, the exact vector might not be the top-1,
        // but it should be in the top results.
        let top_ids: Vec<u64> = results.iter().map(|r| r.id).collect();
        assert!(
            top_ids.contains(&0),
            "expected ID 0 in top-5, got {:?}",
            top_ids
        );
    }

    #[test]
    fn test_ivfpq_search_quality_exhaustive() {
        // With n_probe == n_list and good PQ codebook, should find correct neighbors.
        let dims = 8;
        let mut index = IvfPqIndex::new(
            dims,
            MetricType::L2,
            IvfParams {
                n_list: 3,
                n_probe: 3,
                n_iters: 20,
                use_pq: true,
                pq_m: 4,
                pq_bits: 8,
            },
        );

        // Well-separated vectors.
        let vectors = vec![
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ];
        index.train(&vectors);
        for (i, v) in vectors.iter().enumerate() {
            index.insert(i as u64, v.clone());
        }

        let results = index.search(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2, 0);
        assert_eq!(results.len(), 2);
        // Top result should be ID 0 (exact match) or ID 1 (very close).
        assert!(
            results[0].id == 0 || results[0].id == 1,
            "expected ID 0 or 1 as top result, got {}",
            results[0].id
        );
    }

    #[test]
    fn test_ivfpq_remove() {
        let dims = 8;
        let mut index = IvfPqIndex::new(
            dims,
            MetricType::L2,
            IvfParams {
                n_list: 2,
                n_probe: 2,
                n_iters: 10,
                use_pq: true,
                pq_m: 4,
                pq_bits: 8,
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
        assert!(!index.remove(5));

        let results = index.search(&vectors[5], 20, 0);
        for r in &results {
            assert_ne!(r.id, 5);
        }
    }

    #[test]
    fn test_ivfpq_empty_index() {
        let index = IvfPqIndex::new(
            8,
            MetricType::L2,
            IvfParams {
                n_list: 4,
                n_probe: 2,
                n_iters: 5,
                use_pq: true,
                pq_m: 4,
                pq_bits: 8,
            },
        );
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
        let results = index.search(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 5, 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_ivfpq_upsert() {
        let dims = 8;
        let mut index = IvfPqIndex::new(
            dims,
            MetricType::L2,
            IvfParams {
                n_list: 2,
                n_probe: 2,
                n_iters: 10,
                use_pq: true,
                pq_m: 4,
                pq_bits: 8,
            },
        );

        let vectors = make_vectors(10, dims);
        index.train(&vectors);
        for (i, v) in vectors.iter().enumerate() {
            index.insert(i as u64, v.clone());
        }
        assert_eq!(index.len(), 10);

        // Re-insert ID 0 with a different vector (upsert).
        index.insert(0, vec![99.0; dims]);
        assert_eq!(index.len(), 10); // count should not change
    }

    #[test]
    fn test_ivfpq_synthetic_clusters() {
        let dims = 8;
        let mut index = IvfPqIndex::new(
            dims,
            MetricType::L2,
            IvfParams {
                n_list: 3,
                n_probe: 1,
                n_iters: 20,
                use_pq: true,
                pq_m: 4,
                pq_bits: 8,
            },
        );

        let mut vectors = Vec::new();
        let mut rng = rand::thread_rng();

        // Cluster 0 near origin.
        for _ in 0..30 {
            let v: Vec<f32> = (0..dims).map(|_| rng.gen::<f32>() * 0.1).collect();
            vectors.push(v);
        }
        // Cluster 1 near (10, 10, ...).
        for _ in 0..30 {
            let v: Vec<f32> = (0..dims).map(|_| 10.0 + rng.gen::<f32>() * 0.1).collect();
            vectors.push(v);
        }
        // Cluster 2 near (-10, -10, ...).
        for _ in 0..30 {
            let v: Vec<f32> = (0..dims).map(|_| -10.0 + rng.gen::<f32>() * 0.1).collect();
            vectors.push(v);
        }

        index.train(&vectors);
        for (i, v) in vectors.iter().enumerate() {
            index.insert(i as u64, v.clone());
        }

        // Search near cluster 0.
        let query: Vec<f32> = vec![0.05; dims];
        let results = index.search(&query, 5, 1);
        assert_eq!(results.len(), 5);
        for r in &results {
            assert!(r.id < 30, "expected cluster 0 vector, got id={}", r.id);
        }
    }

    #[test]
    fn test_ivf_params_backward_compat() {
        // Default params should work the same as before.
        let params = IvfParams::default();
        assert_eq!(params.n_list, 100);
        assert_eq!(params.n_probe, 10);
        assert_eq!(params.n_iters, 10);
        assert!(!params.use_pq);
        assert_eq!(params.pq_m, 8);
        assert_eq!(params.pq_bits, 8);
    }
}
