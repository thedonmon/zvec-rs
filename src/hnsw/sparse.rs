//! Sparse HNSW index — HNSW graph over sparse vectors.
//!
//! Uses the same layered graph structure and HNSW algorithm as the dense index
//! (`graph.rs`) but stores `SparseVector` and dispatches to sparse distance
//! kernels (dot, L2, cosine) instead of dense SIMD routines.

use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use rand::Rng;

use super::params::HnswParams;
use super::search::{Candidate, FarCandidate, SearchResult};
use crate::sparse::SparseVector;

// ---------------------------------------------------------------------------
// Metric
// ---------------------------------------------------------------------------

/// Distance metric for sparse vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseMetric {
    /// Inner product (dot product). Higher = more similar.
    IP,
    /// L2 squared distance. Lower = more similar.
    L2,
    /// Cosine similarity. Higher = more similar.
    Cosine,
}

impl SparseMetric {
    #[inline]
    fn is_similarity(self) -> bool {
        matches!(self, SparseMetric::IP | SparseMetric::Cosine)
    }

    #[inline]
    fn is_better(self, a: f32, b: f32) -> bool {
        if self.is_similarity() {
            a > b
        } else {
            a < b
        }
    }

    #[inline]
    fn distance(self, a: &SparseVector, b: &SparseVector) -> f32 {
        match self {
            SparseMetric::IP => SparseVector::dot(a, b),
            SparseMetric::L2 => SparseVector::l2_squared(a, b),
            SparseMetric::Cosine => SparseVector::cosine(a, b),
        }
    }
}

// ---------------------------------------------------------------------------
// Node / NodeStore  (mirrors graph.rs but with SparseVector)
// ---------------------------------------------------------------------------

struct SparseNode {
    vector: SparseVector,
    connections: Vec<RwLock<Vec<u32>>>,
    deleted: std::sync::atomic::AtomicBool,
}

impl SparseNode {
    fn new(vector: SparseVector, levels: usize) -> Self {
        let mut connections = Vec::with_capacity(levels + 1);
        for _ in 0..=levels {
            connections.push(RwLock::new(Vec::new()));
        }
        Self {
            vector,
            connections,
            deleted: std::sync::atomic::AtomicBool::new(false),
        }
    }

    #[inline]
    fn is_deleted(&self) -> bool {
        self.deleted.load(Ordering::Acquire)
    }
}

struct SparseNodeStore {
    nodes: RwLock<Vec<Box<SparseNode>>>,
}

impl SparseNodeStore {
    fn new() -> Self {
        Self {
            nodes: RwLock::new(Vec::new()),
        }
    }

    fn push(&self, node: SparseNode) -> u32 {
        let mut nodes = self.nodes.write();
        let id = nodes.len() as u32;
        nodes.push(Box::new(node));
        id
    }

    fn with_nodes<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&[Box<SparseNode>]) -> R,
    {
        let nodes = self.nodes.read();
        f(&nodes)
    }
}

// ---------------------------------------------------------------------------
// Bloom filter (identical to graph.rs)
// ---------------------------------------------------------------------------

struct BloomFilter {
    bits: Vec<u64>,
    n_bits: usize,
}

impl BloomFilter {
    fn new(expected: usize) -> Self {
        let n_bits = (expected * 10).max(64);
        let n_words = (n_bits + 63) / 64;
        Self {
            bits: vec![0u64; n_words],
            n_bits,
        }
    }

    fn insert(&mut self, value: u32) -> bool {
        let h1 = (value as usize).wrapping_mul(0x9E3779B97F4A7C15_u64 as usize) % self.n_bits;
        let h2 = (value as usize).wrapping_mul(0x517CC1B727220A95_u64 as usize) % self.n_bits;
        let was_set = self.get_bit(h1) && self.get_bit(h2);
        self.set_bit(h1);
        self.set_bit(h2);
        was_set
    }

    fn get_bit(&self, pos: usize) -> bool {
        self.bits[pos / 64] & (1u64 << (pos % 64)) != 0
    }

    fn set_bit(&mut self, pos: usize) {
        self.bits[pos / 64] |= 1u64 << (pos % 64);
    }
}

// ---------------------------------------------------------------------------
// SparseHnswIndex
// ---------------------------------------------------------------------------

const NO_ENTRY: u32 = u32::MAX;

/// HNSW index for sparse vectors.
///
/// Mirrors the dense `HnswIndex` API but operates on `SparseVector` values
/// and uses sparse distance kernels.
pub struct SparseHnswIndex {
    params: HnswParams,
    metric: SparseMetric,
    nodes: SparseNodeStore,
    entry_point: AtomicU32,
    max_level: AtomicUsize,
    id_map: RwLock<std::collections::HashMap<u64, u32>>,
    reverse_id_map: RwLock<Vec<u64>>,
}

unsafe impl Send for SparseHnswIndex {}
unsafe impl Sync for SparseHnswIndex {}

impl SparseHnswIndex {
    /// Create a new empty sparse HNSW index.
    pub fn new(metric: SparseMetric) -> Self {
        Self::with_params(metric, HnswParams::default())
    }

    /// Create with explicit HNSW parameters.
    pub fn with_params(metric: SparseMetric, params: HnswParams) -> Self {
        Self {
            params,
            metric,
            nodes: SparseNodeStore::new(),
            entry_point: AtomicU32::new(NO_ENTRY),
            max_level: AtomicUsize::new(0),
            id_map: RwLock::new(std::collections::HashMap::new()),
            reverse_id_map: RwLock::new(Vec::new()),
        }
    }

    /// Number of active (non-deleted) vectors.
    pub fn len(&self) -> usize {
        self.id_map.read().len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.id_map.read().is_empty()
    }

    /// Insert a sparse vector with the given external ID.
    /// Upsert semantics: if the ID already exists the old entry is replaced.
    pub fn insert(&self, external_id: u64, vector: &SparseVector) {
        // Upsert: soft-delete old entry if exists
        {
            let id_map = self.id_map.read();
            if id_map.contains_key(&external_id) {
                drop(id_map);
                self.remove(external_id);
            }
        }

        let level = self.random_level();
        let node = SparseNode::new(vector.clone(), level);
        let internal_id = self.nodes.push(node);

        // Register in ID maps
        {
            self.id_map.write().insert(external_id, internal_id);
            self.reverse_id_map.write().push(external_id);
        }

        // First node — set as entry point and return
        if self
            .entry_point
            .compare_exchange(NO_ENTRY, internal_id, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            self.max_level.store(level, Ordering::Release);
            return;
        }

        let current_max = self.max_level.load(Ordering::Acquire);

        let mut current_ep = self.entry_point.load(Ordering::Acquire);
        self.nodes.with_nodes(|nodes| {
            // Phase 1: Greedy descent from top to level+1
            if current_max > level {
                for lc in (level + 1..=current_max).rev() {
                    current_ep = self.search_layer_one(nodes, vector, current_ep, lc);
                }
            }

            // Phase 2: Insert at each layer from `level` down to 0
            for lc in (0..=std::cmp::min(level, current_max)).rev() {
                let max_conn = if lc == 0 {
                    self.params.m_max0
                } else {
                    self.params.m
                };

                let neighbors =
                    self.search_layer(nodes, vector, current_ep, self.params.ef_construction, lc);

                let selected: Vec<u32> = if self.params.use_heuristic {
                    self.select_neighbors_heuristic(
                        nodes,
                        vector,
                        &neighbors,
                        max_conn,
                        self.params.extend_candidates,
                    )
                } else {
                    neighbors
                        .iter()
                        .take(max_conn)
                        .map(|c| c.id)
                        .collect()
                };

                // Set forward connections
                {
                    let mut conns = nodes[internal_id as usize].connections[lc].write();
                    *conns = selected.clone();
                }

                // Set reverse connections + prune
                for &neighbor_id in &selected {
                    let mut neighbor_conns =
                        nodes[neighbor_id as usize].connections[lc].write();
                    neighbor_conns.push(internal_id);

                    if neighbor_conns.len() > max_conn {
                        if self.params.use_heuristic {
                            let neighbor_vec = &nodes[neighbor_id as usize].vector;
                            let mut candidates: Vec<Candidate> = neighbor_conns
                                .iter()
                                .filter(|&&id| (id as usize) < nodes.len())
                                .map(|&id| {
                                    let dist = self
                                        .metric
                                        .distance(neighbor_vec, &nodes[id as usize].vector);
                                    Candidate::new(dist, id)
                                })
                                .collect();
                            if self.metric.is_similarity() {
                                candidates.sort_by(|a, b| b.distance.cmp(&a.distance));
                            } else {
                                candidates.sort_by(|a, b| a.distance.cmp(&b.distance));
                            }
                            *neighbor_conns = self.select_neighbors_heuristic(
                                nodes,
                                neighbor_vec,
                                &candidates,
                                max_conn,
                                false,
                            );
                        } else {
                            let neighbor_vec = &nodes[neighbor_id as usize].vector;
                            let mut scored: Vec<(OrderedFloat<f32>, u32)> = neighbor_conns
                                .iter()
                                .filter(|&&id| (id as usize) < nodes.len())
                                .map(|&id| {
                                    let dist = self
                                        .metric
                                        .distance(neighbor_vec, &nodes[id as usize].vector);
                                    (OrderedFloat(dist), id)
                                })
                                .collect();
                            if self.metric.is_similarity() {
                                scored.sort_by(|a, b| b.0.cmp(&a.0));
                            } else {
                                scored.sort_by(|a, b| a.0.cmp(&b.0));
                            }
                            scored.truncate(max_conn);
                            *neighbor_conns = scored.into_iter().map(|(_, id)| id).collect();
                        }
                    }
                }

                if !selected.is_empty() {
                    current_ep = selected[0];
                }
            }
        });

        // Update entry point + max level if this node is higher
        if level > current_max {
            let mut old = current_max;
            while level > old {
                match self.max_level.compare_exchange_weak(
                    old,
                    level,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        self.entry_point.store(internal_id, Ordering::Release);
                        break;
                    }
                    Err(actual) => old = actual,
                }
            }
        }
    }

    /// Remove a vector by external ID (soft delete).
    pub fn remove(&self, external_id: u64) -> bool {
        let internal_id = match self.id_map.write().remove(&external_id) {
            Some(id) => id,
            None => return false,
        };

        self.nodes.with_nodes(|nodes| {
            let node = &nodes[internal_id as usize];
            node.deleted.store(true, Ordering::Release);

            for lc in 0..node.connections.len() {
                let neighbors: Vec<u32> = node.connections[lc].read().clone();
                for &neighbor_id in &neighbors {
                    if (neighbor_id as usize) < nodes.len() {
                        let mut conns = nodes[neighbor_id as usize].connections[lc].write();
                        conns.retain(|&id| id != internal_id);
                    }
                }
                node.connections[lc].write().clear();
            }
        });

        if self.entry_point.load(Ordering::Acquire) == internal_id {
            let new_ep = self
                .id_map
                .read()
                .values()
                .next()
                .copied()
                .unwrap_or(NO_ENTRY);
            self.entry_point.store(new_ep, Ordering::Release);
        }

        true
    }

    /// Search for the top-k nearest neighbors.
    pub fn search(&self, query: &SparseVector, top_k: usize) -> Vec<SearchResult> {
        let entry_id = self.entry_point.load(Ordering::Acquire);
        if entry_id == NO_ENTRY {
            return Vec::new();
        }

        self.nodes.with_nodes(|nodes| {
            let max_level = self.max_level.load(Ordering::Acquire);

            let mut current_ep = entry_id;
            for lc in (1..=max_level).rev() {
                current_ep = self.search_layer_one(nodes, query, current_ep, lc);
            }

            let ef = std::cmp::max(self.params.ef_search, top_k);
            let candidates = self.search_layer(nodes, query, current_ep, ef, 0);

            let reverse_map = self.reverse_id_map.read();
            candidates
                .into_iter()
                .filter(|c| !nodes[c.id as usize].is_deleted())
                .take(top_k)
                .map(|c| {
                    let ext_id = reverse_map[c.id as usize];
                    SearchResult::new(ext_id, c.distance.into_inner())
                })
                .collect()
        })
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn search_layer_one(
        &self,
        nodes: &[Box<SparseNode>],
        query: &SparseVector,
        entry: u32,
        level: usize,
    ) -> u32 {
        let mut current = entry;
        let mut current_dist = self.metric.distance(query, &nodes[current as usize].vector);

        loop {
            let mut changed = false;
            let node = &nodes[current as usize];
            if level >= node.connections.len() {
                break;
            }
            let conns = node.connections[level].read();
            for &neighbor_id in conns.iter() {
                if (neighbor_id as usize) >= nodes.len() {
                    continue;
                }
                let dist = self
                    .metric
                    .distance(query, &nodes[neighbor_id as usize].vector);
                if self.metric.is_better(dist, current_dist) {
                    current_dist = dist;
                    current = neighbor_id;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        current
    }

    fn search_layer(
        &self,
        nodes: &[Box<SparseNode>],
        query: &SparseVector,
        entry: u32,
        ef: usize,
        level: usize,
    ) -> Vec<Candidate> {
        let entry_dist = self.metric.distance(query, &nodes[entry as usize].vector);

        let mut candidates = BinaryHeap::new();
        candidates.push(Candidate::new(entry_dist, entry));

        let mut results = BinaryHeap::new();
        results.push(FarCandidate::new(entry_dist, entry));

        let use_bloom = nodes.len() >= 1000;
        let mut bloom = if use_bloom {
            let mut bf = BloomFilter::new(ef * 4);
            bf.insert(entry);
            Some(bf)
        } else {
            None
        };
        let mut visited_set = if use_bloom {
            None
        } else {
            let mut hs = std::collections::HashSet::new();
            hs.insert(entry);
            Some(hs)
        };

        while let Some(closest) = candidates.pop() {
            let worst = results.peek().unwrap();
            if !self
                .metric
                .is_better(closest.distance.into_inner(), worst.distance.into_inner())
                && results.len() >= ef
            {
                break;
            }

            let conns: Vec<u32> = {
                let node = &nodes[closest.id as usize];
                if level < node.connections.len() {
                    node.connections[level].read().clone()
                } else {
                    continue;
                }
            };

            for neighbor_id in conns {
                let already_visited = if let Some(ref mut bf) = bloom {
                    bf.insert(neighbor_id)
                } else {
                    !visited_set.as_mut().unwrap().insert(neighbor_id)
                };

                if already_visited {
                    continue;
                }

                if (neighbor_id as usize) >= nodes.len() {
                    continue;
                }

                let dist = self
                    .metric
                    .distance(query, &nodes[neighbor_id as usize].vector);

                let worst = results.peek().unwrap();
                if results.len() < ef
                    || self.metric.is_better(dist, worst.distance.into_inner())
                {
                    candidates.push(Candidate::new(dist, neighbor_id));
                    results.push(FarCandidate::new(dist, neighbor_id));
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        let mut result_vec: Vec<Candidate> = results
            .into_iter()
            .map(|fc| Candidate::new(fc.distance.into_inner(), fc.id))
            .collect();

        if self.metric.is_similarity() {
            result_vec.sort_by(|a, b| b.distance.cmp(&a.distance));
        } else {
            result_vec.sort_by(|a, b| a.distance.cmp(&b.distance));
        }

        result_vec
    }

    fn select_neighbors_heuristic(
        &self,
        nodes: &[Box<SparseNode>],
        query_vector: &SparseVector,
        candidates: &[Candidate],
        max_conn: usize,
        extend_candidates: bool,
    ) -> Vec<u32> {
        let mut working = Vec::with_capacity(candidates.len() * 2);
        let mut seen = std::collections::HashSet::with_capacity(candidates.len() * 2);

        for c in candidates {
            if seen.insert(c.id) {
                working.push(*c);
            }
        }

        if extend_candidates {
            for c in candidates {
                let node = &nodes[c.id as usize];
                if !node.connections.is_empty() {
                    let conns = node.connections[0].read();
                    for &neighbor_id in conns.iter() {
                        if (neighbor_id as usize) < nodes.len() && seen.insert(neighbor_id) {
                            let dist = self
                                .metric
                                .distance(query_vector, &nodes[neighbor_id as usize].vector);
                            working.push(Candidate::new(dist, neighbor_id));
                        }
                    }
                }
            }
        }

        if self.metric.is_similarity() {
            working.sort_by(|a, b| b.distance.cmp(&a.distance));
        } else {
            working.sort_by(|a, b| a.distance.cmp(&b.distance));
        }

        let mut selected: Vec<u32> = Vec::with_capacity(max_conn);

        for candidate in &working {
            if selected.len() >= max_conn {
                break;
            }

            let dist_to_query = candidate.distance.into_inner();
            let candidate_vec = &nodes[candidate.id as usize].vector;

            let is_diverse = selected.iter().all(|&sel_id| {
                let dist_to_selected =
                    self.metric.distance(candidate_vec, &nodes[sel_id as usize].vector);
                self.metric.is_better(dist_to_query, dist_to_selected)
            });

            if is_diverse {
                selected.push(candidate.id);
            }
        }

        if selected.len() < max_conn {
            let selected_set: std::collections::HashSet<u32> =
                selected.iter().copied().collect();
            for candidate in &working {
                if selected.len() >= max_conn {
                    break;
                }
                if !selected_set.contains(&candidate.id) {
                    selected.push(candidate.id);
                }
            }
        }

        selected
    }

    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen();
        (-r.ln() * self.params.ml).floor() as usize
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sparse(indices: Vec<u32>, values: Vec<f32>) -> SparseVector {
        SparseVector::new(indices, values).unwrap()
    }

    // --- basic CRUD ---

    #[test]
    fn test_insert_and_search_ip() {
        let index = SparseHnswIndex::new(SparseMetric::IP);

        let v1 = make_sparse(vec![0, 1], vec![1.0, 0.0]);
        let v2 = make_sparse(vec![0, 1], vec![0.0, 1.0]);
        let v3 = make_sparse(vec![0, 1], vec![0.7, 0.7]);

        index.insert(1, &v1);
        index.insert(2, &v2);
        index.insert(3, &v3);

        assert_eq!(index.len(), 3);

        let results = index.search(&v1, 2);
        assert_eq!(results.len(), 2);
        // v1 dot v1 = 1.0, v1 dot v3 = 0.7, v1 dot v2 = 0.0
        assert_eq!(results[0].id, 1);
    }

    #[test]
    fn test_insert_and_search_l2() {
        let index = SparseHnswIndex::new(SparseMetric::L2);

        let v1 = make_sparse(vec![0, 2, 4], vec![1.0, 2.0, 3.0]);
        let v2 = make_sparse(vec![1, 2, 4], vec![10.0, 20.0, 30.0]);
        let v3 = make_sparse(vec![0, 2, 4], vec![1.1, 2.1, 3.1]);

        index.insert(1, &v1);
        index.insert(2, &v2);
        index.insert(3, &v3);

        let results = index.search(&v1, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 1); // exact match, distance 0
        assert_eq!(results[1].id, 3); // very close
    }

    #[test]
    fn test_insert_and_search_cosine() {
        let index = SparseHnswIndex::new(SparseMetric::Cosine);

        let v1 = make_sparse(vec![0], vec![1.0]);
        let v2 = make_sparse(vec![1], vec![1.0]);
        let v3 = make_sparse(vec![0, 1], vec![0.9, 0.1]);

        index.insert(1, &v1);
        index.insert(2, &v2);
        index.insert(3, &v3);

        let results = index.search(&v1, 3);
        assert_eq!(results[0].id, 1); // cosine = 1.0
    }

    #[test]
    fn test_empty_search() {
        let index = SparseHnswIndex::new(SparseMetric::IP);
        let q = make_sparse(vec![0], vec![1.0]);
        let results = index.search(&q, 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_remove() {
        let index = SparseHnswIndex::new(SparseMetric::IP);

        index.insert(1, &make_sparse(vec![0], vec![1.0]));
        index.insert(2, &make_sparse(vec![1], vec![1.0]));
        index.insert(3, &make_sparse(vec![2], vec![1.0]));

        assert!(index.remove(2));
        assert!(!index.remove(99));
        assert_eq!(index.len(), 2);

        let results = index.search(&make_sparse(vec![1], vec![1.0]), 3);
        assert!(results.iter().all(|r| r.id != 2));
    }

    #[test]
    fn test_upsert() {
        let index = SparseHnswIndex::new(SparseMetric::L2);

        let v1 = make_sparse(vec![0], vec![1.0]);
        let v2 = make_sparse(vec![0], vec![5.0]);

        index.insert(1, &v1);
        assert_eq!(index.len(), 1);

        // Re-insert with same ID should replace
        index.insert(1, &v2);
        assert_eq!(index.len(), 1);

        // Searching for v2 should return id 1
        let results = index.search(&v2, 1);
        assert_eq!(results[0].id, 1);
        assert!(results[0].score.abs() < 1e-6); // L2 distance to itself = 0
    }

    // --- many vectors ---

    #[test]
    fn test_insert_many() {
        let index = SparseHnswIndex::with_params(
            SparseMetric::L2,
            HnswParams::new(16, 100).with_ef_search(50),
        );

        let mut rng = rand::thread_rng();
        let max_dim = 100u32;
        for i in 0..200u64 {
            // Random sparse vector with ~10 non-zero entries
            let nnz = rng.gen_range(3..15);
            let mut indices: Vec<u32> = (0..nnz).map(|_| rng.gen_range(0..max_dim)).collect();
            indices.sort();
            indices.dedup();
            let values: Vec<f32> = indices.iter().map(|_| rng.gen::<f32>()).collect();
            let sv = SparseVector::new(indices, values).unwrap();
            index.insert(i, &sv);
        }

        assert_eq!(index.len(), 200);

        let query = make_sparse(vec![0, 10, 50], vec![1.0, 0.5, 0.3]);
        let results = index.search(&query, 10);
        assert_eq!(results.len(), 10);

        // L2: results should be non-decreasing distance
        for w in results.windows(2) {
            assert!(w[0].score <= w[1].score + 1e-6);
        }
    }

    // --- concurrency ---

    #[test]
    fn test_concurrent_insert_and_search() {
        use std::sync::Arc;

        let index = Arc::new(SparseHnswIndex::with_params(
            SparseMetric::IP,
            HnswParams::new(16, 100).with_ef_search(50),
        ));

        let mut handles = vec![];
        for t in 0..4u64 {
            let idx = Arc::clone(&index);
            handles.push(std::thread::spawn(move || {
                let mut rng = rand::thread_rng();
                for i in 0..50 {
                    let id = t * 50 + i;
                    let nnz = rng.gen_range(2..8);
                    let mut indices: Vec<u32> =
                        (0..nnz).map(|_| rng.gen_range(0..50u32)).collect();
                    indices.sort();
                    indices.dedup();
                    let values: Vec<f32> = indices.iter().map(|_| rng.gen::<f32>()).collect();
                    let sv = SparseVector::new(indices, values).unwrap();
                    idx.insert(id, &sv);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(index.len(), 200);

        // Concurrent search
        let mut search_handles = vec![];
        for _ in 0..4 {
            let idx = Arc::clone(&index);
            search_handles.push(std::thread::spawn(move || {
                let mut rng = rand::thread_rng();
                for _ in 0..50 {
                    let q = make_sparse(
                        vec![0, 10, 20],
                        vec![rng.gen(), rng.gen(), rng.gen()],
                    );
                    let results = idx.search(&q, 5);
                    assert!(results.len() <= 5);
                }
            }));
        }

        for h in search_handles {
            h.join().unwrap();
        }
    }

    // --- recall quality ---

    #[test]
    fn test_recall_quality_l2() {
        let n = 500;
        let index = SparseHnswIndex::with_params(
            SparseMetric::L2,
            HnswParams::new(32, 200).with_ef_search(100),
        );

        let mut rng = rand::thread_rng();
        let max_dim = 50u32;

        let mut vectors: Vec<SparseVector> = Vec::with_capacity(n);
        for i in 0..n {
            let nnz = rng.gen_range(5..15);
            let mut indices: Vec<u32> = (0..nnz).map(|_| rng.gen_range(0..max_dim)).collect();
            indices.sort();
            indices.dedup();
            let values: Vec<f32> = indices.iter().map(|_| rng.gen::<f32>()).collect();
            let sv = SparseVector::new(indices, values).unwrap();
            index.insert(i as u64, &sv);
            vectors.push(sv);
        }

        let k = 10;
        let mut total_recall = 0.0;
        let num_queries = 20;

        for _ in 0..num_queries {
            let nnz = rng.gen_range(3..10);
            let mut indices: Vec<u32> = (0..nnz).map(|_| rng.gen_range(0..max_dim)).collect();
            indices.sort();
            indices.dedup();
            let values: Vec<f32> = indices.iter().map(|_| rng.gen::<f32>()).collect();
            let query = SparseVector::new(indices, values).unwrap();

            // Brute-force ground truth
            let mut dists: Vec<(f32, u64)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| (SparseVector::l2_squared(&query, v), i as u64))
                .collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let gt: std::collections::HashSet<u64> =
                dists.iter().take(k).map(|(_, id)| *id).collect();

            let results = index.search(&query, k);
            let found: std::collections::HashSet<u64> =
                results.iter().map(|r| r.id).collect();

            let recall = gt.intersection(&found).count() as f64 / k as f64;
            total_recall += recall;
        }

        let avg_recall = total_recall / num_queries as f64;
        assert!(
            avg_recall > 0.80,
            "Average recall@{k} = {avg_recall:.3}, expected > 0.80"
        );
    }

    #[test]
    fn test_single_element() {
        let index = SparseHnswIndex::new(SparseMetric::IP);
        let v = make_sparse(vec![0, 5, 10], vec![1.0, 2.0, 3.0]);
        index.insert(42, &v);

        let results = index.search(&v, 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 42);
    }

    #[test]
    fn test_is_empty() {
        let index = SparseHnswIndex::new(SparseMetric::L2);
        assert!(index.is_empty());
        index.insert(1, &make_sparse(vec![0], vec![1.0]));
        assert!(!index.is_empty());
    }

    #[test]
    fn test_remove_all() {
        let index = SparseHnswIndex::new(SparseMetric::IP);
        index.insert(1, &make_sparse(vec![0], vec![1.0]));
        index.insert(2, &make_sparse(vec![1], vec![1.0]));

        index.remove(1);
        index.remove(2);

        assert_eq!(index.len(), 0);
        assert!(index.is_empty());

        let results = index.search(&make_sparse(vec![0], vec![1.0]), 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_high_dimensional_sparse() {
        // Sparse vectors with very high max dimension but few non-zeros
        let index = SparseHnswIndex::new(SparseMetric::IP);

        let v1 = make_sparse(vec![0, 10000, 50000], vec![1.0, 2.0, 3.0]);
        let v2 = make_sparse(vec![0, 10000, 50000], vec![0.9, 1.9, 2.9]);
        let v3 = make_sparse(vec![1, 20000, 99999], vec![1.0, 1.0, 1.0]);

        index.insert(1, &v1);
        index.insert(2, &v2);
        index.insert(3, &v3);

        let results = index.search(&v1, 2);
        assert_eq!(results[0].id, 1);
        assert_eq!(results[1].id, 2);
    }
}
