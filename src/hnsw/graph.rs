use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use rand::Rng;

use super::params::HnswParams;
use super::search::{Candidate, FarCandidate, SearchResult};
use crate::distance::MetricType;

/// Simple bloom filter for tracking visited nodes during search.
/// Uses two hash functions for low false positive rate.
struct BloomFilter {
    bits: Vec<u64>,
    n_bits: usize,
}

impl BloomFilter {
    /// Create a bloom filter sized for `expected` elements with ~1% false positive rate.
    fn new(expected: usize) -> Self {
        let n_bits = (expected * 10).max(64); // ~10 bits per element for ~1% FPR
        let n_words = (n_bits + 63) / 64;
        Self {
            bits: vec![0u64; n_words],
            n_bits,
        }
    }

    /// Insert an element. Returns true if it was already (possibly) present.
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

/// A node in the HNSW graph.
///
/// Each node owns its vector and has per-layer connection lists protected
/// by individual RwLocks for fine-grained concurrent access.
struct Node {
    /// Vector data (immutable after construction).
    vector: Vec<f32>,
    /// Per-layer connection lists, each independently locked.
    /// connections[0] = layer 0 (densest), connections[n] = top layer.
    connections: Vec<RwLock<Vec<u32>>>,
    /// Soft-delete flag.
    deleted: std::sync::atomic::AtomicBool,
}

impl Node {
    fn new(vector: Vec<f32>, levels: usize) -> Self {
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

    #[inline]
    #[allow(dead_code)]
    fn max_level(&self) -> usize {
        self.connections.len().saturating_sub(1)
    }
}

/// Thread-safe node storage. Append-only (nodes are boxed so pointers
/// are stable even as the vec grows). Protected by a single RwLock but
/// reads only need a brief lock to get the node reference -- the actual
/// data access is lock-free through the Box indirection.
struct NodeStore {
    /// Boxed nodes for pointer stability. The outer RwLock is held
    /// briefly during reads (to get a reference) and during appends.
    nodes: RwLock<Vec<Box<Node>>>,
}

impl NodeStore {
    fn new() -> Self {
        Self {
            nodes: RwLock::new(Vec::new()),
        }
    }

    /// Append a new node and return its internal ID.
    fn push(&self, node: Node) -> u32 {
        let mut nodes = self.nodes.write();
        let id = nodes.len() as u32;
        nodes.push(Box::new(node));
        id
    }

    /// Get the total number of nodes (including deleted).
    fn total_count(&self) -> usize {
        self.nodes.read().len()
    }

    /// Execute a closure with read access to the node slice.
    /// The closure receives the full slice of boxed nodes.
    ///
    /// SAFETY: The RwLock is held for the duration of the closure.
    /// Nodes are boxed so their addresses are stable even if the vec
    /// is reallocated by a concurrent push (which would need the write lock).
    fn with_nodes<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&[Box<Node>]) -> R,
    {
        let nodes = self.nodes.read();
        f(&nodes)
    }
}

/// Concurrent HNSW index for approximate nearest neighbor search.
///
/// Design for concurrency:
/// - **Reads (search)**: Lock-free after acquiring a brief read lock on the node store.
///   Connection lists use per-node RwLocks so searches never block each other.
/// - **Writes (insert/remove)**: Fine-grained per-node locking. Multiple inserts
///   can proceed concurrently as long as they touch different nodes.
/// - **Entry point & max level**: Atomics -- no locking needed.
/// - **ID mapping**: RwLock-protected HashMap. Brief lock on each access.
///
/// This design allows high-throughput concurrent search with minimal contention,
/// and concurrent inserts that only serialize on shared neighbor nodes.
pub struct HnswIndex {
    params: HnswParams,
    metric: MetricType,
    dims: usize,
    nodes: NodeStore,
    /// Entry point node ID. u32::MAX = no entry point.
    entry_point: AtomicU32,
    max_level: AtomicUsize,
    /// External ID -> internal node index.
    id_map: RwLock<std::collections::HashMap<u64, u32>>,
    /// Internal node index -> external ID.
    reverse_id_map: RwLock<Vec<u64>>,
}

// SAFETY: All fields use proper synchronization.
unsafe impl Send for HnswIndex {}
unsafe impl Sync for HnswIndex {}

const NO_ENTRY: u32 = u32::MAX;

impl HnswIndex {
    /// Create a new empty HNSW index.
    pub fn new(dims: usize, metric: MetricType, params: HnswParams) -> Self {
        Self {
            params,
            metric,
            dims,
            nodes: NodeStore::new(),
            entry_point: AtomicU32::new(NO_ENTRY),
            max_level: AtomicUsize::new(0),
            id_map: RwLock::new(std::collections::HashMap::new()),
            reverse_id_map: RwLock::new(Vec::new()),
        }
    }

    /// Number of active (non-deleted) vectors in the index.
    pub fn len(&self) -> usize {
        self.id_map.read().len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.id_map.read().is_empty()
    }

    /// Insert a vector with the given external ID.
    /// Thread-safe: multiple inserts can proceed concurrently.
    ///
    /// If the ID already exists, the old vector is soft-deleted and
    /// a new one is inserted (upsert semantics).
    pub fn insert(&self, external_id: u64, vector: &[f32]) {
        assert_eq!(vector.len(), self.dims, "vector dimension mismatch");

        // Upsert: soft-delete old entry if exists
        {
            let id_map = self.id_map.read();
            if id_map.contains_key(&external_id) {
                drop(id_map);
                self.remove(external_id);
            }
        }

        let level = self.random_level();
        let node = Node::new(vector.to_vec(), level);
        let internal_id = self.nodes.push(node);

        // Register in ID maps
        {
            self.id_map.write().insert(external_id, internal_id);
            self.reverse_id_map.write().push(external_id);
        }

        // If this is the first node, set as entry point and return
        if self
            .entry_point
            .compare_exchange(NO_ENTRY, internal_id, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            self.max_level.store(level, Ordering::Release);
            return;
        }

        let current_max = self.max_level.load(Ordering::Acquire);

        // Phase 1: Greedy descent from top to level+1
        let mut current_ep = self.entry_point.load(Ordering::Acquire);
        self.nodes.with_nodes(|nodes| {
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

                let neighbors = self.search_layer(nodes, vector, current_ep, self.params.ef_construction, lc);

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

                // Set forward connections (our node -> neighbors)
                {
                    let mut conns = nodes[internal_id as usize].connections[lc].write();
                    *conns = selected.clone();
                }

                // Set reverse connections (each neighbor -> our node) + prune
                for &neighbor_id in &selected {
                    let mut neighbor_conns =
                        nodes[neighbor_id as usize].connections[lc].write();
                    neighbor_conns.push(internal_id);

                    if neighbor_conns.len() > max_conn {
                        if self.params.use_heuristic {
                            // Build candidate list from current connections
                            let neighbor_vec = &nodes[neighbor_id as usize].vector;
                            let mut candidates: Vec<Candidate> = neighbor_conns
                                .iter()
                                .filter(|&&id| (id as usize) < nodes.len())
                                .map(|&id| {
                                    let dist = self.metric.distance(
                                        neighbor_vec,
                                        &nodes[id as usize].vector,
                                    );
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
                            // Simple pruning: keep the closest `max_conn` neighbors
                            let neighbor_vec = &nodes[neighbor_id as usize].vector;
                            let mut scored: Vec<(OrderedFloat<f32>, u32)> = neighbor_conns
                                .iter()
                                .filter(|&&id| (id as usize) < nodes.len())
                                .map(|&id| {
                                    let dist = self.metric.distance(
                                        neighbor_vec,
                                        &nodes[id as usize].vector,
                                    );
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
            // CAS loop to update max_level
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
    /// Thread-safe: the node is marked deleted atomically and its
    /// connections are cleaned up with per-node locks.
    pub fn remove(&self, external_id: u64) -> bool {
        let internal_id = match self.id_map.write().remove(&external_id) {
            Some(id) => id,
            None => return false,
        };

        self.nodes.with_nodes(|nodes| {
            let node = &nodes[internal_id as usize];
            node.deleted.store(true, Ordering::Release);

            // Remove from neighbor connection lists
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

        // Update entry point if needed
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

    /// Search for the top-k nearest neighbors to the query vector.
    /// Thread-safe: multiple searches can run concurrently without blocking.
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<SearchResult> {
        assert_eq!(query.len(), self.dims, "query dimension mismatch");

        let entry_id = self.entry_point.load(Ordering::Acquire);
        if entry_id == NO_ENTRY {
            return Vec::new();
        }

        self.nodes.with_nodes(|nodes| {
            let max_level = self.max_level.load(Ordering::Acquire);

            // Phase 1: Greedy descent from top to layer 1
            let mut current_ep = entry_id;
            for lc in (1..=max_level).rev() {
                current_ep = self.search_layer_one(nodes, query, current_ep, lc);
            }

            // Phase 2: Search layer 0 with ef_search
            let ef = std::cmp::max(self.params.ef_search, top_k);
            let candidates = self.search_layer(nodes, query, current_ep, ef, 0);

            // Convert to results, skipping deleted nodes
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

    /// Greedy best-first search at a single layer, returning the nearest node.
    fn search_layer_one(
        &self,
        nodes: &[Box<Node>],
        query: &[f32],
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
                let dist = self.metric.distance(query, &nodes[neighbor_id as usize].vector);
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

    /// Search a single layer with ef candidates, returning sorted results.
    /// Uses a bloom filter for the visited set on large graphs (>= 1000 nodes)
    /// for faster lookups, falling back to HashSet for small graphs.
    fn search_layer(
        &self,
        nodes: &[Box<Node>],
        query: &[f32],
        entry: u32,
        ef: usize,
        level: usize,
    ) -> Vec<Candidate> {
        let entry_dist = self.metric.distance(query, &nodes[entry as usize].vector);

        // Min-heap: closest candidates to explore
        let mut candidates = BinaryHeap::new();
        candidates.push(Candidate::new(entry_dist, entry));

        // Max-heap: result set (farthest at top for pruning)
        let mut results = BinaryHeap::new();
        results.push(FarCandidate::new(entry_dist, entry));

        let use_bloom = nodes.len() >= 1000;

        // Visited tracking: bloom filter for large graphs, HashSet for small
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

            // Read neighbor connections (brief per-node read lock)
            let conns: Vec<u32> = {
                let node = &nodes[closest.id as usize];
                if level < node.connections.len() {
                    node.connections[level].read().clone()
                } else {
                    continue;
                }
            };

            for neighbor_id in conns {
                // Check visited using either bloom filter or hash set
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

                let dist = self.metric.distance(query, &nodes[neighbor_id as usize].vector);

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

    /// Select neighbors using the heuristic from the HNSW paper (Algorithm 4).
    /// Prefers diverse neighbors -- a candidate is only selected if it's closer
    /// to the query than to any already-selected neighbor. This creates more
    /// diverse connections and improves recall.
    fn select_neighbors_heuristic(
        &self,
        nodes: &[Box<Node>],
        query_vector: &[f32],
        candidates: &[Candidate],
        max_conn: usize,
        extend_candidates: bool,
    ) -> Vec<u32> {
        // Build the working set from candidates
        let mut working = Vec::with_capacity(candidates.len() * 2);
        let mut seen = std::collections::HashSet::with_capacity(candidates.len() * 2);

        for c in candidates {
            if seen.insert(c.id) {
                working.push(*c);
            }
        }

        // Optionally extend by adding neighbors of candidates
        if extend_candidates {
            for c in candidates {
                let node = &nodes[c.id as usize];
                // Only look at layer 0 connections for extension
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

        // Sort by distance (best first)
        if self.metric.is_similarity() {
            working.sort_by(|a, b| b.distance.cmp(&a.distance));
        } else {
            working.sort_by(|a, b| a.distance.cmp(&b.distance));
        }

        // Greedy selection: pick candidate if it's closer to query than to
        // any already-selected neighbor
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
                // Candidate is diverse if it's closer to query than to this selected neighbor
                self.metric.is_better(dist_to_query, dist_to_selected)
            });

            if is_diverse {
                selected.push(candidate.id);
            }
        }

        // If heuristic didn't fill up, add remaining candidates by distance
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

    /// Get the vector for an external ID.
    pub fn get_vector(&self, external_id: u64) -> Option<Vec<f32>> {
        let id_map = self.id_map.read();
        let &internal_id = id_map.get(&external_id)?;
        self.nodes.with_nodes(|nodes| {
            Some(nodes[internal_id as usize].vector.clone())
        })
    }

    /// Get the current entry point, or None if the index is empty.
    pub fn entry_point(&self) -> Option<u32> {
        let ep = self.entry_point.load(Ordering::Acquire);
        if ep == NO_ENTRY {
            None
        } else {
            Some(ep)
        }
    }

    /// Get the current maximum level.
    pub fn max_level(&self) -> usize {
        self.max_level.load(Ordering::Acquire)
    }

    /// Export connection lists for a single node (all layers).
    /// Returns None if the internal_id is out of range.
    pub fn get_connections(&self, internal_id: u32) -> Option<Vec<Vec<u32>>> {
        self.nodes.with_nodes(|nodes| {
            let idx = internal_id as usize;
            if idx >= nodes.len() {
                return None;
            }
            let node = &nodes[idx];
            if node.is_deleted() {
                return None;
            }
            let conns: Vec<Vec<u32>> = node
                .connections
                .iter()
                .map(|c| c.read().clone())
                .collect();
            Some(conns)
        })
    }

    /// Restore a node with pre-built connections (for loading from storage).
    ///
    /// This bypasses the normal insert path -- no neighbor search is performed.
    /// The caller is responsible for ensuring connections are consistent.
    /// Nodes must be restored in internal_id order (0, 1, 2, ...).
    pub fn restore_node(
        &self,
        external_id: u64,
        vector: &[f32],
        connections: Vec<Vec<u32>>,
    ) -> u32 {
        assert_eq!(vector.len(), self.dims, "vector dimension mismatch");

        let level = connections.len().saturating_sub(1);
        let node = Node::new(vector.to_vec(), level);

        // Overwrite connection lists with stored data
        for (i, conns) in connections.into_iter().enumerate() {
            if i < node.connections.len() {
                *node.connections[i].write() = conns;
            }
        }

        let internal_id = self.nodes.push(node);

        // Register in ID maps
        self.id_map.write().insert(external_id, internal_id);
        self.reverse_id_map.write().push(external_id);

        internal_id
    }

    /// Set the entry point and max level directly (for loading from storage).
    pub fn set_entry_point(&self, entry_point: u32, max_level: usize) {
        self.entry_point.store(entry_point, Ordering::Release);
        self.max_level.store(max_level, Ordering::Release);
    }

    /// Get index statistics.
    pub fn stats(&self) -> HnswStats {
        let total = self.nodes.total_count();
        let active = self.id_map.read().len();
        HnswStats {
            total_nodes: total,
            active_nodes: active,
            deleted_nodes: total - active,
            max_level: self.max_level.load(Ordering::Relaxed),
            dims: self.dims,
        }
    }
}

/// Statistics about the HNSW index.
#[derive(Debug, Clone)]
pub struct HnswStats {
    pub total_nodes: usize,
    pub active_nodes: usize,
    pub deleted_nodes: usize,
    pub max_level: usize,
    pub dims: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::MetricType;
    use crate::hnsw::HnswParams;
    use std::sync::Arc;

    fn make_index(dims: usize) -> HnswIndex {
        HnswIndex::new(dims, MetricType::L2, HnswParams::new(16, 100))
    }

    #[test]
    fn test_insert_and_search() {
        let index = make_index(3);
        index.insert(1, &[1.0, 0.0, 0.0]);
        index.insert(2, &[0.0, 1.0, 0.0]);
        index.insert(3, &[0.0, 0.0, 1.0]);
        index.insert(4, &[1.0, 1.0, 0.0]);

        assert_eq!(index.len(), 4);

        let results = index.search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 1);
    }

    #[test]
    fn test_insert_many() {
        let dims = 32;
        let index = make_index(dims);
        let mut rng = rand::thread_rng();

        for i in 0..100u64 {
            let v: Vec<f32> = (0..dims).map(|_| rng.gen()).collect();
            index.insert(i, &v);
        }

        assert_eq!(index.len(), 100);

        let query: Vec<f32> = (0..dims).map(|_| rng.gen()).collect();
        let results = index.search(&query, 10);
        assert_eq!(results.len(), 10);

        for w in results.windows(2) {
            assert!(w[0].score <= w[1].score);
        }
    }

    #[test]
    fn test_remove() {
        let index = make_index(3);
        index.insert(1, &[1.0, 0.0, 0.0]);
        index.insert(2, &[0.0, 1.0, 0.0]);
        index.insert(3, &[0.0, 0.0, 1.0]);

        assert!(index.remove(2));
        assert!(!index.remove(99));

        let results = index.search(&[0.0, 1.0, 0.0], 3);
        assert!(results.iter().all(|r| r.id != 2));
    }

    #[test]
    fn test_upsert() {
        let index = make_index(3);
        index.insert(1, &[1.0, 0.0, 0.0]);
        index.insert(1, &[0.0, 1.0, 0.0]);

        let v = index.get_vector(1).unwrap();
        assert_eq!(v, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_empty_search() {
        let index = make_index(3);
        let results = index.search(&[1.0, 0.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_inner_product_metric() {
        let index = HnswIndex::new(3, MetricType::IP, HnswParams::new(16, 100));
        index.insert(1, &[1.0, 0.0, 0.0]);
        index.insert(2, &[0.0, 1.0, 0.0]);
        index.insert(3, &[0.7, 0.7, 0.0]);

        let results = index.search(&[1.0, 0.0, 0.0], 3);
        assert_eq!(results[0].id, 1);
        assert!((results[0].score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_concurrent_insert_and_search() {
        let dims = 32;
        let index = Arc::new(HnswIndex::new(
            dims,
            MetricType::L2,
            HnswParams::new(16, 100).with_ef_search(50),
        ));

        // Insert 500 vectors from 4 threads concurrently
        let mut handles = vec![];
        for t in 0..4u64 {
            let idx = Arc::clone(&index);
            handles.push(std::thread::spawn(move || {
                let mut rng = rand::thread_rng();
                for i in 0..125 {
                    let id = t * 125 + i;
                    let v: Vec<f32> = (0..dims).map(|_| rng.gen()).collect();
                    idx.insert(id, &v);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(index.len(), 500);

        // Concurrent search from 4 threads
        let mut search_handles = vec![];
        for _ in 0..4 {
            let idx = Arc::clone(&index);
            search_handles.push(std::thread::spawn(move || {
                let mut rng = rand::thread_rng();
                for _ in 0..50 {
                    let query: Vec<f32> = (0..dims).map(|_| rng.gen()).collect();
                    let results = idx.search(&query, 10);
                    assert_eq!(results.len(), 10);
                }
            }));
        }

        for h in search_handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_concurrent_insert_and_search_mixed() {
        // Readers and writers running simultaneously
        let dims = 16;
        let index = Arc::new(HnswIndex::new(
            dims,
            MetricType::L2,
            HnswParams::new(8, 50).with_ef_search(30),
        ));

        // Seed with some initial data
        let mut rng = rand::thread_rng();
        for i in 0..50u64 {
            let v: Vec<f32> = (0..dims).map(|_| rng.gen()).collect();
            index.insert(i, &v);
        }

        let barrier = Arc::new(std::sync::Barrier::new(6));
        let mut handles = vec![];

        // 2 writer threads
        for t in 0..2u64 {
            let idx = Arc::clone(&index);
            let bar = Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || {
                bar.wait();
                let mut rng = rand::thread_rng();
                for i in 0..100 {
                    let id = 1000 + t * 100 + i;
                    let v: Vec<f32> = (0..dims).map(|_| rng.gen()).collect();
                    idx.insert(id, &v);
                }
            }));
        }

        // 4 reader threads
        for _ in 0..4 {
            let idx = Arc::clone(&index);
            let bar = Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || {
                bar.wait();
                let mut rng = rand::thread_rng();
                for _ in 0..200 {
                    let query: Vec<f32> = (0..dims).map(|_| rng.gen()).collect();
                    let _results = idx.search(&query, 5);
                    // Don't assert count -- concurrent inserts may change it
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // All 250 inserts (50 seed + 200 concurrent) should be present
        assert_eq!(index.len(), 250);
    }

    #[test]
    fn test_recall_quality() {
        let dims = 64;
        let n = 1000;
        let index = HnswIndex::new(
            dims,
            MetricType::L2,
            HnswParams::new(32, 200).with_ef_search(100),
        );
        let mut rng = rand::thread_rng();

        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(n);
        for i in 0..n {
            let v: Vec<f32> = (0..dims).map(|_| rng.gen()).collect();
            index.insert(i as u64, &v);
            vectors.push(v);
        }

        let k = 10;
        let mut total_recall = 0.0;
        let num_queries = 20;

        for _ in 0..num_queries {
            let query: Vec<f32> = (0..dims).map(|_| rng.gen()).collect();

            let mut dists: Vec<(f32, u64)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| (crate::distance::l2_squared(&query, v), i as u64))
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
            avg_recall > 0.90,
            "Average recall@{k} = {avg_recall:.3}, expected > 0.90"
        );
    }

    #[test]
    fn test_heuristic_selection_improves_recall() {
        let dims = 64;
        let n = 1000;
        let k = 10;
        let num_queries = 30;
        let mut rng = rand::thread_rng();

        // Generate shared dataset
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dims).map(|_| rng.gen::<f32>()).collect())
            .collect();

        let queries: Vec<Vec<f32>> = (0..num_queries)
            .map(|_| (0..dims).map(|_| rng.gen::<f32>()).collect())
            .collect();

        // Compute ground truth
        let ground_truths: Vec<std::collections::HashSet<u64>> = queries
            .iter()
            .map(|query| {
                let mut dists: Vec<(f32, u64)> = vectors
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (crate::distance::l2_squared(query, v), i as u64))
                    .collect();
                dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                dists.iter().take(k).map(|(_, id)| *id).collect()
            })
            .collect();

        // Build index WITHOUT heuristic
        let index_simple = HnswIndex::new(
            dims,
            MetricType::L2,
            HnswParams::new(32, 200)
                .with_ef_search(100)
                .with_heuristic(false),
        );
        for (i, v) in vectors.iter().enumerate() {
            index_simple.insert(i as u64, v);
        }

        // Build index WITH heuristic
        let index_heuristic = HnswIndex::new(
            dims,
            MetricType::L2,
            HnswParams::new(32, 200)
                .with_ef_search(100)
                .with_heuristic(true),
        );
        for (i, v) in vectors.iter().enumerate() {
            index_heuristic.insert(i as u64, v);
        }

        // Measure recall for both
        let mut recall_simple = 0.0;
        let mut recall_heuristic = 0.0;

        for (qi, query) in queries.iter().enumerate() {
            let gt = &ground_truths[qi];

            let results_simple = index_simple.search(query, k);
            let found_simple: std::collections::HashSet<u64> =
                results_simple.iter().map(|r| r.id).collect();
            recall_simple += gt.intersection(&found_simple).count() as f64 / k as f64;

            let results_heuristic = index_heuristic.search(query, k);
            let found_heuristic: std::collections::HashSet<u64> =
                results_heuristic.iter().map(|r| r.id).collect();
            recall_heuristic += gt.intersection(&found_heuristic).count() as f64 / k as f64;
        }

        recall_simple /= num_queries as f64;
        recall_heuristic /= num_queries as f64;

        // Heuristic should have equal or better recall
        assert!(
            recall_heuristic >= recall_simple - 0.05,
            "Heuristic recall ({recall_heuristic:.3}) should be >= simple recall ({recall_simple:.3}) - 0.05"
        );
    }

    #[test]
    fn test_bloom_filter_basic() {
        let mut bf = BloomFilter::new(100);

        // Fresh filter should not contain anything
        assert!(!bf.insert(42)); // not present before
        assert!(bf.insert(42)); // now present

        assert!(!bf.insert(99));
        assert!(bf.insert(99));

        // Re-check 42 still present
        assert!(bf.insert(42));
    }

    #[test]
    fn test_bloom_filter_false_positive_rate() {
        let n = 10_000;
        let mut bf = BloomFilter::new(n);

        // Insert items 0..n
        for i in 0..n as u32 {
            bf.insert(i);
        }

        // Check items n..2n (none were inserted)
        let mut false_positives = 0;
        let m = 10_000;
        for i in n as u32..(n + m) as u32 {
            if bf.insert(i) {
                false_positives += 1;
            }
        }

        let fpr = false_positives as f64 / m as f64;
        assert!(
            fpr < 0.05,
            "False positive rate {fpr:.4} exceeds 5% threshold"
        );
    }

    #[test]
    fn test_stats() {
        let index = make_index(3);
        index.insert(1, &[1.0, 0.0, 0.0]);
        index.insert(2, &[0.0, 1.0, 0.0]);
        index.remove(1);

        let stats = index.stats();
        assert_eq!(stats.active_nodes, 1);
        assert_eq!(stats.deleted_nodes, 1);
        assert_eq!(stats.total_nodes, 2);
        assert_eq!(stats.dims, 3);
    }
}
