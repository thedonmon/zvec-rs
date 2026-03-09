//! High-level Collection API — the main entry point for zvec-rs.
//!
//! A Collection wraps an HNSW index with metadata storage, field management,
//! and filter-based search. Supports both in-memory and persistent modes.

use std::collections::HashMap;
use std::path::Path;
use std::sync::RwLock;

use crate::distance::MetricType;
use crate::filter::{self, parse_filter};
use crate::hnsw::{HnswIndex, HnswParams};
use crate::storage::Storage;

/// Configuration for creating a new collection.
#[derive(Debug, Clone)]
pub struct CollectionConfig {
    pub dims: usize,
    pub metric: MetricType,
    pub hnsw_params: HnswParams,
}

impl CollectionConfig {
    pub fn new(dims: usize) -> Self {
        Self {
            dims,
            metric: MetricType::IP,
            hnsw_params: HnswParams::default(),
        }
    }

    pub fn with_metric(mut self, metric: MetricType) -> Self {
        self.metric = metric;
        self
    }

    pub fn with_hnsw_params(mut self, params: HnswParams) -> Self {
        self.hnsw_params = params;
        self
    }
}

/// A search result with ID, score, and associated field values.
#[derive(Debug, Clone)]
pub struct SearchHit {
    pub pk: String,
    pub score: f32,
    pub fields: HashMap<String, String>,
}

/// A vector collection with HNSW index, metadata, and field storage.
///
/// Thread-safe: supports concurrent inserts and searches.
pub struct Collection {
    index: HnswIndex,
    /// pk (String) -> field values
    fields: RwLock<HashMap<String, HashMap<String, String>>>,
    /// Internal ID -> pk
    pk_map: RwLock<Vec<String>>,
    /// Persistent storage (None = in-memory only)
    storage: Option<RwLock<Storage>>,
    config: CollectionConfig,
}

impl Collection {
    /// Create a new in-memory collection (no persistence).
    pub fn new(config: CollectionConfig) -> Self {
        Self {
            index: HnswIndex::new(config.dims, config.metric, config.hnsw_params.clone()),
            fields: RwLock::new(HashMap::new()),
            pk_map: RwLock::new(Vec::new()),
            storage: None,
            config,
        }
    }

    /// Open or create a persistent collection at the given path.
    ///
    /// If the database file exists, loads all vectors and metadata from disk.
    /// Otherwise, creates a new empty collection.
    pub fn open(path: impl AsRef<Path>, name: &str, config: CollectionConfig) -> Result<Self, String> {
        let db_path = path.as_ref().join(format!("{}.redb", name));

        let storage = Storage::open(&db_path)
            .map_err(|e| format!("failed to open storage: {}", e))?;

        let mut collection = Self {
            index: HnswIndex::new(config.dims, config.metric, config.hnsw_params.clone()),
            fields: RwLock::new(HashMap::new()),
            pk_map: RwLock::new(Vec::new()),
            storage: Some(RwLock::new(storage)),
            config,
        };

        // Load existing data from disk
        collection.load_from_storage()?;

        Ok(collection)
    }

    /// Load all data from persistent storage into the in-memory index.
    fn load_from_storage(&mut self) -> Result<(), String> {
        let storage = self.storage.as_ref()
            .ok_or("no storage configured")?
            .read()
            .map_err(|e| format!("storage lock: {}", e))?;

        let id_map = storage.load_id_map()
            .map_err(|e| format!("load id_map: {}", e))?;

        if id_map.is_empty() {
            return Ok(());
        }

        // Sort by internal ID to reconstruct pk_map in order
        let mut entries: Vec<(String, u32)> = id_map.into_iter().collect();
        entries.sort_by_key(|(_, id)| *id);

        let mut pk_map = self.pk_map.write().unwrap();
        let mut fields_map = self.fields.write().unwrap();

        for (ext_id, internal_id) in &entries {
            // Ensure pk_map is filled up to this internal_id
            while pk_map.len() <= *internal_id as usize {
                pk_map.push(String::new());
            }
            pk_map[*internal_id as usize] = ext_id.clone();

            // Load vector
            let vector = storage.get_vector(*internal_id)
                .map_err(|e| format!("load vector {}: {}", internal_id, e))?
                .ok_or_else(|| format!("vector {} not found", internal_id))?;

            self.index.insert(*internal_id as u64, &vector);

            // Load metadata
            if let Some(meta) = storage.get_metadata(*internal_id)
                .map_err(|e| format!("load metadata {}: {}", internal_id, e))? {
                fields_map.insert(ext_id.clone(), meta);
            }
        }

        Ok(())
    }

    /// Insert or update a document.
    pub fn upsert(
        &self,
        pk: &str,
        vector: &[f32],
        fields: HashMap<String, String>,
    ) {
        let numeric_id = self.pk_to_id(pk);

        self.index.insert(numeric_id, vector);

        self.fields
            .write()
            .unwrap()
            .insert(pk.to_string(), fields.clone());

        // Persist to storage if configured
        if let Some(ref storage_lock) = self.storage {
            if let Ok(storage) = storage_lock.read() {
                // Get connections from the HNSW index for persistence
                // For now, we persist vector + metadata. Full HNSW graph
                // persistence would require exposing connection data.
                let _ = storage.put_vector(
                    numeric_id as u32,
                    pk,
                    vector,
                    &fields,
                    &[vec![]], // connections stored separately on flush
                );
            }
        }
    }

    /// Remove a document by primary key.
    pub fn remove(&self, pk: &str) -> bool {
        let numeric_id = {
            let pk_map = self.pk_map.read().unwrap();
            pk_map
                .iter()
                .position(|p| p == pk)
                .map(|i| i as u64)
        };

        match numeric_id {
            Some(id) => {
                self.index.remove(id);
                self.fields.write().unwrap().remove(pk);

                // Remove from persistent storage
                if let Some(ref storage_lock) = self.storage {
                    if let Ok(storage) = storage_lock.read() {
                        let _ = storage.remove_vector(pk);
                    }
                }

                true
            }
            None => false,
        }
    }

    /// Fetch a document's fields by primary key.
    pub fn fetch(&self, pk: &str) -> Option<HashMap<String, String>> {
        self.fields.read().unwrap().get(pk).cloned()
    }

    /// Search for nearest neighbors, optionally with a filter.
    pub fn search(
        &self,
        vector: &[f32],
        top_k: usize,
        filter_expr: Option<&str>,
    ) -> Result<Vec<SearchHit>, String> {
        let parsed_filter = match filter_expr {
            Some(expr) => Some(
                parse_filter(expr).map_err(|e| format!("filter parse error: {}", e))?,
            ),
            None => None,
        };

        // Over-fetch when filtering (we may discard some results)
        let fetch_k = if parsed_filter.is_some() {
            top_k * 4
        } else {
            top_k
        };

        let raw_results = self.index.search(vector, fetch_k);
        let fields_map = self.fields.read().unwrap();
        let pk_map = self.pk_map.read().unwrap();

        let mut hits: Vec<SearchHit> = raw_results
            .into_iter()
            .filter_map(|r| {
                let id = r.id as usize;
                if id >= pk_map.len() {
                    return None;
                }
                let pk = &pk_map[id];
                if pk.is_empty() {
                    return None;
                }
                let doc_fields = fields_map.get(pk)?;

                // Apply filter
                if let Some(ref filter) = parsed_filter {
                    if !filter::matches(filter, doc_fields) {
                        return None;
                    }
                }

                Some(SearchHit {
                    pk: pk.clone(),
                    score: r.score,
                    fields: doc_fields.clone(),
                })
            })
            .collect();

        hits.truncate(top_k);
        Ok(hits)
    }

    /// Flush all pending writes to disk.
    /// Returns Ok(true) if flushed, Ok(false) if no storage configured.
    pub fn flush(&self) -> Result<bool, String> {
        match self.storage {
            Some(ref storage_lock) => {
                let mut storage = storage_lock.write()
                    .map_err(|e| format!("storage lock: {}", e))?;

                // Save index state
                let next_id = self.pk_map.read().unwrap().len() as u32;
                let stats = self.index.stats();
                storage.save_state(0, stats.max_level, next_id)
                    .map_err(|e| format!("save state: {}", e))?;

                storage.flush()
                    .map_err(|e| format!("flush: {}", e))?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Get the number of documents in the collection.
    pub fn doc_count(&self) -> usize {
        self.index.len()
    }

    /// Get collection configuration.
    pub fn config(&self) -> &CollectionConfig {
        &self.config
    }

    /// Map a string pk to a numeric ID. Assigns new IDs as needed.
    fn pk_to_id(&self, pk: &str) -> u64 {
        let mut pk_map = self.pk_map.write().unwrap();

        // Check if already exists
        if let Some(pos) = pk_map.iter().position(|p| p == pk) {
            return pos as u64;
        }

        // Assign new ID
        let id = pk_map.len() as u64;
        pk_map.push(pk.to_string());
        id
    }
}

impl Drop for Collection {
    fn drop(&mut self) {
        // Auto-flush on drop
        let _ = self.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use std::sync::Arc;

    fn test_fields(category: &str, tenant: &str) -> HashMap<String, String> {
        let mut f = HashMap::new();
        f.insert("category".to_string(), category.to_string());
        f.insert("tenant".to_string(), tenant.to_string());
        f
    }

    #[test]
    fn test_basic_crud() {
        let col = Collection::new(CollectionConfig::new(3));

        col.upsert("doc-1", &[1.0, 0.0, 0.0], test_fields("a", "t1"));
        col.upsert("doc-2", &[0.0, 1.0, 0.0], test_fields("b", "t1"));

        assert_eq!(col.doc_count(), 2);

        let fetched = col.fetch("doc-1").unwrap();
        assert_eq!(fetched["category"], "a");

        assert!(col.remove("doc-1"));
        assert!(!col.remove("nonexistent"));
        assert!(col.fetch("doc-1").is_none());
    }

    #[test]
    fn test_search_no_filter() {
        let col = Collection::new(CollectionConfig::new(3).with_metric(MetricType::L2));

        col.upsert("doc-1", &[1.0, 0.0, 0.0], test_fields("a", "t1"));
        col.upsert("doc-2", &[0.0, 1.0, 0.0], test_fields("b", "t1"));
        col.upsert("doc-3", &[0.0, 0.0, 1.0], test_fields("c", "t1"));

        let results = col.search(&[1.0, 0.0, 0.0], 2, None).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].pk, "doc-1");
    }

    #[test]
    fn test_search_with_filter() {
        let dims = 16;
        let col = Collection::new(CollectionConfig::new(dims).with_metric(MetricType::L2));

        let mut rng = rand::thread_rng();
        for i in 0..50 {
            let v: Vec<f32> = (0..dims).map(|_| rng.gen()).collect();
            let cat = if i % 2 == 0 { "even" } else { "odd" };
            col.upsert(&format!("doc-{}", i), &v, test_fields(cat, "t1"));
        }

        let query: Vec<f32> = (0..dims).map(|_| rng.gen()).collect();
        let results = col
            .search(&query, 10, Some("category = 'even'"))
            .unwrap();

        for hit in &results {
            assert_eq!(hit.fields["category"], "even");
        }
    }

    #[test]
    fn test_search_with_compound_filter() {
        let dims = 8;
        let col = Collection::new(CollectionConfig::new(dims).with_metric(MetricType::L2));

        let mut rng = rand::thread_rng();
        for i in 0..100 {
            let v: Vec<f32> = (0..dims).map(|_| rng.gen()).collect();
            let cat = if i % 3 == 0 {
                "a"
            } else if i % 3 == 1 {
                "b"
            } else {
                "c"
            };
            let tenant = if i < 50 { "org-1" } else { "org-2" };
            col.upsert(&format!("doc-{}", i), &v, test_fields(cat, tenant));
        }

        let query: Vec<f32> = (0..dims).map(|_| rng.gen()).collect();
        let results = col
            .search(
                &query,
                5,
                Some("category = 'a' AND tenant = 'org-1'"),
            )
            .unwrap();

        for hit in &results {
            assert_eq!(hit.fields["category"], "a");
            assert_eq!(hit.fields["tenant"], "org-1");
        }
    }

    #[test]
    fn test_upsert_updates() {
        let col = Collection::new(CollectionConfig::new(3));

        col.upsert("doc-1", &[1.0, 0.0, 0.0], test_fields("old", "t1"));
        col.upsert("doc-1", &[0.0, 1.0, 0.0], test_fields("new", "t1"));

        let fields = col.fetch("doc-1").unwrap();
        assert_eq!(fields["category"], "new");
    }

    #[test]
    fn test_concurrent_collection() {
        let dims = 16;
        let col = Arc::new(Collection::new(
            CollectionConfig::new(dims).with_metric(MetricType::L2),
        ));

        let mut handles = vec![];

        for t in 0..4u64 {
            let c = Arc::clone(&col);
            handles.push(std::thread::spawn(move || {
                let mut rng = rand::thread_rng();
                for i in 0..50 {
                    let id = t * 50 + i;
                    let v: Vec<f32> = (0..dims).map(|_| rng.gen()).collect();
                    c.upsert(&format!("doc-{}", id), &v, test_fields("cat", "t1"));
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(col.doc_count(), 200);

        let mut search_handles = vec![];
        for _ in 0..4 {
            let c = Arc::clone(&col);
            search_handles.push(std::thread::spawn(move || {
                let mut rng = rand::thread_rng();
                for _ in 0..50 {
                    let q: Vec<f32> = (0..dims).map(|_| rng.gen()).collect();
                    let results = c.search(&q, 5, None).unwrap();
                    assert!(results.len() <= 5);
                }
            }));
        }

        for h in search_handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_bad_filter_returns_error() {
        let col = Collection::new(CollectionConfig::new(3));
        col.upsert("doc-1", &[1.0, 0.0, 0.0], HashMap::new());

        let result = col.search(&[1.0, 0.0, 0.0], 5, Some("= bad"));
        assert!(result.is_err());
    }

    #[test]
    fn test_persistent_collection() {
        let dir = tempfile::tempdir().unwrap();
        let dims = 4;
        let config = CollectionConfig::new(dims).with_metric(MetricType::L2);

        // Create and populate
        {
            let col = Collection::open(dir.path(), "test", config.clone()).unwrap();
            col.upsert("doc-1", &[1.0, 0.0, 0.0, 0.0], test_fields("a", "t1"));
            col.upsert("doc-2", &[0.0, 1.0, 0.0, 0.0], test_fields("b", "t1"));
            col.upsert("doc-3", &[0.0, 0.0, 1.0, 0.0], test_fields("c", "t2"));
            col.flush().unwrap();
        }

        // Reopen and verify data survived
        {
            let col = Collection::open(dir.path(), "test", config.clone()).unwrap();
            assert_eq!(col.doc_count(), 3);

            let fetched = col.fetch("doc-1").unwrap();
            assert_eq!(fetched["category"], "a");

            let fetched = col.fetch("doc-3").unwrap();
            assert_eq!(fetched["tenant"], "t2");

            // Search should work on reloaded data
            let results = col.search(&[1.0, 0.0, 0.0, 0.0], 2, None).unwrap();
            assert!(!results.is_empty());
            assert_eq!(results[0].pk, "doc-1");
        }
    }

    #[test]
    fn test_persistent_remove() {
        let dir = tempfile::tempdir().unwrap();
        let dims = 3;
        let config = CollectionConfig::new(dims).with_metric(MetricType::L2);

        {
            let col = Collection::open(dir.path(), "test", config.clone()).unwrap();
            col.upsert("doc-1", &[1.0, 0.0, 0.0], test_fields("a", "t1"));
            col.upsert("doc-2", &[0.0, 1.0, 0.0], test_fields("b", "t1"));
            col.remove("doc-1");
            col.flush().unwrap();
        }

        {
            let col = Collection::open(dir.path(), "test", config).unwrap();
            assert_eq!(col.doc_count(), 1);
            assert!(col.fetch("doc-1").is_none());
            assert!(col.fetch("doc-2").is_some());
        }
    }

    #[test]
    fn test_flush_in_memory() {
        let col = Collection::new(CollectionConfig::new(3));
        // Flush on in-memory collection should return Ok(false)
        assert_eq!(col.flush().unwrap(), false);
    }
}
