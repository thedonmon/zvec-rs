//! High-level Collection API — the main entry point for zvec-rs.
//!
//! A Collection wraps an HNSW index with metadata storage, field management,
//! inverted indexes for filtered fields, and filter-based search.
//! Supports both in-memory and persistent modes.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::RwLock;

use crate::distance::MetricType;
use crate::filter::{self, parse_filter, FilterExpr};
use crate::hnsw::{HnswIndex, HnswParams};
use crate::schema::{FieldSchema, FieldType};
use crate::storage::Storage;

// ---------------------------------------------------------------------------
// Inverted Index
// ---------------------------------------------------------------------------

/// In-memory inverted index for fast filter evaluation.
///
/// Maps `field_name -> value -> Set<internal_id>`.
/// For Tags fields, each individual tag is indexed separately.
struct InvertedIndex {
    /// field -> value -> set of internal IDs
    index: HashMap<String, HashMap<String, HashSet<u32>>>,
}

impl InvertedIndex {
    fn new() -> Self {
        Self {
            index: HashMap::new(),
        }
    }

    /// Add a document's fields to the inverted index.
    fn insert(&mut self, internal_id: u32, fields: &HashMap<String, String>, schema: &FieldSchema) {
        for (field_name, value) in fields {
            let field_type = schema.field_type(field_name);
            match field_type {
                Some(FieldType::Filtered) => {
                    self.index
                        .entry(field_name.clone())
                        .or_default()
                        .entry(value.clone())
                        .or_default()
                        .insert(internal_id);
                }
                Some(FieldType::Tags) => {
                    // Index each tag separately
                    for tag in value.split(',') {
                        let tag = tag.trim();
                        if !tag.is_empty() {
                            self.index
                                .entry(field_name.clone())
                                .or_default()
                                .entry(tag.to_string())
                                .or_default()
                                .insert(internal_id);
                        }
                    }
                }
                _ => {
                    // String fields or unknown fields: not indexed
                    // But if no schema is defined, index all fields for backward compat
                    if !schema.has_indexed_fields() {
                        self.index
                            .entry(field_name.clone())
                            .or_default()
                            .entry(value.clone())
                            .or_default()
                            .insert(internal_id);
                    }
                }
            }
        }
    }

    /// Remove a document's fields from the inverted index.
    fn remove(&mut self, internal_id: u32, fields: &HashMap<String, String>, schema: &FieldSchema) {
        for (field_name, value) in fields {
            let field_type = schema.field_type(field_name);
            match field_type {
                Some(FieldType::Filtered) => {
                    if let Some(value_map) = self.index.get_mut(field_name) {
                        if let Some(ids) = value_map.get_mut(value) {
                            ids.remove(&internal_id);
                            if ids.is_empty() {
                                value_map.remove(value);
                            }
                        }
                    }
                }
                Some(FieldType::Tags) => {
                    for tag in value.split(',') {
                        let tag = tag.trim();
                        if !tag.is_empty() {
                            if let Some(value_map) = self.index.get_mut(field_name) {
                                if let Some(ids) = value_map.get_mut(tag) {
                                    ids.remove(&internal_id);
                                    if ids.is_empty() {
                                        value_map.remove(tag);
                                    }
                                }
                            }
                        }
                    }
                }
                _ => {
                    if !schema.has_indexed_fields() {
                        if let Some(value_map) = self.index.get_mut(field_name) {
                            if let Some(ids) = value_map.get_mut(value) {
                                ids.remove(&internal_id);
                                if ids.is_empty() {
                                    value_map.remove(value);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Evaluate a filter expression against the inverted index.
    /// Returns the set of internal IDs matching the filter.
    fn evaluate(&self, expr: &FilterExpr, all_ids: &HashSet<u32>) -> HashSet<u32> {
        match expr {
            FilterExpr::Eq(field, value) => {
                self.index
                    .get(field)
                    .and_then(|vm| vm.get(value))
                    .cloned()
                    .unwrap_or_default()
            }
            FilterExpr::Ne(field, value) => {
                let matching = self.index
                    .get(field)
                    .and_then(|vm| vm.get(value))
                    .cloned()
                    .unwrap_or_default();
                all_ids.difference(&matching).copied().collect()
            }
            FilterExpr::In(field, values) => {
                let mut result = HashSet::new();
                if let Some(value_map) = self.index.get(field) {
                    for v in values {
                        if let Some(ids) = value_map.get(v) {
                            result.extend(ids);
                        }
                    }
                }
                result
            }
            FilterExpr::Contains(field, value) => {
                // For tags: each tag is indexed individually, so this is a direct lookup
                self.index
                    .get(field)
                    .and_then(|vm| vm.get(value))
                    .cloned()
                    .unwrap_or_default()
            }
            FilterExpr::Lt(field, value) => {
                let mut result = HashSet::new();
                if let Some(value_map) = self.index.get(field) {
                    for (k, ids) in value_map {
                        if k.as_str() < value.as_str() {
                            result.extend(ids);
                        }
                    }
                }
                result
            }
            FilterExpr::Le(field, value) => {
                let mut result = HashSet::new();
                if let Some(value_map) = self.index.get(field) {
                    for (k, ids) in value_map {
                        if k.as_str() <= value.as_str() {
                            result.extend(ids);
                        }
                    }
                }
                result
            }
            FilterExpr::Gt(field, value) => {
                let mut result = HashSet::new();
                if let Some(value_map) = self.index.get(field) {
                    for (k, ids) in value_map {
                        if k.as_str() > value.as_str() {
                            result.extend(ids);
                        }
                    }
                }
                result
            }
            FilterExpr::Ge(field, value) => {
                let mut result = HashSet::new();
                if let Some(value_map) = self.index.get(field) {
                    for (k, ids) in value_map {
                        if k.as_str() >= value.as_str() {
                            result.extend(ids);
                        }
                    }
                }
                result
            }
            FilterExpr::Like(field, pattern) => {
                let mut result = HashSet::new();
                if let Some(value_map) = self.index.get(field) {
                    let starts = pattern.starts_with('%');
                    let ends = pattern.ends_with('%');
                    for (k, ids) in value_map {
                        let matched = match (starts, ends) {
                            (true, true) => {
                                let inner = &pattern[1..pattern.len() - 1];
                                k.contains(inner)
                            }
                            (true, false) => {
                                let suffix = &pattern[1..];
                                k.ends_with(suffix)
                            }
                            (false, true) => {
                                let prefix = &pattern[..pattern.len() - 1];
                                k.starts_with(prefix)
                            }
                            (false, false) => k == pattern,
                        };
                        if matched {
                            result.extend(ids);
                        }
                    }
                }
                result
            }
            FilterExpr::IsNull(field) => {
                // IDs that do NOT appear in any value for this field
                let has_field: HashSet<u32> = self.index
                    .get(field)
                    .map(|vm| vm.values().flat_map(|ids| ids.iter().copied()).collect())
                    .unwrap_or_default();
                all_ids.difference(&has_field).copied().collect()
            }
            FilterExpr::IsNotNull(field) => {
                // IDs that appear in at least one value for this field
                self.index
                    .get(field)
                    .map(|vm| vm.values().flat_map(|ids| ids.iter().copied()).collect())
                    .unwrap_or_default()
            }
            FilterExpr::And(left, right) => {
                let left_set = self.evaluate(left, all_ids);
                let right_set = self.evaluate(right, all_ids);
                left_set.intersection(&right_set).copied().collect()
            }
            FilterExpr::Or(left, right) => {
                let left_set = self.evaluate(left, all_ids);
                let right_set = self.evaluate(right, all_ids);
                left_set.union(&right_set).copied().collect()
            }
            FilterExpr::Not(inner) => {
                let inner_set = self.evaluate(inner, all_ids);
                all_ids.difference(&inner_set).copied().collect()
            }
        }
    }

    /// Get all internal IDs in the index (for Ne/Not operations).
    fn all_ids(&self) -> HashSet<u32> {
        let mut all = HashSet::new();
        for value_map in self.index.values() {
            for ids in value_map.values() {
                all.extend(ids);
            }
        }
        all
    }
}

// ---------------------------------------------------------------------------
// Collection Config
// ---------------------------------------------------------------------------

/// Configuration for creating a new collection.
#[derive(Debug, Clone)]
pub struct CollectionConfig {
    pub dims: usize,
    pub metric: MetricType,
    pub hnsw_params: HnswParams,
    pub schema: FieldSchema,
}

impl CollectionConfig {
    pub fn new(dims: usize) -> Self {
        Self {
            dims,
            metric: MetricType::IP,
            hnsw_params: HnswParams::default(),
            schema: FieldSchema::empty(),
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

    pub fn with_schema(mut self, schema: FieldSchema) -> Self {
        self.schema = schema;
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

// ---------------------------------------------------------------------------
// Collection
// ---------------------------------------------------------------------------

/// A search result grouped by a field value.
#[derive(Debug, Clone)]
pub struct GroupByResult {
    pub group_key: String,
    pub hits: Vec<SearchHit>,
}

/// A vector collection with HNSW index, metadata, inverted index, and field storage.
///
/// Thread-safe: supports concurrent inserts and searches.
pub struct Collection {
    index: HnswIndex,
    /// pk (String) -> field values
    fields: RwLock<HashMap<String, HashMap<String, String>>>,
    /// Internal ID -> pk
    pk_map: RwLock<Vec<String>>,
    /// Inverted index for filtered/tags fields
    inv_index: RwLock<InvertedIndex>,
    /// Persistent storage (None = in-memory only)
    storage: Option<RwLock<Storage>>,
    /// Mutable schema, separated from config for runtime mutations.
    schema: RwLock<FieldSchema>,
    config: CollectionConfig,
}

impl Collection {
    /// Create a new in-memory collection (no persistence).
    pub fn new(config: CollectionConfig) -> Self {
        let schema = config.schema.clone();
        Self {
            index: HnswIndex::new(config.dims, config.metric, config.hnsw_params.clone()),
            fields: RwLock::new(HashMap::new()),
            pk_map: RwLock::new(Vec::new()),
            inv_index: RwLock::new(InvertedIndex::new()),
            storage: None,
            schema: RwLock::new(schema),
            config,
        }
    }

    /// Open or create a persistent collection at the given path.
    pub fn open(
        path: impl AsRef<Path>,
        name: &str,
        config: CollectionConfig,
    ) -> Result<Self, String> {
        let db_path = path.as_ref().join(format!("{}.redb", name));

        let storage =
            Storage::open(&db_path).map_err(|e| format!("failed to open storage: {}", e))?;

        let schema = config.schema.clone();
        let mut collection = Self {
            index: HnswIndex::new(config.dims, config.metric, config.hnsw_params.clone()),
            fields: RwLock::new(HashMap::new()),
            pk_map: RwLock::new(Vec::new()),
            inv_index: RwLock::new(InvertedIndex::new()),
            storage: Some(RwLock::new(storage)),
            schema: RwLock::new(schema),
            config,
        };

        collection.load_from_storage()?;

        Ok(collection)
    }

    /// Load all data from persistent storage into the in-memory index.
    fn load_from_storage(&mut self) -> Result<(), String> {
        let storage = self
            .storage
            .as_ref()
            .ok_or("no storage configured")?
            .read()
            .map_err(|e| format!("storage lock: {}", e))?;

        // Restore schema from storage if not provided in config
        if !self.schema.read().unwrap().has_indexed_fields() {
            if let Ok(Some(schema_json)) = storage.load_schema() {
                if let Ok(schema) = FieldSchema::from_json(&schema_json) {
                    *self.schema.write().unwrap() = schema;
                }
            }
        }

        let id_map = storage
            .load_id_map()
            .map_err(|e| format!("load id_map: {}", e))?;

        if id_map.is_empty() {
            return Ok(());
        }

        let saved_state = storage
            .load_state()
            .map_err(|e| format!("load state: {}", e))?;

        let mut entries: Vec<(String, u32)> = id_map.into_iter().collect();
        entries.sort_by_key(|(_, id)| *id);

        let mut pk_map = self.pk_map.write().unwrap();
        let mut fields_map = self.fields.write().unwrap();
        let mut inv_index = self.inv_index.write().unwrap();

        let has_connections = entries.first().map_or(false, |(_, id)| {
            storage.get_connections(*id, 0).ok().flatten().is_some()
        });

        for (ext_id, internal_id) in &entries {
            while pk_map.len() <= *internal_id as usize {
                pk_map.push(String::new());
            }
            pk_map[*internal_id as usize] = ext_id.clone();

            let vector = storage
                .get_vector(*internal_id)
                .map_err(|e| format!("load vector {}: {}", internal_id, e))?
                .ok_or_else(|| format!("vector {} not found", internal_id))?;

            if has_connections {
                let mut connections = Vec::new();
                for level in 0u8.. {
                    match storage.get_connections(*internal_id, level) {
                        Ok(Some(conns)) => connections.push(conns),
                        _ => break,
                    }
                }
                if connections.is_empty() {
                    connections.push(Vec::new());
                }
                self.index
                    .restore_node(*internal_id as u64, &vector, connections);
            } else {
                self.index.insert(*internal_id as u64, &vector);
            }

            if let Some(meta) = storage
                .get_metadata(*internal_id)
                .map_err(|e| format!("load metadata {}: {}", internal_id, e))?
            {
                // Rebuild inverted index from loaded metadata
                inv_index.insert(*internal_id, &meta, &self.schema.read().unwrap());
                fields_map.insert(ext_id.clone(), meta);
            }
        }

        if has_connections {
            if let Some((entry_point, max_level, _)) = saved_state {
                if entry_point != u32::MAX {
                    self.index.set_entry_point(entry_point, max_level);
                }
            }
        }

        Ok(())
    }

    /// Insert or update a document.
    pub fn upsert(&self, pk: &str, vector: &[f32], fields: HashMap<String, String>) {
        // Remove old inverted index entries if this is an update
        {
            let fields_map = self.fields.read().unwrap();
            if let Some(old_fields) = fields_map.get(pk) {
                let pk_map = self.pk_map.read().unwrap();
                if let Some(pos) = pk_map.iter().position(|p| p == pk) {
                    self.inv_index
                        .write()
                        .unwrap()
                        .remove(pos as u32, old_fields, &self.schema.read().unwrap());
                }
            }
        }

        let numeric_id = self.pk_to_id(pk);

        self.index.insert(numeric_id, vector);

        // Update inverted index
        self.inv_index
            .write()
            .unwrap()
            .insert(numeric_id as u32, &fields, &self.schema.read().unwrap());

        self.fields
            .write()
            .unwrap()
            .insert(pk.to_string(), fields.clone());

        // Persist to storage
        if let Some(ref storage_lock) = self.storage {
            if let Ok(storage) = storage_lock.read() {
                let internal_id = numeric_id as u32;
                let conns = self
                    .index
                    .get_connections(internal_id)
                    .unwrap_or_else(|| vec![vec![]]);
                let _ = storage.put_vector(internal_id, pk, vector, &fields, &conns);
            }
        }
    }

    /// Remove a document by primary key.
    pub fn remove(&self, pk: &str) -> bool {
        let numeric_id = {
            let pk_map = self.pk_map.read().unwrap();
            pk_map.iter().position(|p| p == pk).map(|i| i as u64)
        };

        match numeric_id {
            Some(id) => {
                // Remove from inverted index
                {
                    let fields_map = self.fields.read().unwrap();
                    if let Some(old_fields) = fields_map.get(pk) {
                        self.inv_index
                            .write()
                            .unwrap()
                            .remove(id as u32, old_fields, &self.schema.read().unwrap());
                    }
                }

                self.index.remove(id);
                self.fields.write().unwrap().remove(pk);

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

    /// Delete all documents matching a filter expression.
    /// Returns the number of documents deleted.
    pub fn delete_by_filter(&self, filter_expr: &str) -> Result<usize, String> {
        let parsed =
            parse_filter(filter_expr).map_err(|e| format!("filter parse error: {}", e))?;

        // Collect PKs to delete (can't delete while iterating)
        let to_delete: Vec<String> = {
            let fields_map = self.fields.read().unwrap();
            fields_map
                .iter()
                .filter(|(_, doc_fields)| filter::matches(&parsed, doc_fields))
                .map(|(pk, _)| pk.clone())
                .collect()
        };

        let count = to_delete.len();
        for pk in &to_delete {
            self.remove(pk);
        }

        Ok(count)
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
            Some(expr) => {
                Some(parse_filter(expr).map_err(|e| format!("filter parse error: {}", e))?)
            }
            None => None,
        };

        // Use inverted index to get matching candidate set
        let matching_ids: Option<HashSet<u32>> = parsed_filter.as_ref().map(|filter| {
            let inv = self.inv_index.read().unwrap();
            let all_ids = inv.all_ids();
            inv.evaluate(filter, &all_ids)
        });

        // Determine over-fetch ratio based on selectivity
        let fetch_k = match &matching_ids {
            Some(ids) if ids.is_empty() => return Ok(Vec::new()),
            Some(ids) => {
                let total = self.index.len();
                if total == 0 {
                    return Ok(Vec::new());
                }
                let selectivity = ids.len() as f64 / total as f64;
                if selectivity < 0.01 {
                    // Very selective: need to fetch a lot more
                    top_k * 100
                } else if selectivity < 0.1 {
                    top_k * 20
                } else if selectivity < 0.5 {
                    top_k * 4
                } else {
                    top_k * 2
                }
            }
            None => top_k,
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

                // Fast path: check inverted index candidate set (O(1) HashSet lookup)
                if let Some(ref ids) = matching_ids {
                    if !ids.contains(&(id as u32)) {
                        return None;
                    }
                }

                let doc_fields = fields_map.get(pk)?;

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
    pub fn flush(&self) -> Result<bool, String> {
        match self.storage {
            Some(ref storage_lock) => {
                let mut storage = storage_lock
                    .write()
                    .map_err(|e| format!("storage lock: {}", e))?;

                // Persist schema
                {
                    let schema = self.schema.read().unwrap();
                    if schema.has_indexed_fields() {
                        let schema_json = serde_json::to_string(
                            &schema
                                .fields()
                                .iter()
                                .map(|(name, ft)| {
                                    let mut m = HashMap::new();
                                    m.insert(
                                        "name".to_string(),
                                        name.clone(),
                                    );
                                    m.insert(
                                        "type".to_string(),
                                        match ft {
                                            FieldType::String => "string",
                                            FieldType::Filtered => "filtered",
                                            FieldType::Tags => "tags",
                                        }
                                        .to_string(),
                                    );
                                    m
                                })
                                .collect::<Vec<_>>(),
                        )
                        .map_err(|e| format!("serialize schema: {}", e))?;
                        storage
                            .save_schema(&schema_json)
                            .map_err(|e| format!("save schema: {}", e))?;
                    }
                }

                // Batch persist all active nodes
                let pk_map = self.pk_map.read().unwrap();
                let fields_map = self.fields.read().unwrap();

                let mut batch: Vec<(
                    u32,
                    &str,
                    Vec<f32>,
                    HashMap<String, String>,
                    Vec<Vec<u32>>,
                )> = Vec::new();

                for (idx, pk) in pk_map.iter().enumerate() {
                    if pk.is_empty() {
                        continue;
                    }
                    let internal_id = idx as u32;
                    if let Some(conns) = self.index.get_connections(internal_id) {
                        let ext_id = idx as u64;
                        if let Some(vector) = self.index.get_vector(ext_id) {
                            let fields = fields_map.get(pk).cloned().unwrap_or_default();
                            batch.push((internal_id, pk, vector, fields, conns));
                        }
                    }
                }

                // Write in batches of 1000 for memory efficiency
                for chunk in batch.chunks(1000) {
                    let refs: Vec<(u32, &str, &[f32], &HashMap<String, String>, &[Vec<u32>])> =
                        chunk
                            .iter()
                            .map(|(id, pk, v, f, c)| (*id, &pk[..], v.as_slice(), f, c.as_slice()))
                            .collect();
                    storage
                        .put_vectors_batch(&refs)
                        .map_err(|e| format!("batch persist: {}", e))?;
                }

                let next_id = pk_map.len() as u32;
                let entry_point = self.index.entry_point().unwrap_or(u32::MAX);
                let max_level = self.index.max_level();
                storage
                    .save_state(entry_point, max_level, next_id)
                    .map_err(|e| format!("save state: {}", e))?;

                storage.flush().map_err(|e| format!("flush: {}", e))?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Batch insert or update multiple documents.
    /// More efficient than individual upserts for large imports.
    pub fn upsert_batch(&self, docs: Vec<(&str, &[f32], HashMap<String, String>)>) {
        for (pk, vector, fields) in docs {
            self.upsert(pk, vector, fields);
        }
    }

    /// Search with output field selection — only return specified fields.
    pub fn search_with_fields(
        &self,
        vector: &[f32],
        top_k: usize,
        filter_expr: Option<&str>,
        output_fields: &[&str],
    ) -> Result<Vec<SearchHit>, String> {
        let mut hits = self.search(vector, top_k, filter_expr)?;

        if !output_fields.is_empty() {
            let field_set: HashSet<&str> = output_fields.iter().copied().collect();
            for hit in &mut hits {
                hit.fields.retain(|k, _| field_set.contains(k.as_str()));
            }
        }

        Ok(hits)
    }

    /// Optimize the collection: flush to disk and compact the database.
    pub fn optimize(&self) -> Result<bool, String> {
        self.flush()
    }

    /// Get the number of documents in the collection.
    pub fn doc_count(&self) -> usize {
        self.index.len()
    }

    /// Get collection configuration.
    pub fn config(&self) -> &CollectionConfig {
        &self.config
    }

    // -----------------------------------------------------------------------
    // GroupBy Search
    // -----------------------------------------------------------------------

    /// Search and group results by a field value.
    ///
    /// Returns up to `max_groups` groups, each with up to `top_k_per_group` hits.
    /// Groups are sorted by the best score in each group.
    pub fn group_by_search(
        &self,
        vector: &[f32],
        group_field: &str,
        max_groups: usize,
        top_k_per_group: usize,
        filter_expr: Option<&str>,
    ) -> Result<Vec<GroupByResult>, String> {
        // Over-fetch to have enough results for grouping.
        let over_fetch = max_groups * top_k_per_group * 4;
        let all_hits = self.search(vector, over_fetch, filter_expr)?;

        // Group by the specified field.
        let mut groups: HashMap<String, Vec<SearchHit>> = HashMap::new();
        for hit in all_hits {
            if let Some(key) = hit.fields.get(group_field) {
                groups
                    .entry(key.clone())
                    .or_default()
                    .push(hit);
            }
        }

        // Truncate each group and collect.
        let mut results: Vec<GroupByResult> = groups
            .into_iter()
            .map(|(key, mut hits)| {
                hits.truncate(top_k_per_group);
                GroupByResult {
                    group_key: key,
                    hits,
                }
            })
            .collect();

        // Sort groups by best score (first hit in each group, since search results are ordered).
        let is_similarity = self.config.metric.is_similarity();
        results.sort_by(|a, b| {
            let score_a = a.hits.first().map(|h| h.score).unwrap_or(f32::NEG_INFINITY);
            let score_b = b.hits.first().map(|h| h.score).unwrap_or(f32::NEG_INFINITY);
            if is_similarity {
                score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
            }
        });

        results.truncate(max_groups);
        Ok(results)
    }

    // -----------------------------------------------------------------------
    // Schema Mutations
    // -----------------------------------------------------------------------

    /// Add a new field to the schema.
    ///
    /// If the new field is Filtered or Tags, rebuilds inverted index entries
    /// for all existing documents that have a value for this field.
    pub fn add_field(&self, name: &str, field_type: FieldType) -> Result<(), String> {
        let mut schema = self.schema.write().unwrap();
        if !schema.add_field(name.to_string(), field_type) {
            return Err(format!("field '{}' already exists", name));
        }

        // If indexed, rebuild inverted index entries for existing docs.
        if matches!(field_type, FieldType::Filtered | FieldType::Tags) {
            let fields_map = self.fields.read().unwrap();
            let pk_map = self.pk_map.read().unwrap();
            let mut inv = self.inv_index.write().unwrap();

            for (idx, pk) in pk_map.iter().enumerate() {
                if pk.is_empty() {
                    continue;
                }
                if let Some(doc_fields) = fields_map.get(pk) {
                    if let Some(value) = doc_fields.get(name) {
                        // Build a single-field map to reuse InvertedIndex::insert logic
                        let single = {
                            let mut m = HashMap::new();
                            m.insert(name.to_string(), value.clone());
                            m
                        };
                        inv.insert(idx as u32, &single, &schema);
                    }
                }
            }
        }

        Ok(())
    }

    /// Remove a field from the schema and all documents.
    ///
    /// Removes the field from the schema, all document field maps,
    /// and the inverted index.
    pub fn drop_field(&self, name: &str) -> Result<(), String> {
        let mut schema = self.schema.write().unwrap();
        if !schema.remove_field(name) {
            return Err(format!("field '{}' not found in schema", name));
        }

        // Remove from inverted index.
        {
            let mut inv = self.inv_index.write().unwrap();
            inv.index.remove(name);
        }

        // Remove from all document field maps.
        {
            let mut fields_map = self.fields.write().unwrap();
            for doc_fields in fields_map.values_mut() {
                doc_fields.remove(name);
            }
        }

        Ok(())
    }

    /// Rename a field across the schema and all documents.
    ///
    /// Updates the schema, all document field maps, and the inverted index.
    pub fn rename_field(&self, old_name: &str, new_name: &str) -> Result<(), String> {
        let mut schema = self.schema.write().unwrap();
        if !schema.rename_field(old_name, new_name) {
            return Err(format!(
                "cannot rename '{}' to '{}': source not found or target already exists",
                old_name, new_name
            ));
        }

        // Rename in inverted index.
        {
            let mut inv = self.inv_index.write().unwrap();
            if let Some(entries) = inv.index.remove(old_name) {
                inv.index.insert(new_name.to_string(), entries);
            }
        }

        // Rename in all document field maps.
        {
            let mut fields_map = self.fields.write().unwrap();
            for doc_fields in fields_map.values_mut() {
                if let Some(value) = doc_fields.remove(old_name) {
                    doc_fields.insert(new_name.to_string(), value);
                }
            }
        }

        Ok(())
    }

    /// Map a string pk to a numeric ID. Assigns new IDs as needed.
    fn pk_to_id(&self, pk: &str) -> u64 {
        let mut pk_map = self.pk_map.write().unwrap();

        if let Some(pos) = pk_map.iter().position(|p| p == pk) {
            return pos as u64;
        }

        let id = pk_map.len() as u64;
        pk_map.push(pk.to_string());
        id
    }
}

impl Drop for Collection {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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

    fn test_schema() -> FieldSchema {
        FieldSchema::new(vec![
            ("category".into(), FieldType::Filtered),
            ("tenant".into(), FieldType::Filtered),
        ])
    }

    fn test_schema_with_tags() -> FieldSchema {
        FieldSchema::new(vec![
            ("content".into(), FieldType::String),
            ("category".into(), FieldType::Filtered),
            ("tenant".into(), FieldType::Filtered),
            ("labels".into(), FieldType::Tags),
        ])
    }

    #[test]
    fn test_basic_crud() {
        let col = Collection::new(CollectionConfig::new(3).with_schema(test_schema()));

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
        let col = Collection::new(
            CollectionConfig::new(3)
                .with_metric(MetricType::L2)
                .with_schema(test_schema()),
        );

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
        let col = Collection::new(
            CollectionConfig::new(dims)
                .with_metric(MetricType::L2)
                .with_schema(test_schema()),
        );

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
        let col = Collection::new(
            CollectionConfig::new(dims)
                .with_metric(MetricType::L2)
                .with_schema(test_schema()),
        );

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
    fn test_search_with_tags_contains() {
        let dims = 4;
        let col = Collection::new(
            CollectionConfig::new(dims)
                .with_metric(MetricType::L2)
                .with_schema(test_schema_with_tags()),
        );

        let mut fields1 = HashMap::new();
        fields1.insert("category".to_string(), "lang".to_string());
        fields1.insert("labels".to_string(), "rust,systems,fast".to_string());
        col.upsert("doc-1", &[1.0, 0.0, 0.0, 0.0], fields1);

        let mut fields2 = HashMap::new();
        fields2.insert("category".to_string(), "lang".to_string());
        fields2.insert("labels".to_string(), "python,scripting,fast".to_string());
        col.upsert("doc-2", &[0.0, 1.0, 0.0, 0.0], fields2);

        let mut fields3 = HashMap::new();
        fields3.insert("category".to_string(), "lang".to_string());
        fields3.insert("labels".to_string(), "java,enterprise".to_string());
        col.upsert("doc-3", &[0.0, 0.0, 1.0, 0.0], fields3);

        // CONTAINS should match individual tags
        let results = col
            .search(&[0.5, 0.5, 0.5, 0.0], 10, Some("labels CONTAINS 'rust'"))
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].pk, "doc-1");

        // CONTAINS 'fast' should match doc-1 and doc-2
        let results = col
            .search(&[0.5, 0.5, 0.5, 0.0], 10, Some("labels CONTAINS 'fast'"))
            .unwrap();
        assert_eq!(results.len(), 2);

        // Compound: CONTAINS + AND
        let results = col
            .search(
                &[0.5, 0.5, 0.5, 0.0],
                10,
                Some("labels CONTAINS 'fast' AND labels CONTAINS 'rust'"),
            )
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].pk, "doc-1");
    }

    #[test]
    fn test_search_with_in_filter() {
        let dims = 4;
        let col = Collection::new(
            CollectionConfig::new(dims)
                .with_metric(MetricType::L2)
                .with_schema(test_schema()),
        );

        col.upsert("doc-1", &[1.0, 0.0, 0.0, 0.0], test_fields("a", "t1"));
        col.upsert("doc-2", &[0.0, 1.0, 0.0, 0.0], test_fields("b", "t1"));
        col.upsert("doc-3", &[0.0, 0.0, 1.0, 0.0], test_fields("c", "t1"));

        let results = col
            .search(
                &[0.5, 0.5, 0.5, 0.0],
                10,
                Some("category IN ('a', 'c')"),
            )
            .unwrap();
        assert_eq!(results.len(), 2);
        let pks: HashSet<String> = results.iter().map(|h| h.pk.clone()).collect();
        assert!(pks.contains("doc-1"));
        assert!(pks.contains("doc-3"));
    }

    #[test]
    fn test_search_with_not_filter() {
        let dims = 4;
        let col = Collection::new(
            CollectionConfig::new(dims)
                .with_metric(MetricType::L2)
                .with_schema(test_schema()),
        );

        col.upsert("doc-1", &[1.0, 0.0, 0.0, 0.0], test_fields("a", "t1"));
        col.upsert("doc-2", &[0.0, 1.0, 0.0, 0.0], test_fields("b", "t1"));
        col.upsert("doc-3", &[0.0, 0.0, 1.0, 0.0], test_fields("c", "t1"));

        let results = col
            .search(
                &[0.5, 0.5, 0.5, 0.0],
                10,
                Some("NOT category = 'a'"),
            )
            .unwrap();
        assert_eq!(results.len(), 2);
        for hit in &results {
            assert_ne!(hit.fields["category"], "a");
        }
    }

    #[test]
    fn test_delete_by_filter() {
        let dims = 4;
        let col = Collection::new(
            CollectionConfig::new(dims)
                .with_metric(MetricType::L2)
                .with_schema(test_schema()),
        );

        col.upsert("doc-1", &[1.0, 0.0, 0.0, 0.0], test_fields("a", "t1"));
        col.upsert("doc-2", &[0.0, 1.0, 0.0, 0.0], test_fields("b", "t1"));
        col.upsert("doc-3", &[0.0, 0.0, 1.0, 0.0], test_fields("a", "t2"));
        col.upsert("doc-4", &[0.0, 0.0, 0.0, 1.0], test_fields("c", "t1"));

        assert_eq!(col.doc_count(), 4);

        let deleted = col.delete_by_filter("category = 'a'").unwrap();
        assert_eq!(deleted, 2);
        assert_eq!(col.doc_count(), 2);
        assert!(col.fetch("doc-1").is_none());
        assert!(col.fetch("doc-3").is_none());
        assert!(col.fetch("doc-2").is_some());
        assert!(col.fetch("doc-4").is_some());
    }

    #[test]
    fn test_delete_by_compound_filter() {
        let dims = 4;
        let col = Collection::new(
            CollectionConfig::new(dims)
                .with_metric(MetricType::L2)
                .with_schema(test_schema()),
        );

        col.upsert("doc-1", &[1.0, 0.0, 0.0, 0.0], test_fields("a", "t1"));
        col.upsert("doc-2", &[0.0, 1.0, 0.0, 0.0], test_fields("a", "t2"));
        col.upsert("doc-3", &[0.0, 0.0, 1.0, 0.0], test_fields("b", "t1"));

        let deleted = col
            .delete_by_filter("category = 'a' AND tenant = 't1'")
            .unwrap();
        assert_eq!(deleted, 1);
        assert_eq!(col.doc_count(), 2);
        assert!(col.fetch("doc-1").is_none());
        assert!(col.fetch("doc-2").is_some());
    }

    #[test]
    fn test_upsert_updates() {
        let col = Collection::new(CollectionConfig::new(3).with_schema(test_schema()));

        col.upsert("doc-1", &[1.0, 0.0, 0.0], test_fields("old", "t1"));
        col.upsert("doc-1", &[0.0, 1.0, 0.0], test_fields("new", "t1"));

        let fields = col.fetch("doc-1").unwrap();
        assert_eq!(fields["category"], "new");
    }

    #[test]
    fn test_upsert_updates_inverted_index() {
        let dims = 4;
        let col = Collection::new(
            CollectionConfig::new(dims)
                .with_metric(MetricType::L2)
                .with_schema(test_schema()),
        );

        col.upsert("doc-1", &[1.0, 0.0, 0.0, 0.0], test_fields("old_cat", "t1"));
        col.upsert("doc-2", &[0.0, 1.0, 0.0, 0.0], test_fields("other", "t1"));

        // Searching for old category should find doc-1
        let results = col
            .search(&[0.5, 0.5, 0.0, 0.0], 10, Some("category = 'old_cat'"))
            .unwrap();
        assert_eq!(results.len(), 1);

        // Update doc-1's category
        col.upsert("doc-1", &[1.0, 0.0, 0.0, 0.0], test_fields("new_cat", "t1"));

        // Old category should no longer match
        let results = col
            .search(&[0.5, 0.5, 0.0, 0.0], 10, Some("category = 'old_cat'"))
            .unwrap();
        assert_eq!(results.len(), 0);

        // New category should match
        let results = col
            .search(&[0.5, 0.5, 0.0, 0.0], 10, Some("category = 'new_cat'"))
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].pk, "doc-1");
    }

    #[test]
    fn test_concurrent_collection() {
        let dims = 16;
        let col = Arc::new(Collection::new(
            CollectionConfig::new(dims)
                .with_metric(MetricType::L2)
                .with_schema(test_schema()),
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
    fn test_no_schema_backward_compat() {
        // Without a schema, all fields should still be filterable (backward compat)
        let dims = 4;
        let col = Collection::new(CollectionConfig::new(dims).with_metric(MetricType::L2));

        col.upsert("doc-1", &[1.0, 0.0, 0.0, 0.0], test_fields("a", "t1"));
        col.upsert("doc-2", &[0.0, 1.0, 0.0, 0.0], test_fields("b", "t1"));

        let results = col
            .search(&[0.5, 0.5, 0.0, 0.0], 10, Some("category = 'a'"))
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].pk, "doc-1");
    }

    #[test]
    fn test_persistent_collection() {
        let dir = tempfile::tempdir().unwrap();
        let dims = 4;
        let config = CollectionConfig::new(dims)
            .with_metric(MetricType::L2)
            .with_schema(test_schema());

        {
            let col = Collection::open(dir.path(), "test", config.clone()).unwrap();
            col.upsert("doc-1", &[1.0, 0.0, 0.0, 0.0], test_fields("a", "t1"));
            col.upsert("doc-2", &[0.0, 1.0, 0.0, 0.0], test_fields("b", "t1"));
            col.upsert("doc-3", &[0.0, 0.0, 1.0, 0.0], test_fields("c", "t2"));
            col.flush().unwrap();
        }

        {
            let col = Collection::open(dir.path(), "test", config.clone()).unwrap();
            assert_eq!(col.doc_count(), 3);

            let fetched = col.fetch("doc-1").unwrap();
            assert_eq!(fetched["category"], "a");

            let fetched = col.fetch("doc-3").unwrap();
            assert_eq!(fetched["tenant"], "t2");

            let results = col.search(&[1.0, 0.0, 0.0, 0.0], 2, None).unwrap();
            assert!(!results.is_empty());
            assert_eq!(results[0].pk, "doc-1");
        }
    }

    #[test]
    fn test_persistent_remove() {
        let dir = tempfile::tempdir().unwrap();
        let dims = 3;
        let config = CollectionConfig::new(dims)
            .with_metric(MetricType::L2)
            .with_schema(test_schema());

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
    fn test_persistent_graph_topology() {
        let dir = tempfile::tempdir().unwrap();
        let dims = 8;
        let config = CollectionConfig::new(dims)
            .with_metric(MetricType::L2)
            .with_schema(test_schema());

        let mut rng = rand::thread_rng();
        let mut vectors: Vec<(String, Vec<f32>)> = Vec::new();
        for i in 0..20 {
            let v: Vec<f32> = (0..dims).map(|_| rng.gen()).collect();
            vectors.push((format!("doc-{}", i), v));
        }
        let query: Vec<f32> = (0..dims).map(|_| rng.gen()).collect();

        let results_before;
        {
            let col = Collection::open(dir.path(), "test", config.clone()).unwrap();
            for (pk, v) in &vectors {
                col.upsert(pk, v, test_fields("a", "t1"));
            }
            results_before = col.search(&query, 5, None).unwrap();
            col.flush().unwrap();
        }

        {
            let col = Collection::open(dir.path(), "test", config.clone()).unwrap();
            assert_eq!(col.doc_count(), 20);
            let results_after = col.search(&query, 5, None).unwrap();

            assert_eq!(results_before.len(), results_after.len());
            for (a, b) in results_before.iter().zip(results_after.iter()) {
                assert_eq!(a.pk, b.pk, "search result order changed after reload");
            }
        }
    }

    #[test]
    fn test_persistent_upsert_after_reload() {
        let dir = tempfile::tempdir().unwrap();
        let dims = 4;
        let config = CollectionConfig::new(dims)
            .with_metric(MetricType::L2)
            .with_schema(test_schema());

        {
            let col = Collection::open(dir.path(), "test", config.clone()).unwrap();
            col.upsert("doc-1", &[1.0, 0.0, 0.0, 0.0], test_fields("a", "t1"));
            col.upsert("doc-2", &[0.0, 1.0, 0.0, 0.0], test_fields("b", "t1"));
            col.flush().unwrap();
        }

        {
            let col = Collection::open(dir.path(), "test", config.clone()).unwrap();
            assert_eq!(col.doc_count(), 2);
            col.upsert("doc-3", &[0.0, 0.0, 1.0, 0.0], test_fields("c", "t1"));
            col.upsert("doc-4", &[0.0, 0.0, 0.0, 1.0], test_fields("d", "t1"));
            assert_eq!(col.doc_count(), 4);

            let results = col.search(&[0.0, 0.0, 1.0, 0.0], 1, None).unwrap();
            assert_eq!(results[0].pk, "doc-3");
            col.flush().unwrap();
        }

        {
            let col = Collection::open(dir.path(), "test", config).unwrap();
            assert_eq!(col.doc_count(), 4);
            let f = col.fetch("doc-4").unwrap();
            assert_eq!(f["category"], "d");
        }
    }

    #[test]
    fn test_persistent_filter_after_reload() {
        let dir = tempfile::tempdir().unwrap();
        let dims = 4;
        let config = CollectionConfig::new(dims)
            .with_metric(MetricType::L2)
            .with_schema(test_schema());

        {
            let col = Collection::open(dir.path(), "test", config.clone()).unwrap();
            col.upsert("doc-1", &[1.0, 0.0, 0.0, 0.0], test_fields("a", "t1"));
            col.upsert("doc-2", &[0.0, 1.0, 0.0, 0.0], test_fields("b", "t1"));
            col.upsert("doc-3", &[0.0, 0.0, 1.0, 0.0], test_fields("a", "t2"));
            col.flush().unwrap();
        }

        {
            // Reopen — inverted index should be rebuilt from metadata
            let col = Collection::open(dir.path(), "test", config).unwrap();
            assert_eq!(col.doc_count(), 3);

            // Filter should work using rebuilt inverted index
            let results = col
                .search(&[0.5, 0.5, 0.5, 0.0], 10, Some("category = 'a'"))
                .unwrap();
            assert_eq!(results.len(), 2);

            let results = col
                .search(
                    &[0.5, 0.5, 0.5, 0.0],
                    10,
                    Some("category = 'a' AND tenant = 't1'"),
                )
                .unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].pk, "doc-1");
        }
    }

    #[test]
    fn test_batch_upsert() {
        let dims = 4;
        let col = Collection::new(
            CollectionConfig::new(dims)
                .with_metric(MetricType::L2)
                .with_schema(test_schema()),
        );

        let docs: Vec<(&str, &[f32], HashMap<String, String>)> = vec![
            ("doc-1", &[1.0, 0.0, 0.0, 0.0], test_fields("a", "t1")),
            ("doc-2", &[0.0, 1.0, 0.0, 0.0], test_fields("b", "t1")),
            ("doc-3", &[0.0, 0.0, 1.0, 0.0], test_fields("c", "t2")),
        ];
        col.upsert_batch(docs);

        assert_eq!(col.doc_count(), 3);
        assert_eq!(col.fetch("doc-2").unwrap()["category"], "b");
    }

    #[test]
    fn test_search_with_output_fields() {
        let dims = 4;
        let col = Collection::new(
            CollectionConfig::new(dims)
                .with_metric(MetricType::L2)
                .with_schema(test_schema()),
        );

        col.upsert("doc-1", &[1.0, 0.0, 0.0, 0.0], test_fields("a", "t1"));

        // Only return 'category' field
        let results = col
            .search_with_fields(&[1.0, 0.0, 0.0, 0.0], 1, None, &["category"])
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].fields.contains_key("category"));
        assert!(!results[0].fields.contains_key("tenant"));
    }

    #[test]
    fn test_persistent_schema_survives_restart() {
        let dir = tempfile::tempdir().unwrap();
        let dims = 4;
        let config = CollectionConfig::new(dims)
            .with_metric(MetricType::L2)
            .with_schema(test_schema());

        {
            let col = Collection::open(dir.path(), "test", config.clone()).unwrap();
            col.upsert("doc-1", &[1.0, 0.0, 0.0, 0.0], test_fields("a", "t1"));
            col.upsert("doc-2", &[0.0, 1.0, 0.0, 0.0], test_fields("b", "t1"));
            col.flush().unwrap();
        }

        // Reopen WITHOUT providing a schema — it should be loaded from storage
        let config_no_schema = CollectionConfig::new(dims).with_metric(MetricType::L2);
        {
            let col = Collection::open(dir.path(), "test", config_no_schema).unwrap();
            assert_eq!(col.doc_count(), 2);

            // Filters should still work because schema was loaded from storage
            let results = col
                .search(&[0.5, 0.5, 0.0, 0.0], 10, Some("category = 'a'"))
                .unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].pk, "doc-1");
        }
    }

    #[test]
    fn test_cosine_metric_collection() {
        let dims = 4;
        let col = Collection::new(
            CollectionConfig::new(dims).with_metric(MetricType::Cosine),
        );

        col.upsert("doc-1", &[1.0, 0.0, 0.0, 0.0], HashMap::new());
        col.upsert("doc-2", &[0.0, 1.0, 0.0, 0.0], HashMap::new());
        col.upsert("doc-3", &[0.7, 0.7, 0.0, 0.0], HashMap::new());

        let results = col.search(&[1.0, 0.0, 0.0, 0.0], 3, None).unwrap();
        // doc-1 should be closest (cosine=1.0)
        assert_eq!(results[0].pk, "doc-1");
    }

    #[test]
    fn test_flush_in_memory() {
        let col = Collection::new(CollectionConfig::new(3));
        assert_eq!(col.flush().unwrap(), false);
    }

    // -----------------------------------------------------------------------
    // GroupBy tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_group_by_search() {
        let dims = 4;
        let col = Collection::new(
            CollectionConfig::new(dims)
                .with_metric(MetricType::L2)
                .with_schema(test_schema()),
        );

        // Insert docs in 3 categories with varying vectors.
        col.upsert("doc-1", &[1.0, 0.0, 0.0, 0.0], test_fields("a", "t1"));
        col.upsert("doc-2", &[0.9, 0.1, 0.0, 0.0], test_fields("a", "t1"));
        col.upsert("doc-3", &[0.0, 1.0, 0.0, 0.0], test_fields("b", "t1"));
        col.upsert("doc-4", &[0.0, 0.9, 0.1, 0.0], test_fields("b", "t1"));
        col.upsert("doc-5", &[0.0, 0.0, 1.0, 0.0], test_fields("c", "t1"));

        let groups = col
            .group_by_search(&[1.0, 0.0, 0.0, 0.0], "category", 3, 2, None)
            .unwrap();

        // Should have 3 groups.
        assert_eq!(groups.len(), 3);

        // Each group should have at most 2 hits.
        for g in &groups {
            assert!(g.hits.len() <= 2);
            // All hits in a group should share the same category.
            for hit in &g.hits {
                assert_eq!(hit.fields["category"], g.group_key);
            }
        }

        // First group should be "a" (closest to query [1,0,0,0]).
        assert_eq!(groups[0].group_key, "a");
    }

    #[test]
    fn test_group_by_with_filter() {
        let dims = 4;
        let col = Collection::new(
            CollectionConfig::new(dims)
                .with_metric(MetricType::L2)
                .with_schema(test_schema()),
        );

        col.upsert("doc-1", &[1.0, 0.0, 0.0, 0.0], test_fields("a", "t1"));
        col.upsert("doc-2", &[0.9, 0.1, 0.0, 0.0], test_fields("a", "t2"));
        col.upsert("doc-3", &[0.0, 1.0, 0.0, 0.0], test_fields("b", "t1"));
        col.upsert("doc-4", &[0.0, 0.0, 1.0, 0.0], test_fields("b", "t2"));

        // Group by category but filter to tenant = 't1'.
        let groups = col
            .group_by_search(
                &[0.5, 0.5, 0.5, 0.0],
                "category",
                10,
                5,
                Some("tenant = 't1'"),
            )
            .unwrap();

        // Should only contain docs from t1.
        assert_eq!(groups.len(), 2); // a and b
        for g in &groups {
            for hit in &g.hits {
                assert_eq!(hit.fields["tenant"], "t1");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Schema mutation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_field() {
        let dims = 4;
        let col = Collection::new(
            CollectionConfig::new(dims)
                .with_metric(MetricType::L2)
                .with_schema(FieldSchema::new(vec![
                    ("category".into(), FieldType::Filtered),
                ])),
        );

        // Insert docs with a "priority" field that isn't in the schema yet.
        let mut f1 = HashMap::new();
        f1.insert("category".to_string(), "a".to_string());
        f1.insert("priority".to_string(), "high".to_string());
        col.upsert("doc-1", &[1.0, 0.0, 0.0, 0.0], f1);

        let mut f2 = HashMap::new();
        f2.insert("category".to_string(), "b".to_string());
        f2.insert("priority".to_string(), "low".to_string());
        col.upsert("doc-2", &[0.0, 1.0, 0.0, 0.0], f2);

        // "priority" is not indexed yet, so filtering on it won't work via inverted index.
        // Add the field as Filtered.
        col.add_field("priority", FieldType::Filtered).unwrap();

        // Now filtering on "priority" should work.
        let results = col
            .search(&[0.5, 0.5, 0.0, 0.0], 10, Some("priority = 'high'"))
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].pk, "doc-1");

        // Adding duplicate should fail.
        assert!(col.add_field("priority", FieldType::Filtered).is_err());
    }

    #[test]
    fn test_drop_field() {
        let dims = 4;
        let col = Collection::new(
            CollectionConfig::new(dims)
                .with_metric(MetricType::L2)
                .with_schema(test_schema()),
        );

        col.upsert("doc-1", &[1.0, 0.0, 0.0, 0.0], test_fields("a", "t1"));
        col.upsert("doc-2", &[0.0, 1.0, 0.0, 0.0], test_fields("b", "t1"));

        // Drop the "category" field.
        col.drop_field("category").unwrap();

        // Field should be gone from documents.
        let f = col.fetch("doc-1").unwrap();
        assert!(!f.contains_key("category"));
        assert!(f.contains_key("tenant")); // other fields preserved

        // Filtering on dropped field should return empty (no inverted index entries).
        let results = col
            .search(&[0.5, 0.5, 0.0, 0.0], 10, Some("category = 'a'"))
            .unwrap();
        assert_eq!(results.len(), 0);

        // Dropping non-existent field should fail.
        assert!(col.drop_field("nonexistent").is_err());
    }

    #[test]
    fn test_rename_field() {
        let dims = 4;
        let col = Collection::new(
            CollectionConfig::new(dims)
                .with_metric(MetricType::L2)
                .with_schema(test_schema()),
        );

        col.upsert("doc-1", &[1.0, 0.0, 0.0, 0.0], test_fields("a", "t1"));
        col.upsert("doc-2", &[0.0, 1.0, 0.0, 0.0], test_fields("b", "t1"));

        // Rename "category" to "group".
        col.rename_field("category", "group").unwrap();

        // Old name should be gone, new name should have the value.
        let f = col.fetch("doc-1").unwrap();
        assert!(!f.contains_key("category"));
        assert_eq!(f["group"], "a");

        // Filtering on old name should yield nothing.
        let results = col
            .search(&[0.5, 0.5, 0.0, 0.0], 10, Some("category = 'a'"))
            .unwrap();
        assert_eq!(results.len(), 0);

        // Filtering on new name should work.
        let results = col
            .search(&[0.5, 0.5, 0.0, 0.0], 10, Some("group = 'a'"))
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].pk, "doc-1");

        // Renaming non-existent field should fail.
        assert!(col.rename_field("nonexistent", "x").is_err());

        // Renaming to existing field should fail.
        assert!(col.rename_field("group", "tenant").is_err());
    }
}
