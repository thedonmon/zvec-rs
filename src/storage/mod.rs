//! Pluggable storage backends for zvec-rs.
//!
//! Storage layout (for persistent backends):
//! - `vectors` table: internal_id (u32) -> raw f32 bytes
//! - `metadata` table: internal_id (u32) -> JSON-encoded field map
//! - `id_map` table: external_id (string) -> internal_id (u32)
//! - `state` table: "entry_point", "max_level", "next_id" -> u64 values

mod redb_backend;
mod memory_backend;

pub use redb_backend::RedbBackend;
pub use memory_backend::InMemoryBackend;

use std::collections::HashMap;
use std::fmt;
use std::path::Path;

/// Error type for storage operations.
#[derive(Debug)]
pub enum StorageError {
    Redb(redb::Error),
    Database(redb::DatabaseError),
    Transaction(redb::TransactionError),
    Table(redb::TableError),
    Storage(redb::StorageError),
    Commit(redb::CommitError),
    Serde(serde_json::Error),
    Other(String),
}

impl From<redb::Error> for StorageError {
    fn from(e: redb::Error) -> Self {
        StorageError::Redb(e)
    }
}

impl From<redb::DatabaseError> for StorageError {
    fn from(e: redb::DatabaseError) -> Self {
        StorageError::Database(e)
    }
}

impl From<redb::TransactionError> for StorageError {
    fn from(e: redb::TransactionError) -> Self {
        StorageError::Transaction(e)
    }
}

impl From<redb::TableError> for StorageError {
    fn from(e: redb::TableError) -> Self {
        StorageError::Table(e)
    }
}

impl From<redb::StorageError> for StorageError {
    fn from(e: redb::StorageError) -> Self {
        StorageError::Storage(e)
    }
}

impl From<redb::CommitError> for StorageError {
    fn from(e: redb::CommitError) -> Self {
        StorageError::Commit(e)
    }
}

impl From<serde_json::Error> for StorageError {
    fn from(e: serde_json::Error) -> Self {
        StorageError::Serde(e)
    }
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StorageError::Redb(e) => write!(f, "redb error: {e}"),
            StorageError::Database(e) => write!(f, "database error: {e}"),
            StorageError::Transaction(e) => write!(f, "transaction error: {e}"),
            StorageError::Table(e) => write!(f, "table error: {e}"),
            StorageError::Storage(e) => write!(f, "storage error: {e}"),
            StorageError::Commit(e) => write!(f, "commit error: {e}"),
            StorageError::Serde(e) => write!(f, "serde error: {e}"),
            StorageError::Other(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for StorageError {}

/// Trait defining the storage backend interface.
///
/// All methods mirror the original `Storage` API so that backends are
/// drop-in replacements for each other.
pub trait StorageBackend {
    /// Store a vector with its metadata and connections.
    fn put_vector(
        &self,
        internal_id: u32,
        external_id: &str,
        vector: &[f32],
        fields: &HashMap<String, String>,
        connections: &[Vec<u32>],
    ) -> Result<(), StorageError>;

    /// Remove a vector by external ID. Returns the internal ID if found.
    fn remove_vector(&self, external_id: &str) -> Result<Option<u32>, StorageError>;

    /// Load a vector by internal ID.
    fn get_vector(&self, internal_id: u32) -> Result<Option<Vec<f32>>, StorageError>;

    /// Load metadata for a vector.
    fn get_metadata(
        &self,
        internal_id: u32,
    ) -> Result<Option<HashMap<String, String>>, StorageError>;

    /// Load connections for a node at a specific layer.
    fn get_connections(
        &self,
        internal_id: u32,
        level: u8,
    ) -> Result<Option<Vec<u32>>, StorageError>;

    /// Save index state (entry point, max level, next ID).
    fn save_state(
        &self,
        entry_point: u32,
        max_level: usize,
        next_id: u32,
    ) -> Result<(), StorageError>;

    /// Load index state.
    fn load_state(&self) -> Result<Option<(u32, usize, u32)>, StorageError>;

    /// Get all external ID -> internal ID mappings.
    fn load_id_map(&self) -> Result<HashMap<String, u32>, StorageError>;

    /// Save the schema JSON string.
    fn save_schema(&self, schema_json: &str) -> Result<(), StorageError>;

    /// Load the schema JSON string.
    fn load_schema(&self) -> Result<Option<String>, StorageError>;

    /// Store multiple vectors in a single transaction (batch).
    fn put_vectors_batch(
        &self,
        items: &[(u32, &str, &[f32], &HashMap<String, String>, &[Vec<u32>])],
    ) -> Result<(), StorageError>;

    /// Flush all pending writes to disk (no-op for in-memory backends).
    fn flush(&mut self) -> Result<(), StorageError>;

    /// Return the backend type name (for diagnostics).
    fn backend_type(&self) -> &'static str;

    /// Return the approximate size in bytes on disk, if applicable.
    fn disk_size(&self) -> Option<u64>;
}

/// The storage engine, wrapping a pluggable backend.
///
/// By default, uses `RedbBackend` for persistent storage. Use
/// `Storage::in_memory()` for a purely in-memory backend.
pub enum Storage {
    Redb(RedbBackend),
    InMemory(InMemoryBackend),
}

impl Storage {
    /// Open or create a persistent (redb) database at the given path.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StorageError> {
        Ok(Storage::Redb(RedbBackend::open(path)?))
    }

    /// Create a new in-memory storage backend.
    pub fn in_memory() -> Self {
        Storage::InMemory(InMemoryBackend::new())
    }

    /// Get a reference to the underlying backend as a trait object.
    fn backend(&self) -> &dyn StorageBackend {
        match self {
            Storage::Redb(b) => b,
            Storage::InMemory(b) => b,
        }
    }

    /// Get a mutable reference to the underlying backend.
    fn backend_mut(&mut self) -> &mut dyn StorageBackend {
        match self {
            Storage::Redb(b) => b,
            Storage::InMemory(b) => b,
        }
    }

    // --- Delegate all methods to the backend ---

    pub fn put_vector(
        &self,
        internal_id: u32,
        external_id: &str,
        vector: &[f32],
        fields: &HashMap<String, String>,
        connections: &[Vec<u32>],
    ) -> Result<(), StorageError> {
        self.backend()
            .put_vector(internal_id, external_id, vector, fields, connections)
    }

    pub fn remove_vector(&self, external_id: &str) -> Result<Option<u32>, StorageError> {
        self.backend().remove_vector(external_id)
    }

    pub fn get_vector(&self, internal_id: u32) -> Result<Option<Vec<f32>>, StorageError> {
        self.backend().get_vector(internal_id)
    }

    pub fn get_metadata(
        &self,
        internal_id: u32,
    ) -> Result<Option<HashMap<String, String>>, StorageError> {
        self.backend().get_metadata(internal_id)
    }

    pub fn get_connections(
        &self,
        internal_id: u32,
        level: u8,
    ) -> Result<Option<Vec<u32>>, StorageError> {
        self.backend().get_connections(internal_id, level)
    }

    pub fn save_state(
        &self,
        entry_point: u32,
        max_level: usize,
        next_id: u32,
    ) -> Result<(), StorageError> {
        self.backend().save_state(entry_point, max_level, next_id)
    }

    pub fn load_state(&self) -> Result<Option<(u32, usize, u32)>, StorageError> {
        self.backend().load_state()
    }

    pub fn load_id_map(&self) -> Result<HashMap<String, u32>, StorageError> {
        self.backend().load_id_map()
    }

    pub fn save_schema(&self, schema_json: &str) -> Result<(), StorageError> {
        self.backend().save_schema(schema_json)
    }

    pub fn load_schema(&self) -> Result<Option<String>, StorageError> {
        self.backend().load_schema()
    }

    pub fn put_vectors_batch(
        &self,
        items: &[(u32, &str, &[f32], &HashMap<String, String>, &[Vec<u32>])],
    ) -> Result<(), StorageError> {
        self.backend().put_vectors_batch(items)
    }

    pub fn flush(&mut self) -> Result<(), StorageError> {
        self.backend_mut().flush()
    }

    /// Return the backend type name.
    pub fn backend_type(&self) -> &'static str {
        self.backend().backend_type()
    }

    /// Return approximate disk size in bytes, if applicable.
    pub fn disk_size(&self) -> Option<u64> {
        self.backend().disk_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run a standard test suite against any storage backend.
    fn test_roundtrip_vector_on(storage: Storage) {
        let vector = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut fields = HashMap::new();
        fields.insert("title".to_string(), "hello".to_string());
        let connections = vec![vec![1, 2, 3], vec![4, 5]];

        storage
            .put_vector(0, "doc-1", &vector, &fields, &connections)
            .unwrap();

        let loaded = storage.get_vector(0).unwrap().unwrap();
        assert_eq!(loaded, vector);

        let meta = storage.get_metadata(0).unwrap().unwrap();
        assert_eq!(meta["title"], "hello");

        let conns = storage.get_connections(0, 0).unwrap().unwrap();
        assert_eq!(conns, vec![1, 2, 3]);

        let conns = storage.get_connections(0, 1).unwrap().unwrap();
        assert_eq!(conns, vec![4, 5]);
    }

    fn test_remove_vector_on(storage: Storage) {
        let fields = HashMap::new();
        storage
            .put_vector(0, "doc-1", &[1.0, 2.0], &fields, &[vec![]])
            .unwrap();

        let removed = storage.remove_vector("doc-1").unwrap();
        assert_eq!(removed, Some(0));

        assert!(storage.get_vector(0).unwrap().is_none());
        assert!(storage.remove_vector("doc-1").unwrap().is_none());
    }

    fn test_state_roundtrip_on(storage: Storage) {
        storage.save_state(42, 3, 100).unwrap();

        let (ep, ml, next) = storage.load_state().unwrap().unwrap();
        assert_eq!(ep, 42);
        assert_eq!(ml, 3);
        assert_eq!(next, 100);
    }

    fn test_id_map_on(storage: Storage) {
        let fields = HashMap::new();
        storage
            .put_vector(0, "doc-a", &[1.0], &fields, &[vec![]])
            .unwrap();
        storage
            .put_vector(1, "doc-b", &[2.0], &fields, &[vec![]])
            .unwrap();

        let map = storage.load_id_map().unwrap();
        assert_eq!(map.len(), 2);
        assert_eq!(map["doc-a"], 0);
        assert_eq!(map["doc-b"], 1);
    }

    fn test_schema_roundtrip_on(storage: Storage) {
        let schema_json = r#"[{"name":"category","type":"filtered"}]"#;
        storage.save_schema(schema_json).unwrap();
        let loaded = storage.load_schema().unwrap().unwrap();
        assert_eq!(loaded, schema_json);
    }

    fn test_batch_on(storage: Storage) {
        let fields1 = HashMap::new();
        let fields2 = HashMap::new();
        let conns1 = vec![vec![1]];
        let conns2 = vec![vec![0]];
        let items: Vec<(u32, &str, &[f32], &HashMap<String, String>, &[Vec<u32>])> = vec![
            (0, "a", &[1.0, 2.0], &fields1, &conns1),
            (1, "b", &[3.0, 4.0], &fields2, &conns2),
        ];
        storage.put_vectors_batch(&items).unwrap();

        assert_eq!(storage.get_vector(0).unwrap().unwrap(), vec![1.0, 2.0]);
        assert_eq!(storage.get_vector(1).unwrap().unwrap(), vec![3.0, 4.0]);
        let map = storage.load_id_map().unwrap();
        assert_eq!(map.len(), 2);
    }

    // --- Redb tests ---

    #[test]
    fn test_redb_roundtrip_vector() {
        let dir = tempfile::tempdir().unwrap();
        let storage = Storage::open(dir.path().join("test.redb")).unwrap();
        test_roundtrip_vector_on(storage);
    }

    #[test]
    fn test_redb_remove_vector() {
        let dir = tempfile::tempdir().unwrap();
        let storage = Storage::open(dir.path().join("test.redb")).unwrap();
        test_remove_vector_on(storage);
    }

    #[test]
    fn test_redb_state_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let storage = Storage::open(dir.path().join("test.redb")).unwrap();
        test_state_roundtrip_on(storage);
    }

    #[test]
    fn test_redb_id_map() {
        let dir = tempfile::tempdir().unwrap();
        let storage = Storage::open(dir.path().join("test.redb")).unwrap();
        test_id_map_on(storage);
    }

    #[test]
    fn test_redb_schema_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let storage = Storage::open(dir.path().join("test.redb")).unwrap();
        test_schema_roundtrip_on(storage);
    }

    #[test]
    fn test_redb_batch() {
        let dir = tempfile::tempdir().unwrap();
        let storage = Storage::open(dir.path().join("test.redb")).unwrap();
        test_batch_on(storage);
    }

    #[test]
    fn test_redb_persistence_across_opens() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.redb");

        {
            let storage = Storage::open(&path).unwrap();
            let fields = HashMap::new();
            storage
                .put_vector(0, "doc-1", &[1.0, 2.0, 3.0], &fields, &[vec![1]])
                .unwrap();
            storage.save_state(0, 1, 1).unwrap();
        }

        let storage = Storage::open(&path).unwrap();
        let v = storage.get_vector(0).unwrap().unwrap();
        assert_eq!(v, vec![1.0, 2.0, 3.0]);

        let (ep, ml, next) = storage.load_state().unwrap().unwrap();
        assert_eq!(ep, 0);
        assert_eq!(ml, 1);
        assert_eq!(next, 1);
    }

    #[test]
    fn test_redb_backend_type() {
        let dir = tempfile::tempdir().unwrap();
        let storage = Storage::open(dir.path().join("test.redb")).unwrap();
        assert_eq!(storage.backend_type(), "redb");
    }

    #[test]
    fn test_redb_disk_size() {
        let dir = tempfile::tempdir().unwrap();
        let storage = Storage::open(dir.path().join("test.redb")).unwrap();
        assert!(storage.disk_size().is_some());
    }

    // --- InMemory tests ---

    #[test]
    fn test_memory_roundtrip_vector() {
        test_roundtrip_vector_on(Storage::in_memory());
    }

    #[test]
    fn test_memory_remove_vector() {
        test_remove_vector_on(Storage::in_memory());
    }

    #[test]
    fn test_memory_state_roundtrip() {
        test_state_roundtrip_on(Storage::in_memory());
    }

    #[test]
    fn test_memory_id_map() {
        test_id_map_on(Storage::in_memory());
    }

    #[test]
    fn test_memory_schema_roundtrip() {
        test_schema_roundtrip_on(Storage::in_memory());
    }

    #[test]
    fn test_memory_batch() {
        test_batch_on(Storage::in_memory());
    }

    #[test]
    fn test_memory_backend_type() {
        let storage = Storage::in_memory();
        assert_eq!(storage.backend_type(), "memory");
    }

    #[test]
    fn test_memory_disk_size() {
        let storage = Storage::in_memory();
        assert!(storage.disk_size().is_none());
    }
}
