//! Persistent storage backed by redb (pure Rust embedded KV store).
//!
//! Storage layout:
//! - `vectors` table: internal_id (u32) -> raw f32 bytes
//! - `metadata` table: internal_id (u32) -> JSON-encoded field map
//! - `id_map` table: external_id (string) -> internal_id (u32)
//! - `state` table: "entry_point", "max_level", "next_id" -> u64 values

use std::path::Path;

use redb::{Database, ReadableTable, TableDefinition};

const VECTORS: TableDefinition<u32, &[u8]> = TableDefinition::new("vectors");
const METADATA: TableDefinition<u32, &str> = TableDefinition::new("metadata");
const ID_MAP: TableDefinition<&str, u32> = TableDefinition::new("id_map");
const CONNECTIONS: TableDefinition<(u32, u8), &[u8]> =
    TableDefinition::new("connections");
const STATE: TableDefinition<&str, u64> = TableDefinition::new("state");

/// Convert &[f32] to Vec<u8> (safe, no alignment issues).
fn f32_slice_to_bytes(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &val in data {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
}

/// Convert &[u8] back to Vec<f32> (safe, handles unaligned data).
fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Convert &[u32] to Vec<u8> (safe).
fn u32_slice_to_bytes(data: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &val in data {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
}

/// Convert &[u8] back to Vec<u32> (safe, handles unaligned data).
fn bytes_to_u32_vec(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Persistent storage engine using redb.
pub struct Storage {
    db: Database,
}

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

impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StorageError::Redb(e) => write!(f, "redb error: {e}"),
            StorageError::Database(e) => write!(f, "database error: {e}"),
            StorageError::Transaction(e) => write!(f, "transaction error: {e}"),
            StorageError::Table(e) => write!(f, "table error: {e}"),
            StorageError::Storage(e) => write!(f, "storage error: {e}"),
            StorageError::Commit(e) => write!(f, "commit error: {e}"),
            StorageError::Serde(e) => write!(f, "serde error: {e}"),
        }
    }
}

impl std::error::Error for StorageError {}

impl Storage {
    /// Open or create a database at the given path.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StorageError> {
        let db = Database::create(path.as_ref())?;

        // Create tables on first open
        let txn = db.begin_write()?;
        {
            let _ = txn.open_table(VECTORS)?;
            let _ = txn.open_table(METADATA)?;
            let _ = txn.open_table(ID_MAP)?;
            let _ = txn.open_table(CONNECTIONS)?;
            let _ = txn.open_table(STATE)?;
        }
        txn.commit()?;

        Ok(Self { db })
    }

    /// Store a vector with its metadata and connections.
    pub fn put_vector(
        &self,
        internal_id: u32,
        external_id: &str,
        vector: &[f32],
        fields: &std::collections::HashMap<String, String>,
        connections: &[Vec<u32>],
    ) -> Result<(), StorageError> {
        let txn = self.db.begin_write()?;
        {
            // Store vector as raw bytes (safe copy via bytemuck-style conversion)
            let vector_bytes = f32_slice_to_bytes(vector);
            let mut vectors_table = txn.open_table(VECTORS)?;
            vectors_table.insert(internal_id, vector_bytes.as_slice())?;

            // Store metadata as JSON
            let meta_json = serde_json::to_string(fields)?;
            let mut meta_table = txn.open_table(METADATA)?;
            meta_table.insert(internal_id, meta_json.as_str())?;

            // Store ID mapping
            let mut id_table = txn.open_table(ID_MAP)?;
            id_table.insert(external_id, internal_id)?;

            // Store connections per layer
            let mut conn_table = txn.open_table(CONNECTIONS)?;
            for (level, conns) in connections.iter().enumerate() {
                let conn_bytes = u32_slice_to_bytes(conns);
                conn_table.insert((internal_id, level as u8), conn_bytes.as_slice())?;
            }
        }
        txn.commit()?;
        Ok(())
    }

    /// Remove a vector by external ID.
    pub fn remove_vector(&self, external_id: &str) -> Result<Option<u32>, StorageError> {
        let txn = self.db.begin_write()?;
        let internal_id;
        {
            let mut id_table = txn.open_table(ID_MAP)?;
            internal_id = match id_table.remove(external_id)? {
                Some(v) => v.value(),
                None => return Ok(None),
            };

            let mut vectors_table = txn.open_table(VECTORS)?;
            vectors_table.remove(internal_id)?;

            let mut meta_table = txn.open_table(METADATA)?;
            meta_table.remove(internal_id)?;

            // Remove all connection layers
            let mut conn_table = txn.open_table(CONNECTIONS)?;
            for level in 0..=255u8 {
                if conn_table.remove((internal_id, level))?.is_none() {
                    break;
                }
            }
        }
        txn.commit()?;
        Ok(Some(internal_id))
    }

    /// Load a vector by internal ID.
    pub fn get_vector(&self, internal_id: u32) -> Result<Option<Vec<f32>>, StorageError> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(VECTORS)?;
        match table.get(internal_id)? {
            Some(bytes) => Ok(Some(bytes_to_f32_vec(bytes.value()))),
            None => Ok(None),
        }
    }

    /// Load metadata for a vector.
    pub fn get_metadata(
        &self,
        internal_id: u32,
    ) -> Result<Option<std::collections::HashMap<String, String>>, StorageError> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(METADATA)?;
        match table.get(internal_id)? {
            Some(json_str) => {
                let fields = serde_json::from_str(json_str.value())?;
                Ok(Some(fields))
            }
            None => Ok(None),
        }
    }

    /// Load connections for a node at a specific layer.
    pub fn get_connections(
        &self,
        internal_id: u32,
        level: u8,
    ) -> Result<Option<Vec<u32>>, StorageError> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(CONNECTIONS)?;
        match table.get((internal_id, level))? {
            Some(bytes) => Ok(Some(bytes_to_u32_vec(bytes.value()))),
            None => Ok(None),
        }
    }

    /// Save index state (entry point, max level, next ID).
    pub fn save_state(
        &self,
        entry_point: u32,
        max_level: usize,
        next_id: u32,
    ) -> Result<(), StorageError> {
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(STATE)?;
            table.insert("entry_point", entry_point as u64)?;
            table.insert("max_level", max_level as u64)?;
            table.insert("next_id", next_id as u64)?;
        }
        txn.commit()?;
        Ok(())
    }

    /// Load index state.
    pub fn load_state(&self) -> Result<Option<(u32, usize, u32)>, StorageError> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(STATE)?;

        let entry_point = match table.get("entry_point")? {
            Some(v) => v.value() as u32,
            None => return Ok(None),
        };
        let max_level = match table.get("max_level")? {
            Some(v) => v.value() as usize,
            None => return Ok(None),
        };
        let next_id = match table.get("next_id")? {
            Some(v) => v.value() as u32,
            None => return Ok(None),
        };

        Ok(Some((entry_point, max_level, next_id)))
    }

    /// Get all external ID -> internal ID mappings.
    pub fn load_id_map(&self) -> Result<std::collections::HashMap<String, u32>, StorageError> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(ID_MAP)?;
        let mut map = std::collections::HashMap::new();
        let iter = table.iter()?;
        for entry in iter {
            let entry = entry?;
            map.insert(entry.0.value().to_string(), entry.1.value());
        }
        Ok(map)
    }

    /// Save the schema JSON string.
    pub fn save_schema(&self, schema_json: &str) -> Result<(), StorageError> {
        let txn = self.db.begin_write()?;
        {
            let table = txn.open_table(STATE)?;
            // Store schema as a u64-keyed entry won't work since STATE is &str->u64.
            // Use METADATA table with a sentinel key instead.
            drop(table);
            let mut meta = txn.open_table(METADATA)?;
            // Use internal_id u32::MAX as sentinel for schema storage
            meta.insert(u32::MAX, schema_json)?;
        }
        txn.commit()?;
        Ok(())
    }

    /// Load the schema JSON string.
    pub fn load_schema(&self) -> Result<Option<String>, StorageError> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(METADATA)?;
        match table.get(u32::MAX)? {
            Some(v) => Ok(Some(v.value().to_string())),
            None => Ok(None),
        }
    }

    /// Store multiple vectors in a single transaction (batch).
    pub fn put_vectors_batch(
        &self,
        items: &[(u32, &str, &[f32], &std::collections::HashMap<String, String>, &[Vec<u32>])],
    ) -> Result<(), StorageError> {
        let txn = self.db.begin_write()?;
        {
            let mut vectors_table = txn.open_table(VECTORS)?;
            let mut meta_table = txn.open_table(METADATA)?;
            let mut id_table = txn.open_table(ID_MAP)?;
            let mut conn_table = txn.open_table(CONNECTIONS)?;

            for &(internal_id, external_id, vector, fields, connections) in items {
                let vector_bytes = f32_slice_to_bytes(vector);
                vectors_table.insert(internal_id, vector_bytes.as_slice())?;

                let meta_json = serde_json::to_string(fields)?;
                meta_table.insert(internal_id, meta_json.as_str())?;

                id_table.insert(external_id, internal_id)?;

                for (level, conns) in connections.iter().enumerate() {
                    let conn_bytes = u32_slice_to_bytes(conns);
                    conn_table.insert((internal_id, level as u8), conn_bytes.as_slice())?;
                }
            }
        }
        txn.commit()?;
        Ok(())
    }

    /// Flush all pending writes to disk.
    pub fn flush(&mut self) -> Result<(), StorageError> {
        // redb commits are durable by default, but we can compact
        self.db
            .compact()
            .map_err(|e| StorageError::Redb(redb::Error::from(e)))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_vector() {
        let dir = tempfile::tempdir().unwrap();
        let storage = Storage::open(dir.path().join("test.redb")).unwrap();

        let vector = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut fields = std::collections::HashMap::new();
        fields.insert("title".to_string(), "hello".to_string());
        let connections = vec![vec![1, 2, 3], vec![4, 5]];

        storage
            .put_vector(0, "doc-1", &vector, &fields, &connections)
            .unwrap();

        // Read back
        let loaded = storage.get_vector(0).unwrap().unwrap();
        assert_eq!(loaded, vector);

        let meta = storage.get_metadata(0).unwrap().unwrap();
        assert_eq!(meta["title"], "hello");

        let conns = storage.get_connections(0, 0).unwrap().unwrap();
        assert_eq!(conns, vec![1, 2, 3]);

        let conns = storage.get_connections(0, 1).unwrap().unwrap();
        assert_eq!(conns, vec![4, 5]);
    }

    #[test]
    fn test_remove_vector() {
        let dir = tempfile::tempdir().unwrap();
        let storage = Storage::open(dir.path().join("test.redb")).unwrap();

        let fields = std::collections::HashMap::new();
        storage
            .put_vector(0, "doc-1", &[1.0, 2.0], &fields, &[vec![]])
            .unwrap();

        let removed = storage.remove_vector("doc-1").unwrap();
        assert_eq!(removed, Some(0));

        assert!(storage.get_vector(0).unwrap().is_none());
        assert!(storage.remove_vector("doc-1").unwrap().is_none());
    }

    #[test]
    fn test_state_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let storage = Storage::open(dir.path().join("test.redb")).unwrap();

        storage.save_state(42, 3, 100).unwrap();

        let (ep, ml, next) = storage.load_state().unwrap().unwrap();
        assert_eq!(ep, 42);
        assert_eq!(ml, 3);
        assert_eq!(next, 100);
    }

    #[test]
    fn test_id_map() {
        let dir = tempfile::tempdir().unwrap();
        let storage = Storage::open(dir.path().join("test.redb")).unwrap();

        let fields = std::collections::HashMap::new();
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

    #[test]
    fn test_persistence_across_opens() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.redb");

        {
            let storage = Storage::open(&path).unwrap();
            let fields = std::collections::HashMap::new();
            storage
                .put_vector(0, "doc-1", &[1.0, 2.0, 3.0], &fields, &[vec![1]])
                .unwrap();
            storage.save_state(0, 1, 1).unwrap();
        }

        // Reopen
        let storage = Storage::open(&path).unwrap();
        let v = storage.get_vector(0).unwrap().unwrap();
        assert_eq!(v, vec![1.0, 2.0, 3.0]);

        let (ep, ml, next) = storage.load_state().unwrap().unwrap();
        assert_eq!(ep, 0);
        assert_eq!(ml, 1);
        assert_eq!(next, 1);
    }
}
