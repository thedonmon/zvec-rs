//! Persistent storage backed by redb (pure Rust embedded KV store).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use redb::{Database, ReadableTable, TableDefinition};

use super::{StorageBackend, StorageError};

const VECTORS: TableDefinition<u32, &[u8]> = TableDefinition::new("vectors");
const METADATA: TableDefinition<u32, &str> = TableDefinition::new("metadata");
const ID_MAP: TableDefinition<&str, u32> = TableDefinition::new("id_map");
const CONNECTIONS: TableDefinition<(u32, u8), &[u8]> = TableDefinition::new("connections");
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
pub struct RedbBackend {
    db: Database,
    path: PathBuf,
}

impl RedbBackend {
    /// Open or create a database at the given path.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StorageError> {
        let path = path.as_ref().to_path_buf();
        let db = Database::create(&path)?;

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

        Ok(Self { db, path })
    }
}

impl StorageBackend for RedbBackend {
    fn put_vector(
        &self,
        internal_id: u32,
        external_id: &str,
        vector: &[f32],
        fields: &HashMap<String, String>,
        connections: &[Vec<u32>],
    ) -> Result<(), StorageError> {
        let txn = self.db.begin_write()?;
        {
            let vector_bytes = f32_slice_to_bytes(vector);
            let mut vectors_table = txn.open_table(VECTORS)?;
            vectors_table.insert(internal_id, vector_bytes.as_slice())?;

            let meta_json = serde_json::to_string(fields)?;
            let mut meta_table = txn.open_table(METADATA)?;
            meta_table.insert(internal_id, meta_json.as_str())?;

            let mut id_table = txn.open_table(ID_MAP)?;
            id_table.insert(external_id, internal_id)?;

            let mut conn_table = txn.open_table(CONNECTIONS)?;
            for (level, conns) in connections.iter().enumerate() {
                let conn_bytes = u32_slice_to_bytes(conns);
                conn_table.insert((internal_id, level as u8), conn_bytes.as_slice())?;
            }
        }
        txn.commit()?;
        Ok(())
    }

    fn remove_vector(&self, external_id: &str) -> Result<Option<u32>, StorageError> {
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

    fn get_vector(&self, internal_id: u32) -> Result<Option<Vec<f32>>, StorageError> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(VECTORS)?;
        match table.get(internal_id)? {
            Some(bytes) => Ok(Some(bytes_to_f32_vec(bytes.value()))),
            None => Ok(None),
        }
    }

    fn get_metadata(
        &self,
        internal_id: u32,
    ) -> Result<Option<HashMap<String, String>>, StorageError> {
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

    fn get_connections(
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

    fn save_state(
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

    fn load_state(&self) -> Result<Option<(u32, usize, u32)>, StorageError> {
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

    fn load_id_map(&self) -> Result<HashMap<String, u32>, StorageError> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(ID_MAP)?;
        let mut map = HashMap::new();
        let iter = table.iter()?;
        for entry in iter {
            let entry = entry?;
            map.insert(entry.0.value().to_string(), entry.1.value());
        }
        Ok(map)
    }

    fn save_schema(&self, schema_json: &str) -> Result<(), StorageError> {
        let txn = self.db.begin_write()?;
        {
            let mut meta = txn.open_table(METADATA)?;
            meta.insert(u32::MAX, schema_json)?;
        }
        txn.commit()?;
        Ok(())
    }

    fn load_schema(&self) -> Result<Option<String>, StorageError> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(METADATA)?;
        match table.get(u32::MAX)? {
            Some(v) => Ok(Some(v.value().to_string())),
            None => Ok(None),
        }
    }

    fn put_vectors_batch(
        &self,
        items: &[(u32, &str, &[f32], &HashMap<String, String>, &[Vec<u32>])],
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

    fn flush(&mut self) -> Result<(), StorageError> {
        self.db
            .compact()
            .map_err(|e| StorageError::Redb(redb::Error::from(e)))?;
        Ok(())
    }

    fn backend_type(&self) -> &'static str {
        "redb"
    }

    fn disk_size(&self) -> Option<u64> {
        std::fs::metadata(&self.path).ok().map(|m| m.len())
    }
}
