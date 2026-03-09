//! In-memory storage backend using HashMaps.
//!
//! Useful for testing and ephemeral collections that don't need persistence.

use std::collections::HashMap;
use std::sync::RwLock;

use super::{StorageBackend, StorageError};

/// Purely in-memory storage backend. No disk I/O.
pub struct InMemoryBackend {
    /// internal_id -> vector data
    vectors: RwLock<HashMap<u32, Vec<f32>>>,
    /// internal_id -> metadata fields
    metadata: RwLock<HashMap<u32, HashMap<String, String>>>,
    /// external_id -> internal_id
    id_map: RwLock<HashMap<String, u32>>,
    /// (internal_id, level) -> connections
    connections: RwLock<HashMap<(u32, u8), Vec<u32>>>,
    /// state: (entry_point, max_level, next_id)
    state: RwLock<Option<(u32, usize, u32)>>,
    /// schema JSON
    schema: RwLock<Option<String>>,
}

impl InMemoryBackend {
    /// Create a new empty in-memory backend.
    pub fn new() -> Self {
        Self {
            vectors: RwLock::new(HashMap::new()),
            metadata: RwLock::new(HashMap::new()),
            id_map: RwLock::new(HashMap::new()),
            connections: RwLock::new(HashMap::new()),
            state: RwLock::new(None),
            schema: RwLock::new(None),
        }
    }
}

impl Default for InMemoryBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl StorageBackend for InMemoryBackend {
    fn put_vector(
        &self,
        internal_id: u32,
        external_id: &str,
        vector: &[f32],
        fields: &HashMap<String, String>,
        conns: &[Vec<u32>],
    ) -> Result<(), StorageError> {
        self.vectors
            .write()
            .unwrap()
            .insert(internal_id, vector.to_vec());
        self.metadata
            .write()
            .unwrap()
            .insert(internal_id, fields.clone());
        self.id_map
            .write()
            .unwrap()
            .insert(external_id.to_string(), internal_id);

        let mut conn_map = self.connections.write().unwrap();
        for (level, c) in conns.iter().enumerate() {
            conn_map.insert((internal_id, level as u8), c.clone());
        }
        Ok(())
    }

    fn remove_vector(&self, external_id: &str) -> Result<Option<u32>, StorageError> {
        let internal_id = match self.id_map.write().unwrap().remove(external_id) {
            Some(id) => id,
            None => return Ok(None),
        };

        self.vectors.write().unwrap().remove(&internal_id);
        self.metadata.write().unwrap().remove(&internal_id);

        let mut conn_map = self.connections.write().unwrap();
        for level in 0..=255u8 {
            if conn_map.remove(&(internal_id, level)).is_none() {
                break;
            }
        }

        Ok(Some(internal_id))
    }

    fn get_vector(&self, internal_id: u32) -> Result<Option<Vec<f32>>, StorageError> {
        Ok(self.vectors.read().unwrap().get(&internal_id).cloned())
    }

    fn get_metadata(
        &self,
        internal_id: u32,
    ) -> Result<Option<HashMap<String, String>>, StorageError> {
        Ok(self.metadata.read().unwrap().get(&internal_id).cloned())
    }

    fn get_connections(
        &self,
        internal_id: u32,
        level: u8,
    ) -> Result<Option<Vec<u32>>, StorageError> {
        Ok(self
            .connections
            .read()
            .unwrap()
            .get(&(internal_id, level))
            .cloned())
    }

    fn save_state(
        &self,
        entry_point: u32,
        max_level: usize,
        next_id: u32,
    ) -> Result<(), StorageError> {
        *self.state.write().unwrap() = Some((entry_point, max_level, next_id));
        Ok(())
    }

    fn load_state(&self) -> Result<Option<(u32, usize, u32)>, StorageError> {
        Ok(*self.state.read().unwrap())
    }

    fn load_id_map(&self) -> Result<HashMap<String, u32>, StorageError> {
        Ok(self.id_map.read().unwrap().clone())
    }

    fn save_schema(&self, schema_json: &str) -> Result<(), StorageError> {
        *self.schema.write().unwrap() = Some(schema_json.to_string());
        Ok(())
    }

    fn load_schema(&self) -> Result<Option<String>, StorageError> {
        Ok(self.schema.read().unwrap().clone())
    }

    fn put_vectors_batch(
        &self,
        items: &[(u32, &str, &[f32], &HashMap<String, String>, &[Vec<u32>])],
    ) -> Result<(), StorageError> {
        let mut vectors = self.vectors.write().unwrap();
        let mut metadata = self.metadata.write().unwrap();
        let mut id_map = self.id_map.write().unwrap();
        let mut conn_map = self.connections.write().unwrap();

        for &(internal_id, external_id, vector, fields, conns) in items {
            vectors.insert(internal_id, vector.to_vec());
            metadata.insert(internal_id, fields.clone());
            id_map.insert(external_id.to_string(), internal_id);
            for (level, c) in conns.iter().enumerate() {
                conn_map.insert((internal_id, level as u8), c.clone());
            }
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<(), StorageError> {
        // No-op for in-memory backend
        Ok(())
    }

    fn backend_type(&self) -> &'static str {
        "memory"
    }

    fn disk_size(&self) -> Option<u64> {
        None
    }
}
