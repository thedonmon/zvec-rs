//! Schema definitions for collection fields.
//!
//! Field types control how values are stored and indexed:
//! - `String` — stored only, returned in search results, not indexed
//! - `Filtered` — stored + inverted index for exact-match filtering
//! - `Tags` — stored as comma-separated values + inverted index per tag

use std::collections::HashMap;

/// The type of a field in the schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FieldType {
    /// Stored string field. Returned in results but not indexed for filtering.
    String,
    /// Filtered field. Stored and indexed with an inverted index for fast
    /// exact-match queries (`field = 'value'`, `field IN (...)`, etc.).
    Filtered,
    /// Tags field. Stored as comma-separated values. Each individual tag is
    /// indexed in the inverted index for `CONTAINS` queries.
    Tags,
}

/// Schema defining the fields of a collection.
#[derive(Debug, Clone)]
pub struct FieldSchema {
    fields: Vec<(String, FieldType)>,
}

impl FieldSchema {
    /// Create a new schema from a list of (name, type) pairs.
    pub fn new(fields: Vec<(String, FieldType)>) -> Self {
        Self { fields }
    }

    /// Create a schema with no fields (all fields stored as untyped strings).
    pub fn empty() -> Self {
        Self { fields: Vec::new() }
    }

    /// Parse a schema from JSON.
    ///
    /// Expected format: `[{"name": "category", "type": "filtered"}, ...]`
    pub fn from_json(json: &str) -> Result<Self, String> {
        if json.is_empty() || json == "[]" {
            return Ok(Self::empty());
        }

        let entries: Vec<HashMap<String, String>> =
            serde_json::from_str(json).map_err(|e| format!("schema JSON parse error: {}", e))?;

        let mut fields = Vec::with_capacity(entries.len());
        for entry in entries {
            let name = entry
                .get("name")
                .ok_or("schema field missing 'name'")?
                .clone();
            let type_str = entry
                .get("type")
                .ok_or("schema field missing 'type'")?
                .as_str();
            let field_type = match type_str {
                "string" => FieldType::String,
                "filtered" => FieldType::Filtered,
                "tags" => FieldType::Tags,
                other => return Err(format!("unknown field type: '{}'", other)),
            };
            fields.push((name, field_type));
        }

        Ok(Self { fields })
    }

    /// Get the type of a field by name.
    pub fn field_type(&self, name: &str) -> Option<FieldType> {
        self.fields
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, t)| *t)
    }

    /// Get all field names that should be indexed (Filtered + Tags).
    pub fn indexed_fields(&self) -> Vec<(&str, FieldType)> {
        self.fields
            .iter()
            .filter(|(_, t)| matches!(t, FieldType::Filtered | FieldType::Tags))
            .map(|(n, t)| (n.as_str(), *t))
            .collect()
    }

    /// Whether a field is indexed.
    pub fn is_indexed(&self, name: &str) -> bool {
        self.field_type(name)
            .map_or(false, |t| matches!(t, FieldType::Filtered | FieldType::Tags))
    }

    /// Whether the schema has any indexed fields.
    pub fn has_indexed_fields(&self) -> bool {
        self.fields
            .iter()
            .any(|(_, t)| matches!(t, FieldType::Filtered | FieldType::Tags))
    }

    /// Get all field definitions.
    pub fn fields(&self) -> &[(String, FieldType)] {
        &self.fields
    }
}

impl Default for FieldSchema {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_json() {
        let json = r#"[
            {"name": "content", "type": "string"},
            {"name": "category", "type": "filtered"},
            {"name": "labels", "type": "tags"}
        ]"#;
        let schema = FieldSchema::from_json(json).unwrap();
        assert_eq!(schema.field_type("content"), Some(FieldType::String));
        assert_eq!(schema.field_type("category"), Some(FieldType::Filtered));
        assert_eq!(schema.field_type("labels"), Some(FieldType::Tags));
        assert_eq!(schema.field_type("nonexistent"), None);
    }

    #[test]
    fn test_indexed_fields() {
        let schema = FieldSchema::new(vec![
            ("content".into(), FieldType::String),
            ("category".into(), FieldType::Filtered),
            ("labels".into(), FieldType::Tags),
        ]);
        let indexed = schema.indexed_fields();
        assert_eq!(indexed.len(), 2);
        assert!(indexed.iter().any(|(n, _)| *n == "category"));
        assert!(indexed.iter().any(|(n, _)| *n == "labels"));
    }

    #[test]
    fn test_empty_json() {
        let schema = FieldSchema::from_json("").unwrap();
        assert!(!schema.has_indexed_fields());
        let schema = FieldSchema::from_json("[]").unwrap();
        assert!(!schema.has_indexed_fields());
    }

    #[test]
    fn test_invalid_type() {
        let json = r#"[{"name": "x", "type": "bogus"}]"#;
        assert!(FieldSchema::from_json(json).is_err());
    }
}
