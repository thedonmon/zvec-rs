//! SQL-like filter expressions for vector search results.
//!
//! Supports: `field = 'value'`, `field != 'value'`, `field IN ('a', 'b')`,
//! `AND`, `OR`, `NOT`, and parenthesized grouping.
//!
//! Example: `category = 'science' AND tenant_id = 'org-42'`

mod parser;

pub use parser::{parse_filter, FilterExpr};

/// Evaluate a filter expression against a set of fields.
pub fn matches(
    expr: &FilterExpr,
    fields: &std::collections::HashMap<String, String>,
) -> bool {
    match expr {
        FilterExpr::Eq(field, value) => {
            fields.get(field).map_or(false, |v| v == value)
        }
        FilterExpr::Ne(field, value) => {
            fields.get(field).map_or(true, |v| v != value)
        }
        FilterExpr::In(field, values) => {
            fields.get(field).map_or(false, |v| values.contains(v))
        }
        FilterExpr::Contains(field, value) => {
            // For tags fields (comma-separated): check if any tag matches
            fields.get(field).map_or(false, |v| {
                v.split(',').any(|tag| tag.trim() == value)
            })
        }
        FilterExpr::And(left, right) => {
            matches(left, fields) && matches(right, fields)
        }
        FilterExpr::Or(left, right) => {
            matches(left, fields) || matches(right, fields)
        }
        FilterExpr::Not(inner) => !matches(inner, fields),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn fields(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn test_eq() {
        let expr = parse_filter("category = 'science'").unwrap();
        assert!(matches(&expr, &fields(&[("category", "science")])));
        assert!(!matches(&expr, &fields(&[("category", "math")])));
        assert!(!matches(&expr, &fields(&[])));
    }

    #[test]
    fn test_ne() {
        let expr = parse_filter("status != 'deleted'").unwrap();
        assert!(matches(&expr, &fields(&[("status", "active")])));
        assert!(!matches(&expr, &fields(&[("status", "deleted")])));
        assert!(matches(&expr, &fields(&[]))); // missing field != 'deleted' is true
    }

    #[test]
    fn test_and() {
        let expr = parse_filter("a = '1' AND b = '2'").unwrap();
        assert!(matches(&expr, &fields(&[("a", "1"), ("b", "2")])));
        assert!(!matches(&expr, &fields(&[("a", "1"), ("b", "3")])));
    }

    #[test]
    fn test_or() {
        let expr = parse_filter("a = '1' OR b = '2'").unwrap();
        assert!(matches(&expr, &fields(&[("a", "1")])));
        assert!(matches(&expr, &fields(&[("b", "2")])));
        assert!(!matches(&expr, &fields(&[("a", "2"), ("b", "3")])));
    }

    #[test]
    fn test_in() {
        let expr = parse_filter("color IN ('red', 'blue', 'green')").unwrap();
        assert!(matches(&expr, &fields(&[("color", "red")])));
        assert!(matches(&expr, &fields(&[("color", "blue")])));
        assert!(!matches(&expr, &fields(&[("color", "yellow")])));
    }

    #[test]
    fn test_contains() {
        let expr = parse_filter("tags CONTAINS 'rust'").unwrap();
        assert!(matches(
            &expr,
            &fields(&[("tags", "rust, elixir, python")])
        ));
        assert!(!matches(&expr, &fields(&[("tags", "go, python")])));
    }

    #[test]
    fn test_complex() {
        let expr =
            parse_filter("(category = 'science' OR category = 'math') AND tenant = 'org-1'")
                .unwrap();
        assert!(matches(
            &expr,
            &fields(&[("category", "science"), ("tenant", "org-1")])
        ));
        assert!(!matches(
            &expr,
            &fields(&[("category", "science"), ("tenant", "org-2")])
        ));
        assert!(!matches(
            &expr,
            &fields(&[("category", "art"), ("tenant", "org-1")])
        ));
    }

    #[test]
    fn test_not() {
        let expr = parse_filter("NOT category = 'deleted'").unwrap();
        assert!(matches(&expr, &fields(&[("category", "active")])));
        assert!(!matches(&expr, &fields(&[("category", "deleted")])));
    }
}
