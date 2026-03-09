//! SQL-like query engine for Collections.
//!
//! Supports SELECT, COUNT(*), and GROUP BY queries with WHERE filters,
//! ORDER BY (by field or `_distance`), and LIMIT.
//!
//! # Examples
//! ```text
//! SELECT field1, field2 FROM collection WHERE category = 'news' ORDER BY _distance LIMIT 10
//! SELECT COUNT(*) FROM collection WHERE category = 'news'
//! SELECT category, COUNT(*) FROM collection GROUP BY category WHERE tenant = 'acme'
//! ```

use std::collections::HashMap;

use crate::collection::Collection;
use crate::filter::{self, parse_filter, FilterExpr};

// ---------------------------------------------------------------------------
// Query types
// ---------------------------------------------------------------------------

/// The kind of query to execute.
#[derive(Debug, Clone, PartialEq)]
enum QueryKind {
    /// SELECT field1, field2 ... (or SELECT * for all fields)
    Select,
    /// SELECT COUNT(*)
    Count,
    /// SELECT field, COUNT(*) ... GROUP BY field
    GroupCount(String),
}

/// Sort order for results.
#[derive(Debug, Clone, PartialEq)]
pub enum OrderBy {
    /// Sort by vector distance (requires a query vector).
    Distance,
    /// Sort by a field value (lexicographic).
    Field(String),
}

/// A parsed SQL-like query.
#[derive(Debug, Clone)]
pub struct Query {
    kind: QueryKind,
    /// Fields to return (empty = all fields for Select, ignored for Count/GroupCount).
    fields: Vec<String>,
    /// Optional WHERE filter expression.
    filter: Option<FilterExpr>,
    /// Optional ORDER BY clause.
    order_by: Option<OrderBy>,
    /// Maximum number of results (0 = unlimited).
    limit: usize,
}

/// Result of executing a query.
#[derive(Debug, Clone, PartialEq)]
pub enum QueryResult {
    /// Rows of field -> value maps.
    Rows(Vec<HashMap<String, String>>),
    /// A count.
    Count(usize),
    /// Grouped counts: (group_key, count) pairs.
    GroupCount(Vec<(String, usize)>),
}

// ---------------------------------------------------------------------------
// Builder API
// ---------------------------------------------------------------------------

impl Query {
    /// Start building a SELECT query with the given fields.
    /// Pass `&["*"]` to select all fields.
    pub fn select(fields: &[&str]) -> Self {
        Self {
            kind: QueryKind::Select,
            fields: fields.iter().map(|s| s.to_string()).collect(),
            filter: None,
            order_by: None,
            limit: 0,
        }
    }

    /// Create a COUNT(*) query.
    pub fn count() -> Self {
        Self {
            kind: QueryKind::Count,
            fields: Vec::new(),
            filter: None,
            order_by: None,
            limit: 0,
        }
    }

    /// Create a GROUP BY query: SELECT field, COUNT(*) ... GROUP BY field.
    pub fn group_count(field: &str) -> Self {
        Self {
            kind: QueryKind::GroupCount(field.to_string()),
            fields: vec![field.to_string()],
            filter: None,
            order_by: None,
            limit: 0,
        }
    }

    /// Add a WHERE filter from a filter expression string.
    pub fn where_filter(mut self, expr: &str) -> Result<Self, String> {
        let parsed = parse_filter(expr).map_err(|e| e.to_string())?;
        self.filter = Some(parsed);
        Ok(self)
    }

    /// Add an ORDER BY clause.
    pub fn order_by(mut self, field: &str) -> Self {
        self.order_by = Some(if field == "_distance" {
            OrderBy::Distance
        } else {
            OrderBy::Field(field.to_string())
        });
        self
    }

    /// Set the maximum number of results.
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = n;
        self
    }

    // ------------------------------------------------------------------
    // SQL Parser
    // ------------------------------------------------------------------

    /// Parse a SQL string into a Query.
    ///
    /// Supported forms:
    /// - `SELECT f1, f2 FROM <name> [WHERE ...] [ORDER BY ...] [LIMIT n]`
    /// - `SELECT COUNT(*) FROM <name> [WHERE ...]`
    /// - `SELECT f, COUNT(*) FROM <name> GROUP BY f [WHERE ...]`
    pub fn parse(sql: &str) -> Result<Self, String> {
        let tokens = sql_tokenize(sql);
        if tokens.is_empty() {
            return Err("empty query".to_string());
        }

        let mut pos = 0;

        // Expect SELECT
        expect_keyword(&tokens, &mut pos, "SELECT")?;

        // Detect COUNT(*)
        if pos + 2 < tokens.len()
            && tokens[pos].eq_ignore_ascii_case("COUNT")
            && tokens[pos + 1] == "("
            && tokens[pos + 2] == "*"
        {
            // SELECT COUNT(*) FROM ...
            pos += 3;
            expect_token(&tokens, &mut pos, ")")?;
            expect_keyword(&tokens, &mut pos, "FROM")?;
            // Skip collection name
            if pos >= tokens.len() {
                return Err("expected collection name after FROM".to_string());
            }
            pos += 1; // skip collection name

            let filter = parse_optional_where(&tokens, &mut pos)?;

            return Ok(Query {
                kind: QueryKind::Count,
                fields: Vec::new(),
                filter,
                order_by: None,
                limit: 0,
            });
        }

        // Parse field list
        let mut fields = Vec::new();
        loop {
            if pos >= tokens.len() {
                return Err("unexpected end of query in field list".to_string());
            }

            // Check for "field, COUNT(*)" pattern (group by)
            if pos + 4 < tokens.len()
                && tokens[pos + 1] == ","
                && tokens[pos + 2].eq_ignore_ascii_case("COUNT")
                && tokens[pos + 3] == "("
                && tokens[pos + 4] == "*"
            {
                let group_field = tokens[pos].clone();
                pos += 5;
                expect_token(&tokens, &mut pos, ")")?;
                expect_keyword(&tokens, &mut pos, "FROM")?;
                // skip collection name
                if pos >= tokens.len() {
                    return Err("expected collection name after FROM".to_string());
                }
                pos += 1;

                // Expect GROUP BY
                expect_keyword(&tokens, &mut pos, "GROUP")?;
                expect_keyword(&tokens, &mut pos, "BY")?;
                if pos >= tokens.len() {
                    return Err("expected field name after GROUP BY".to_string());
                }
                let gb_field = tokens[pos].clone();
                pos += 1;

                if gb_field != group_field {
                    return Err(format!(
                        "GROUP BY field '{}' does not match SELECT field '{}'",
                        gb_field, group_field
                    ));
                }

                let filter = parse_optional_where(&tokens, &mut pos)?;

                return Ok(Query {
                    kind: QueryKind::GroupCount(group_field),
                    fields: Vec::new(),
                    filter,
                    order_by: None,
                    limit: 0,
                });
            }

            fields.push(tokens[pos].clone());
            pos += 1;

            if pos < tokens.len() && tokens[pos] == "," {
                pos += 1; // skip comma
            } else {
                break;
            }
        }

        // Expect FROM
        expect_keyword(&tokens, &mut pos, "FROM")?;
        // Skip collection name
        if pos >= tokens.len() {
            return Err("expected collection name after FROM".to_string());
        }
        pos += 1;

        // Optional WHERE
        let filter = parse_optional_where(&tokens, &mut pos)?;

        // Optional ORDER BY
        let order_by = if pos < tokens.len() && tokens[pos].eq_ignore_ascii_case("ORDER") {
            pos += 1;
            expect_keyword(&tokens, &mut pos, "BY")?;
            if pos >= tokens.len() {
                return Err("expected field after ORDER BY".to_string());
            }
            let field = tokens[pos].clone();
            pos += 1;
            Some(if field == "_distance" {
                OrderBy::Distance
            } else {
                OrderBy::Field(field)
            })
        } else {
            None
        };

        // Optional LIMIT
        let limit = if pos < tokens.len() && tokens[pos].eq_ignore_ascii_case("LIMIT") {
            pos += 1;
            if pos >= tokens.len() {
                return Err("expected number after LIMIT".to_string());
            }
            let n: usize = tokens[pos]
                .parse()
                .map_err(|_| format!("invalid LIMIT value: '{}'", tokens[pos]))?;
            pos += 1;
            n
        } else {
            0
        };

        let _ = pos; // suppress unused warning

        Ok(Query {
            kind: QueryKind::Select,
            fields,
            filter,
            order_by,
            limit,
        })
    }
}

// ---------------------------------------------------------------------------
// SQL tokenizer helpers
// ---------------------------------------------------------------------------

/// Simple tokenizer that splits SQL into words, respecting quoted strings
/// and punctuation tokens `(`, `)`, `,`, `*`.
fn sql_tokenize(input: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        match chars[i] {
            ' ' | '\t' | '\n' | '\r' => i += 1,
            '(' | ')' | ',' | '*' => {
                tokens.push(chars[i].to_string());
                i += 1;
            }
            '!' if i + 1 < chars.len() && chars[i + 1] == '=' => {
                tokens.push("!=".to_string());
                i += 2;
            }
            '<' if i + 1 < chars.len() && chars[i + 1] == '=' => {
                tokens.push("<=".to_string());
                i += 2;
            }
            '>' if i + 1 < chars.len() && chars[i + 1] == '=' => {
                tokens.push(">=".to_string());
                i += 2;
            }
            '=' | '<' | '>' => {
                tokens.push(chars[i].to_string());
                i += 1;
            }
            '\'' | '"' => {
                let quote = chars[i];
                i += 1;
                let start = i;
                while i < chars.len() && chars[i] != quote {
                    i += 1;
                }
                let s: String = chars[start..i].iter().collect();
                // Keep the quotes so the WHERE clause re-parser sees them
                tokens.push(format!("{}{}{}", quote, s, quote));
                if i < chars.len() {
                    i += 1; // skip closing quote
                }
            }
            c if c.is_alphanumeric() || c == '_' => {
                let start = i;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                tokens.push(word);
            }
            _ => i += 1, // skip unknown
        }
    }

    tokens
}

fn expect_keyword(tokens: &[String], pos: &mut usize, kw: &str) -> Result<(), String> {
    if *pos >= tokens.len() {
        return Err(format!("expected '{}' but reached end of query", kw));
    }
    if !tokens[*pos].eq_ignore_ascii_case(kw) {
        return Err(format!(
            "expected '{}', got '{}'",
            kw, tokens[*pos]
        ));
    }
    *pos += 1;
    Ok(())
}

fn expect_token(tokens: &[String], pos: &mut usize, tok: &str) -> Result<(), String> {
    if *pos >= tokens.len() {
        return Err(format!("expected '{}' but reached end of query", tok));
    }
    if tokens[*pos] != tok {
        return Err(format!("expected '{}', got '{}'", tok, tokens[*pos]));
    }
    *pos += 1;
    Ok(())
}

/// Parse an optional WHERE clause by reconstructing the filter substring
/// from tokens and delegating to the existing filter parser.
fn parse_optional_where(tokens: &[String], pos: &mut usize) -> Result<Option<FilterExpr>, String> {
    if *pos >= tokens.len() || !tokens[*pos].eq_ignore_ascii_case("WHERE") {
        return Ok(None);
    }
    *pos += 1; // skip WHERE

    // Collect tokens until we hit ORDER, LIMIT, GROUP, or end
    let mut filter_tokens = Vec::new();
    while *pos < tokens.len() {
        let upper = tokens[*pos].to_uppercase();
        if upper == "ORDER" || upper == "LIMIT" || upper == "GROUP" {
            break;
        }
        filter_tokens.push(tokens[*pos].clone());
        *pos += 1;
    }

    if filter_tokens.is_empty() {
        return Err("expected filter expression after WHERE".to_string());
    }

    let filter_str = filter_tokens.join(" ");
    let parsed = parse_filter(&filter_str).map_err(|e| format!("WHERE parse error: {}", e))?;
    Ok(Some(parsed))
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

impl Collection {
    /// Execute a query against this collection.
    ///
    /// - For `ORDER BY _distance`, a query `vector` must be provided.
    /// - For `SELECT` queries without `ORDER BY _distance`, the vector is optional.
    /// - For `COUNT(*)` and `GROUP BY` queries, the vector is ignored.
    pub fn execute_query(
        &self,
        query: &Query,
        vector: Option<&[f32]>,
    ) -> Result<QueryResult, String> {
        match &query.kind {
            QueryKind::Count => self.exec_count(query),
            QueryKind::GroupCount(field) => self.exec_group_count(field.clone(), query),
            QueryKind::Select => self.exec_select(query, vector),
        }
    }

    fn exec_count(&self, query: &Query) -> Result<QueryResult, String> {
        let count = self.matching_docs(&query.filter)?.len();
        Ok(QueryResult::Count(count))
    }

    fn exec_group_count(&self, field: String, query: &Query) -> Result<QueryResult, String> {
        let docs = self.matching_docs(&query.filter)?;

        let mut groups: HashMap<String, usize> = HashMap::new();
        for (_, fields) in &docs {
            if let Some(value) = fields.get(&field) {
                *groups.entry(value.clone()).or_default() += 1;
            }
        }

        let mut result: Vec<(String, usize)> = groups.into_iter().collect();
        result.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        Ok(QueryResult::GroupCount(result))
    }

    fn exec_select(
        &self,
        query: &Query,
        vector: Option<&[f32]>,
    ) -> Result<QueryResult, String> {
        // If ORDER BY _distance, we need vector search
        if query.order_by == Some(OrderBy::Distance) {
            let vec = vector.ok_or("ORDER BY _distance requires a query vector")?;
            let limit = if query.limit > 0 {
                query.limit
            } else {
                self.doc_count()
            };
            let filter_str = query.filter.as_ref().map(|f| filter_to_string(f));
            let hits = self.search(vec, limit, filter_str.as_deref())?;

            let select_all = query.fields.is_empty()
                || (query.fields.len() == 1 && query.fields[0] == "*");

            let rows: Vec<HashMap<String, String>> = hits
                .into_iter()
                .map(|hit| {
                    let mut row = if select_all {
                        hit.fields
                    } else {
                        let mut m = HashMap::new();
                        for f in &query.fields {
                            if let Some(v) = hit.fields.get(f) {
                                m.insert(f.clone(), v.clone());
                            }
                        }
                        m
                    };
                    row.insert("_pk".to_string(), hit.pk);
                    row.insert("_distance".to_string(), hit.score.to_string());
                    row
                })
                .collect();

            return Ok(QueryResult::Rows(rows));
        }

        // Non-distance ordered query: scan all matching docs
        let docs = self.matching_docs(&query.filter)?;

        let select_all =
            query.fields.is_empty() || (query.fields.len() == 1 && query.fields[0] == "*");

        let mut rows: Vec<HashMap<String, String>> = docs
            .into_iter()
            .map(|(pk, fields)| {
                let mut row = if select_all {
                    fields
                } else {
                    let mut m = HashMap::new();
                    for f in &query.fields {
                        if let Some(v) = fields.get(f) {
                            m.insert(f.clone(), v.clone());
                        }
                    }
                    m
                };
                row.insert("_pk".to_string(), pk);
                row
            })
            .collect();

        // ORDER BY field (lexicographic)
        if let Some(OrderBy::Field(ref field)) = query.order_by {
            let f = field.clone();
            rows.sort_by(|a, b| {
                let va = a.get(&f).map(|s| s.as_str()).unwrap_or("");
                let vb = b.get(&f).map(|s| s.as_str()).unwrap_or("");
                va.cmp(vb)
            });
        }

        // LIMIT
        if query.limit > 0 {
            rows.truncate(query.limit);
        }

        Ok(QueryResult::Rows(rows))
    }

    /// Return all (pk, fields) pairs matching the optional filter.
    fn matching_docs(
        &self,
        filter: &Option<FilterExpr>,
    ) -> Result<Vec<(String, HashMap<String, String>)>, String> {
        let fields_map = self.fields.read().map_err(|e| format!("lock: {}", e))?;

        let docs: Vec<(String, HashMap<String, String>)> = match filter {
            Some(expr) => fields_map
                .iter()
                .filter(|(_, doc_fields)| filter::matches(expr, doc_fields))
                .map(|(pk, f)| (pk.clone(), f.clone()))
                .collect(),
            None => fields_map
                .iter()
                .map(|(pk, f)| (pk.clone(), f.clone()))
                .collect(),
        };

        Ok(docs)
    }
}

// ---------------------------------------------------------------------------
// Filter expression to string (for passing back to Collection::search)
// ---------------------------------------------------------------------------

fn filter_to_string(expr: &FilterExpr) -> String {
    match expr {
        FilterExpr::Eq(f, v) => format!("{} = '{}'", f, v),
        FilterExpr::Ne(f, v) => format!("{} != '{}'", f, v),
        FilterExpr::Lt(f, v) => format!("{} < '{}'", f, v),
        FilterExpr::Le(f, v) => format!("{} <= '{}'", f, v),
        FilterExpr::Gt(f, v) => format!("{} > '{}'", f, v),
        FilterExpr::Ge(f, v) => format!("{} >= '{}'", f, v),
        FilterExpr::In(f, vals) => {
            let vs: Vec<String> = vals.iter().map(|v| format!("'{}'", v)).collect();
            format!("{} IN ({})", f, vs.join(", "))
        }
        FilterExpr::Contains(f, v) => format!("{} CONTAINS '{}'", f, v),
        FilterExpr::Like(f, v) => format!("{} LIKE '{}'", f, v),
        FilterExpr::IsNull(f) => format!("{} IS NULL", f),
        FilterExpr::IsNotNull(f) => format!("{} IS NOT NULL", f),
        FilterExpr::And(l, r) => format!("({} AND {})", filter_to_string(l), filter_to_string(r)),
        FilterExpr::Or(l, r) => format!("({} OR {})", filter_to_string(l), filter_to_string(r)),
        FilterExpr::Not(e) => format!("NOT ({})", filter_to_string(e)),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collection::CollectionConfig;
    use crate::distance::MetricType;
    use crate::schema::{FieldSchema, FieldType};

    fn test_schema() -> FieldSchema {
        FieldSchema::new(vec![
            ("category".into(), FieldType::Filtered),
            ("tenant".into(), FieldType::Filtered),
            ("name".into(), FieldType::String),
        ])
    }

    fn setup_collection() -> Collection {
        let col = Collection::new(
            CollectionConfig::new(3)
                .with_metric(MetricType::L2)
                .with_schema(test_schema()),
        );

        let mut f1 = HashMap::new();
        f1.insert("category".to_string(), "news".to_string());
        f1.insert("tenant".to_string(), "acme".to_string());
        f1.insert("name".to_string(), "alpha".to_string());
        col.upsert("doc-1", &[1.0, 0.0, 0.0], f1);

        let mut f2 = HashMap::new();
        f2.insert("category".to_string(), "blog".to_string());
        f2.insert("tenant".to_string(), "acme".to_string());
        f2.insert("name".to_string(), "beta".to_string());
        col.upsert("doc-2", &[0.0, 1.0, 0.0], f2);

        let mut f3 = HashMap::new();
        f3.insert("category".to_string(), "news".to_string());
        f3.insert("tenant".to_string(), "globex".to_string());
        f3.insert("name".to_string(), "gamma".to_string());
        col.upsert("doc-3", &[0.0, 0.0, 1.0], f3);

        let mut f4 = HashMap::new();
        f4.insert("category".to_string(), "blog".to_string());
        f4.insert("tenant".to_string(), "globex".to_string());
        f4.insert("name".to_string(), "delta".to_string());
        col.upsert("doc-4", &[1.0, 1.0, 0.0], f4);

        col
    }

    // ---- Parse tests ----

    #[test]
    fn test_parse_select() {
        let q = Query::parse("SELECT name, category FROM my_collection").unwrap();
        assert_eq!(q.fields, vec!["name", "category"]);
        assert!(q.filter.is_none());
        assert!(q.order_by.is_none());
        assert_eq!(q.limit, 0);
    }

    #[test]
    fn test_parse_select_star() {
        let q = Query::parse("SELECT * FROM col").unwrap();
        assert_eq!(q.fields, vec!["*"]);
    }

    #[test]
    fn test_parse_select_where() {
        let q =
            Query::parse("SELECT name FROM col WHERE category = 'news'").unwrap();
        assert!(q.filter.is_some());
    }

    #[test]
    fn test_parse_select_order_limit() {
        let q = Query::parse(
            "SELECT name FROM col WHERE category = 'news' ORDER BY _distance LIMIT 5",
        )
        .unwrap();
        assert_eq!(q.order_by, Some(OrderBy::Distance));
        assert_eq!(q.limit, 5);
    }

    #[test]
    fn test_parse_select_order_by_field() {
        let q =
            Query::parse("SELECT name FROM col ORDER BY name LIMIT 10").unwrap();
        assert_eq!(q.order_by, Some(OrderBy::Field("name".to_string())));
        assert_eq!(q.limit, 10);
    }

    #[test]
    fn test_parse_count() {
        let q = Query::parse("SELECT COUNT(*) FROM col WHERE tenant = 'acme'").unwrap();
        assert_eq!(q.kind, QueryKind::Count);
        assert!(q.filter.is_some());
    }

    #[test]
    fn test_parse_count_no_where() {
        let q = Query::parse("SELECT COUNT(*) FROM col").unwrap();
        assert_eq!(q.kind, QueryKind::Count);
        assert!(q.filter.is_none());
    }

    #[test]
    fn test_parse_group_by() {
        let q = Query::parse(
            "SELECT category, COUNT(*) FROM col GROUP BY category WHERE tenant = 'acme'",
        )
        .unwrap();
        assert_eq!(q.kind, QueryKind::GroupCount("category".to_string()));
        assert!(q.filter.is_some());
    }

    #[test]
    fn test_parse_group_by_no_where() {
        let q = Query::parse("SELECT category, COUNT(*) FROM col GROUP BY category")
            .unwrap();
        assert_eq!(q.kind, QueryKind::GroupCount("category".to_string()));
        assert!(q.filter.is_none());
    }

    #[test]
    fn test_parse_errors() {
        assert!(Query::parse("").is_err());
        assert!(Query::parse("INSERT INTO foo").is_err());
        assert!(Query::parse("SELECT").is_err());
        assert!(Query::parse("SELECT name").is_err()); // missing FROM
    }

    // ---- Builder tests ----

    #[test]
    fn test_builder_select() {
        let q = Query::select(&["name", "category"])
            .where_filter("tenant = 'acme'")
            .unwrap()
            .order_by("name")
            .limit(10);

        assert_eq!(q.fields, vec!["name", "category"]);
        assert!(q.filter.is_some());
        assert_eq!(q.order_by, Some(OrderBy::Field("name".to_string())));
        assert_eq!(q.limit, 10);
    }

    #[test]
    fn test_builder_count() {
        let q = Query::count()
            .where_filter("category = 'news'")
            .unwrap();
        assert_eq!(q.kind, QueryKind::Count);
        assert!(q.filter.is_some());
    }

    #[test]
    fn test_builder_group_count() {
        let q = Query::group_count("category")
            .where_filter("tenant = 'acme'")
            .unwrap();
        assert_eq!(q.kind, QueryKind::GroupCount("category".to_string()));
    }

    // ---- Execution tests ----

    #[test]
    fn test_exec_count_all() {
        let col = setup_collection();
        let q = Query::count();
        let result = col.execute_query(&q, None).unwrap();
        assert_eq!(result, QueryResult::Count(4));
    }

    #[test]
    fn test_exec_count_filtered() {
        let col = setup_collection();
        let q = Query::count()
            .where_filter("category = 'news'")
            .unwrap();
        let result = col.execute_query(&q, None).unwrap();
        assert_eq!(result, QueryResult::Count(2));
    }

    #[test]
    fn test_exec_group_count() {
        let col = setup_collection();
        let q = Query::group_count("category");
        let result = col.execute_query(&q, None).unwrap();

        if let QueryResult::GroupCount(groups) = result {
            assert_eq!(groups.len(), 2);
            // Both categories have 2 docs each
            for (_, count) in &groups {
                assert_eq!(*count, 2);
            }
        } else {
            panic!("expected GroupCount result");
        }
    }

    #[test]
    fn test_exec_group_count_filtered() {
        let col = setup_collection();
        let q = Query::group_count("category")
            .where_filter("tenant = 'acme'")
            .unwrap();
        let result = col.execute_query(&q, None).unwrap();

        if let QueryResult::GroupCount(groups) = result {
            assert_eq!(groups.len(), 2);
            for (_, count) in &groups {
                assert_eq!(*count, 1);
            }
        } else {
            panic!("expected GroupCount result");
        }
    }

    #[test]
    fn test_exec_select_all() {
        let col = setup_collection();
        let q = Query::select(&["*"]);
        let result = col.execute_query(&q, None).unwrap();

        if let QueryResult::Rows(rows) = result {
            assert_eq!(rows.len(), 4);
            // Each row should have _pk and the fields
            for row in &rows {
                assert!(row.contains_key("_pk"));
                assert!(row.contains_key("category"));
            }
        } else {
            panic!("expected Rows result");
        }
    }

    #[test]
    fn test_exec_select_specific_fields() {
        let col = setup_collection();
        let q = Query::select(&["name"]);
        let result = col.execute_query(&q, None).unwrap();

        if let QueryResult::Rows(rows) = result {
            for row in &rows {
                assert!(row.contains_key("_pk"));
                assert!(row.contains_key("name"));
                // Should NOT have category since we only asked for name
                assert!(!row.contains_key("category"));
            }
        } else {
            panic!("expected Rows result");
        }
    }

    #[test]
    fn test_exec_select_with_filter() {
        let col = setup_collection();
        let q = Query::select(&["name"])
            .where_filter("category = 'news'")
            .unwrap();
        let result = col.execute_query(&q, None).unwrap();

        if let QueryResult::Rows(rows) = result {
            assert_eq!(rows.len(), 2);
        } else {
            panic!("expected Rows result");
        }
    }

    #[test]
    fn test_exec_select_order_by_field() {
        let col = setup_collection();
        let q = Query::select(&["name"]).order_by("name");
        let result = col.execute_query(&q, None).unwrap();

        if let QueryResult::Rows(rows) = result {
            let names: Vec<&str> = rows
                .iter()
                .map(|r| r.get("name").unwrap().as_str())
                .collect();
            assert_eq!(names, vec!["alpha", "beta", "delta", "gamma"]);
        } else {
            panic!("expected Rows result");
        }
    }

    #[test]
    fn test_exec_select_limit() {
        let col = setup_collection();
        let q = Query::select(&["name"]).order_by("name").limit(2);
        let result = col.execute_query(&q, None).unwrap();

        if let QueryResult::Rows(rows) = result {
            assert_eq!(rows.len(), 2);
            assert_eq!(rows[0].get("name").unwrap(), "alpha");
            assert_eq!(rows[1].get("name").unwrap(), "beta");
        } else {
            panic!("expected Rows result");
        }
    }

    #[test]
    fn test_exec_select_order_by_distance() {
        let col = setup_collection();
        let q = Query::select(&["name"])
            .order_by("_distance")
            .limit(2);
        let result = col
            .execute_query(&q, Some(&[1.0, 0.0, 0.0]))
            .unwrap();

        if let QueryResult::Rows(rows) = result {
            assert_eq!(rows.len(), 2);
            // doc-1 has vector [1,0,0], so L2 distance to [1,0,0] = 0
            assert_eq!(rows[0].get("_pk").unwrap(), "doc-1");
            assert!(rows[0].contains_key("_distance"));
        } else {
            panic!("expected Rows result");
        }
    }

    #[test]
    fn test_exec_distance_requires_vector() {
        let col = setup_collection();
        let q = Query::select(&["name"]).order_by("_distance").limit(2);
        let result = col.execute_query(&q, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_exec_parsed_select() {
        let col = setup_collection();
        let q = Query::parse(
            "SELECT name FROM mycol WHERE category = 'news' ORDER BY name LIMIT 10",
        )
        .unwrap();
        let result = col.execute_query(&q, None).unwrap();

        if let QueryResult::Rows(rows) = result {
            assert_eq!(rows.len(), 2);
            let names: Vec<&str> = rows
                .iter()
                .map(|r| r.get("name").unwrap().as_str())
                .collect();
            assert_eq!(names, vec!["alpha", "gamma"]);
        } else {
            panic!("expected Rows result");
        }
    }

    #[test]
    fn test_exec_parsed_count() {
        let col = setup_collection();
        let q = Query::parse("SELECT COUNT(*) FROM mycol WHERE tenant = 'acme'").unwrap();
        let result = col.execute_query(&q, None).unwrap();
        assert_eq!(result, QueryResult::Count(2));
    }

    #[test]
    fn test_exec_parsed_group_by() {
        let col = setup_collection();
        let q = Query::parse("SELECT tenant, COUNT(*) FROM mycol GROUP BY tenant").unwrap();
        let result = col.execute_query(&q, None).unwrap();

        if let QueryResult::GroupCount(groups) = result {
            assert_eq!(groups.len(), 2);
            for (_, count) in &groups {
                assert_eq!(*count, 2);
            }
        } else {
            panic!("expected GroupCount result");
        }
    }

    #[test]
    fn test_exec_parsed_distance_search() {
        let col = setup_collection();
        let q =
            Query::parse("SELECT name FROM mycol ORDER BY _distance LIMIT 1").unwrap();
        let result = col
            .execute_query(&q, Some(&[0.0, 0.0, 1.0]))
            .unwrap();

        if let QueryResult::Rows(rows) = result {
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].get("_pk").unwrap(), "doc-3");
        } else {
            panic!("expected Rows result");
        }
    }

    #[test]
    fn test_filter_to_string_roundtrip() {
        // Ensure filter_to_string produces valid filter strings
        let expr = parse_filter("category = 'news' AND tenant = 'acme'").unwrap();
        let s = filter_to_string(&expr);
        let re_parsed = parse_filter(&s).unwrap();
        // Both should match the same document
        let mut fields = HashMap::new();
        fields.insert("category".to_string(), "news".to_string());
        fields.insert("tenant".to_string(), "acme".to_string());
        assert!(filter::matches(&expr, &fields));
        assert!(filter::matches(&re_parsed, &fields));
    }
}
