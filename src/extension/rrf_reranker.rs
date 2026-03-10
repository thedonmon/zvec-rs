use std::collections::HashMap;

use crate::collection::SearchHit;
use super::error::ExtensionError;
use super::reranker::{RerankInput, Reranker};

/// Reciprocal Rank Fusion reranker.
///
/// Combines results from multiple vector queries using the RRF formula:
/// `score(doc) = sum over lists of 1 / (k + rank + 1)`
///
/// This is a purely algorithmic reranker — no model or API calls needed.
/// Works well for combining dense + sparse hybrid search results.
pub struct RrfReranker {
    top_n: usize,
    /// Smoothing constant (default 60). Higher values reduce the advantage
    /// of top-ranked documents.
    rank_constant: usize,
}

impl RrfReranker {
    /// Create a new RRF reranker.
    ///
    /// # Arguments
    /// * `top_n` — maximum results to return
    /// * `rank_constant` — RRF smoothing constant k (default 60)
    pub fn new(top_n: usize, rank_constant: usize) -> Self {
        Self {
            top_n,
            rank_constant,
        }
    }

    /// Create with default rank constant of 60.
    pub fn with_top_n(top_n: usize) -> Self {
        Self::new(top_n, 60)
    }
}

impl Reranker for RrfReranker {
    fn top_n(&self) -> usize {
        self.top_n
    }

    fn rerank(&self, query_results: &RerankInput) -> Result<Vec<SearchHit>, ExtensionError> {
        if query_results.is_empty() {
            return Err(ExtensionError::EmptyInput);
        }

        let k = self.rank_constant as f32;
        let mut scores: HashMap<&str, f32> = HashMap::new();
        // Track the best (first-seen) SearchHit for each pk so we can return full fields.
        let mut best_hit: HashMap<&str, &SearchHit> = HashMap::new();

        for hits in query_results.values() {
            for (rank, hit) in hits.iter().enumerate() {
                let rrf_score = 1.0 / (k + rank as f32 + 1.0);
                *scores.entry(&hit.pk).or_insert(0.0) += rrf_score;
                best_hit.entry(&hit.pk).or_insert(hit);
            }
        }

        let mut ranked: Vec<SearchHit> = scores
            .into_iter()
            .map(|(pk, score)| {
                let original = best_hit[pk];
                SearchHit {
                    pk: pk.to_string(),
                    score,
                    fields: original.fields.clone(),
                }
            })
            .collect();

        ranked.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(self.top_n);
        Ok(ranked)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(pk: &str, score: f32) -> SearchHit {
        SearchHit {
            pk: pk.to_string(),
            score,
            fields: HashMap::new(),
        }
    }

    #[test]
    fn test_rrf_single_list() {
        let reranker = RrfReranker::with_top_n(10);
        let mut input = RerankInput::new();
        input.insert(
            "dense".into(),
            vec![hit("a", 0.9), hit("b", 0.8), hit("c", 0.7)],
        );

        let results = reranker.rerank(&input).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].pk, "a"); // rank 0 => highest RRF
        assert_eq!(results[1].pk, "b");
        assert_eq!(results[2].pk, "c");
    }

    #[test]
    fn test_rrf_multi_list_fusion() {
        let reranker = RrfReranker::with_top_n(10);
        let mut input = RerankInput::new();
        input.insert("dense".into(), vec![hit("a", 0.9), hit("b", 0.5)]);
        input.insert("sparse".into(), vec![hit("b", 0.95), hit("c", 0.6)]);

        let results = reranker.rerank(&input).unwrap();
        // "b" appears in both lists: rank 1 + rank 0 => higher combined RRF
        // "a" appears once at rank 0; "c" appears once at rank 1
        assert_eq!(results[0].pk, "b");
    }

    #[test]
    fn test_rrf_truncates() {
        let reranker = RrfReranker::with_top_n(1);
        let mut input = RerankInput::new();
        input.insert(
            "dense".into(),
            vec![hit("a", 0.9), hit("b", 0.8)],
        );
        let results = reranker.rerank(&input).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_rrf_empty_input() {
        let reranker = RrfReranker::with_top_n(10);
        let input = RerankInput::new();
        assert!(reranker.rerank(&input).is_err());
    }

    #[test]
    fn test_rrf_custom_rank_constant() {
        let reranker = RrfReranker::new(10, 1);
        let mut input = RerankInput::new();
        input.insert("dense".into(), vec![hit("a", 0.9), hit("b", 0.8)]);

        let results = reranker.rerank(&input).unwrap();
        // With k=1: rank 0 => 1/(1+0+1) = 0.5, rank 1 => 1/(1+1+1) = 0.333
        assert!((results[0].score - 0.5).abs() < 1e-6);
        assert!((results[1].score - 1.0 / 3.0).abs() < 1e-6);
    }
}
