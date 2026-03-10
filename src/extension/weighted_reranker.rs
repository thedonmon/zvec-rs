use std::collections::HashMap;

use crate::collection::SearchHit;
use crate::distance::MetricType;
use super::error::ExtensionError;
use super::reranker::{RerankInput, Reranker};

/// Weighted score fusion reranker with metric-aware normalization.
///
/// Combines results from multiple vector queries by:
/// 1. Normalizing each list's scores based on the distance metric
/// 2. Applying per-field weights
/// 3. Summing the weighted normalized scores
///
/// This is a purely algorithmic reranker — no model or API calls needed.
pub struct WeightedReranker {
    top_n: usize,
    /// Per-field weights. Missing fields default to 1.0.
    weights: HashMap<String, f32>,
    /// Distance metric, used for score normalization.
    metric: MetricType,
}

impl WeightedReranker {
    /// Create a new weighted reranker.
    ///
    /// # Arguments
    /// * `top_n` — maximum results to return
    /// * `metric` — distance metric for score normalization
    /// * `weights` — per-field weights (field_name → weight)
    pub fn new(top_n: usize, metric: MetricType, weights: HashMap<String, f32>) -> Self {
        Self {
            top_n,
            weights,
            metric,
        }
    }

    /// Create with uniform weights (1.0 for all fields).
    pub fn uniform(top_n: usize, metric: MetricType) -> Self {
        Self::new(top_n, metric, HashMap::new())
    }

    /// Normalize a raw distance/similarity score to [0, 1] where higher is better.
    fn normalize_score(&self, score: f32) -> f32 {
        match self.metric {
            MetricType::L2 => {
                // L2: lower is better. Map via arctan: 1 - 2*atan(score)/π
                1.0 - 2.0 * score.atan() / std::f32::consts::PI
            }
            MetricType::IP | MetricType::MIPS => {
                // IP: higher is better. Map via arctan: 0.5 + atan(score)/π
                0.5 + score.atan() / std::f32::consts::PI
            }
            MetricType::Cosine => {
                // Cosine distance: lower is better, range [0, 2].
                // Convert to similarity: 1 - score/2
                1.0 - score / 2.0
            }
        }
    }
}

impl Reranker for WeightedReranker {
    fn top_n(&self) -> usize {
        self.top_n
    }

    fn rerank(&self, query_results: &RerankInput) -> Result<Vec<SearchHit>, ExtensionError> {
        if query_results.is_empty() {
            return Err(ExtensionError::EmptyInput);
        }

        let mut scores: HashMap<&str, f32> = HashMap::new();
        let mut best_hit: HashMap<&str, &SearchHit> = HashMap::new();

        for (field_name, hits) in query_results {
            let weight = self.weights.get(field_name).copied().unwrap_or(1.0);

            for hit in hits {
                let normalized = self.normalize_score(hit.score);
                *scores.entry(&hit.pk).or_insert(0.0) += weight * normalized;
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
    fn test_weighted_uniform_cosine() {
        let reranker = WeightedReranker::uniform(10, MetricType::Cosine);
        let mut input = RerankInput::new();
        input.insert("dense".into(), vec![hit("a", 0.1), hit("b", 0.5)]);
        input.insert("sparse".into(), vec![hit("b", 0.2), hit("c", 0.8)]);

        let results = reranker.rerank(&input).unwrap();
        // "b" appears in both lists with combined weighted scores
        assert_eq!(results[0].pk, "b");
    }

    #[test]
    fn test_weighted_custom_weights() {
        let mut weights = HashMap::new();
        weights.insert("dense".to_string(), 10.0);
        weights.insert("sparse".to_string(), 0.1);

        let reranker = WeightedReranker::new(10, MetricType::Cosine, weights);
        let mut input = RerankInput::new();
        // "a" is top in dense (heavily weighted), "b" is top in sparse (low weight)
        input.insert("dense".into(), vec![hit("a", 0.0), hit("b", 1.0)]);
        input.insert("sparse".into(), vec![hit("b", 0.0), hit("a", 1.0)]);

        let results = reranker.rerank(&input).unwrap();
        // "a" should win because dense weight=10.0 and its dense score is perfect (0.0 cosine dist)
        assert_eq!(results[0].pk, "a");
    }

    #[test]
    fn test_weighted_l2_normalization() {
        let reranker = WeightedReranker::uniform(10, MetricType::L2);
        let mut input = RerankInput::new();
        // L2: lower is better. score=0 should normalize to ~1.0
        input.insert("dense".into(), vec![hit("a", 0.0), hit("b", 100.0)]);

        let results = reranker.rerank(&input).unwrap();
        assert_eq!(results[0].pk, "a"); // closer (L2=0) should rank higher
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn test_weighted_ip_normalization() {
        let reranker = WeightedReranker::uniform(10, MetricType::IP);
        let mut input = RerankInput::new();
        // IP: higher is better. High IP score should normalize higher.
        input.insert("dense".into(), vec![hit("a", 10.0), hit("b", -5.0)]);

        let results = reranker.rerank(&input).unwrap();
        assert_eq!(results[0].pk, "a");
    }

    #[test]
    fn test_weighted_truncates() {
        let reranker = WeightedReranker::uniform(1, MetricType::Cosine);
        let mut input = RerankInput::new();
        input.insert("dense".into(), vec![hit("a", 0.1), hit("b", 0.5)]);

        let results = reranker.rerank(&input).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_weighted_empty_input() {
        let reranker = WeightedReranker::uniform(10, MetricType::Cosine);
        let input = RerankInput::new();
        assert!(reranker.rerank(&input).is_err());
    }
}
