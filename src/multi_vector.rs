//! Multi-vector / mixed reducer queries.
//!
//! Combines results from multiple index types or multiple query vectors
//! using various fusion strategies (RRF, weighted sum, intersect, union).

use std::collections::HashMap;

use crate::hnsw::SearchResult;

/// Strategy for fusing multiple result lists into a single ranked list.
#[derive(Debug, Clone)]
pub enum FusionMethod {
    /// Reciprocal Rank Fusion: score = sum(1 / (rank + k)) across lists.
    /// The constant k defaults to 60 (standard RRF parameter).
    RRF,
    /// Weighted sum of normalized scores. Each result list has a corresponding weight.
    WeightedSum(Vec<f32>),
    /// Intersection: only IDs present in ALL result lists survive. Best score wins.
    Intersect,
    /// Union: all unique IDs, best score wins across lists.
    Union,
}

/// A multi-vector query that holds multiple (vector, weight) pairs and a fusion method.
#[derive(Debug, Clone)]
pub struct MultiVectorQuery {
    /// Each entry is (query_vector, weight).
    pub vectors: Vec<(Vec<f32>, f32)>,
    /// How to combine the result lists.
    pub method: FusionMethod,
}

impl MultiVectorQuery {
    /// Create a new multi-vector query.
    pub fn new(vectors: Vec<(Vec<f32>, f32)>, method: FusionMethod) -> Self {
        Self { vectors, method }
    }
}

/// Fuse multiple search result lists into a single ranked list of `k` results.
///
/// # Arguments
/// * `results` - slice of result lists, one per query/index
/// * `method` - the fusion strategy to apply
/// * `k` - maximum number of results to return
pub fn fuse_results(results: &[Vec<SearchResult>], method: &FusionMethod, k: usize) -> Vec<SearchResult> {
    if results.is_empty() || k == 0 {
        return Vec::new();
    }

    let mut scored: Vec<(u64, f32)> = match method {
        FusionMethod::RRF => fuse_rrf(results),
        FusionMethod::WeightedSum(weights) => fuse_weighted_sum(results, weights),
        FusionMethod::Intersect => fuse_intersect(results),
        FusionMethod::Union => fuse_union(results),
    };

    // Sort descending by fused score (higher = better).
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);

    scored
        .into_iter()
        .map(|(id, score)| SearchResult::new(id, score))
        .collect()
}

/// Reciprocal Rank Fusion: score(id) = sum over lists of 1 / (rank + 60).
fn fuse_rrf(results: &[Vec<SearchResult>]) -> Vec<(u64, f32)> {
    const RRF_K: f32 = 60.0;
    let mut scores: HashMap<u64, f32> = HashMap::new();

    for list in results {
        for (rank, result) in list.iter().enumerate() {
            let rrf_score = 1.0 / (rank as f32 + RRF_K);
            *scores.entry(result.id).or_insert(0.0) += rrf_score;
        }
    }

    scores.into_iter().collect()
}

/// Weighted sum fusion with per-list min-max score normalization.
fn fuse_weighted_sum(results: &[Vec<SearchResult>], weights: &[f32]) -> Vec<(u64, f32)> {
    let mut scores: HashMap<u64, f32> = HashMap::new();

    for (list_idx, list) in results.iter().enumerate() {
        let w = weights.get(list_idx).copied().unwrap_or(1.0);

        if list.is_empty() {
            continue;
        }

        // Min-max normalization of scores within this list.
        let min_score = list.iter().map(|r| r.score).fold(f32::INFINITY, f32::min);
        let max_score = list.iter().map(|r| r.score).fold(f32::NEG_INFINITY, f32::max);
        let range = max_score - min_score;

        for result in list {
            let normalized = if range.abs() < 1e-12 {
                1.0 // all scores equal — treat as max
            } else {
                (result.score - min_score) / range
            };
            *scores.entry(result.id).or_insert(0.0) += w * normalized;
        }
    }

    scores.into_iter().collect()
}

/// Intersection: only IDs present in ALL result lists. Best (highest) score wins.
fn fuse_intersect(results: &[Vec<SearchResult>]) -> Vec<(u64, f32)> {
    if results.is_empty() {
        return Vec::new();
    }

    // Count how many lists each ID appears in, and track best score.
    let mut counts: HashMap<u64, usize> = HashMap::new();
    let mut best_scores: HashMap<u64, f32> = HashMap::new();
    let n_lists = results.len();

    for list in results {
        // Use a set to count each ID at most once per list.
        let mut seen = std::collections::HashSet::new();
        for result in list {
            if seen.insert(result.id) {
                *counts.entry(result.id).or_insert(0) += 1;
            }
            let entry = best_scores.entry(result.id).or_insert(f32::NEG_INFINITY);
            if result.score > *entry {
                *entry = result.score;
            }
        }
    }

    counts
        .into_iter()
        .filter(|(_, count)| *count == n_lists)
        .map(|(id, _)| (id, *best_scores.get(&id).unwrap()))
        .collect()
}

/// Union: all unique IDs, best (highest) score wins.
fn fuse_union(results: &[Vec<SearchResult>]) -> Vec<(u64, f32)> {
    let mut best_scores: HashMap<u64, f32> = HashMap::new();

    for list in results {
        for result in list {
            let entry = best_scores.entry(result.id).or_insert(f32::NEG_INFINITY);
            if result.score > *entry {
                *entry = result.score;
            }
        }
    }

    best_scores.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_results(ids_scores: &[(u64, f32)]) -> Vec<SearchResult> {
        ids_scores
            .iter()
            .map(|&(id, score)| SearchResult::new(id, score))
            .collect()
    }

    #[test]
    fn test_rrf_basic() {
        let list_a = make_results(&[(1, 0.9), (2, 0.8), (3, 0.7)]);
        let list_b = make_results(&[(2, 0.95), (1, 0.85), (4, 0.6)]);

        let fused = fuse_results(&[list_a, list_b], &FusionMethod::RRF, 10);
        assert!(!fused.is_empty());

        // IDs 1 and 2 appear in both lists, so they should have higher RRF scores.
        let id_scores: HashMap<u64, f32> = fused.iter().map(|r| (r.id, r.score)).collect();
        assert!(id_scores.contains_key(&1));
        assert!(id_scores.contains_key(&2));

        // ID 2 is rank 0 in list_b and rank 1 in list_a => 1/60 + 1/61
        // ID 1 is rank 0 in list_a and rank 1 in list_b => 1/60 + 1/61
        // Both should have the same RRF score.
        let score_1 = id_scores[&1];
        let score_2 = id_scores[&2];
        assert!((score_1 - score_2).abs() < 1e-6);
    }

    #[test]
    fn test_rrf_ranking_order() {
        // ID 1 at rank 0 in both lists, ID 2 at rank 1 in both.
        let list_a = make_results(&[(1, 10.0), (2, 5.0)]);
        let list_b = make_results(&[(1, 10.0), (2, 5.0)]);

        let fused = fuse_results(&[list_a, list_b], &FusionMethod::RRF, 2);
        assert_eq!(fused.len(), 2);
        assert_eq!(fused[0].id, 1); // rank 0 in both => highest RRF
        assert_eq!(fused[1].id, 2);
    }

    #[test]
    fn test_weighted_sum_basic() {
        let list_a = make_results(&[(1, 1.0), (2, 0.5)]);
        let list_b = make_results(&[(2, 1.0), (3, 0.5)]);

        let fused = fuse_results(
            &[list_a, list_b],
            &FusionMethod::WeightedSum(vec![1.0, 1.0]),
            10,
        );
        assert!(!fused.is_empty());

        let id_scores: HashMap<u64, f32> = fused.iter().map(|r| (r.id, r.score)).collect();
        // ID 2 appears in both lists: normalized to 0.0 in list_a (min), 1.0 in list_b (max).
        // So its weighted score = 0.0 + 1.0 = 1.0
        // ID 1: normalized to 1.0 in list_a, not in list_b => 1.0
        // ID 3: not in list_a, normalized to 0.0 in list_b => 0.0
        assert!(id_scores[&3] < id_scores[&1]);
    }

    #[test]
    fn test_weighted_sum_with_weights() {
        let list_a = make_results(&[(1, 1.0), (2, 0.0)]);
        let list_b = make_results(&[(2, 1.0), (1, 0.0)]);

        // Weight list_a heavily.
        let fused = fuse_results(
            &[list_a, list_b],
            &FusionMethod::WeightedSum(vec![10.0, 1.0]),
            10,
        );
        assert_eq!(fused[0].id, 1); // ID 1 has score 10*1 + 1*0 = 10
        assert_eq!(fused[1].id, 2); // ID 2 has score 10*0 + 1*1 = 1
    }

    #[test]
    fn test_intersect_basic() {
        let list_a = make_results(&[(1, 0.9), (2, 0.8), (3, 0.7)]);
        let list_b = make_results(&[(2, 0.95), (3, 0.6), (4, 0.5)]);

        let fused = fuse_results(&[list_a, list_b], &FusionMethod::Intersect, 10);
        let ids: Vec<u64> = fused.iter().map(|r| r.id).collect();
        // Only IDs 2 and 3 appear in both lists.
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&2));
        assert!(ids.contains(&3));
        assert!(!ids.contains(&1));
        assert!(!ids.contains(&4));
    }

    #[test]
    fn test_intersect_no_overlap() {
        let list_a = make_results(&[(1, 0.9)]);
        let list_b = make_results(&[(2, 0.95)]);

        let fused = fuse_results(&[list_a, list_b], &FusionMethod::Intersect, 10);
        assert!(fused.is_empty());
    }

    #[test]
    fn test_union_basic() {
        let list_a = make_results(&[(1, 0.9), (2, 0.8)]);
        let list_b = make_results(&[(2, 0.95), (3, 0.6)]);

        let fused = fuse_results(&[list_a, list_b], &FusionMethod::Union, 10);
        let ids: Vec<u64> = fused.iter().map(|r| r.id).collect();
        assert_eq!(ids.len(), 3);
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
        assert!(ids.contains(&3));

        // ID 2: best score is 0.95 (from list_b).
        let score_2 = fused.iter().find(|r| r.id == 2).unwrap().score;
        assert!((score_2 - 0.95).abs() < 1e-6);
    }

    #[test]
    fn test_union_best_score_wins() {
        let list_a = make_results(&[(1, 0.5)]);
        let list_b = make_results(&[(1, 0.9)]);

        let fused = fuse_results(&[list_a, list_b], &FusionMethod::Union, 10);
        assert_eq!(fused.len(), 1);
        assert!((fused[0].score - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_fuse_empty_input() {
        let fused = fuse_results(&[], &FusionMethod::RRF, 10);
        assert!(fused.is_empty());
    }

    #[test]
    fn test_fuse_k_zero() {
        let list = make_results(&[(1, 0.9)]);
        let fused = fuse_results(&[list], &FusionMethod::RRF, 0);
        assert!(fused.is_empty());
    }

    #[test]
    fn test_fuse_truncates_to_k() {
        let list_a = make_results(&[(1, 0.9), (2, 0.8), (3, 0.7), (4, 0.6)]);
        let fused = fuse_results(&[list_a], &FusionMethod::RRF, 2);
        assert_eq!(fused.len(), 2);
    }

    #[test]
    fn test_multi_vector_query_construction() {
        let query = MultiVectorQuery::new(
            vec![
                (vec![1.0, 0.0, 0.0], 0.7),
                (vec![0.0, 1.0, 0.0], 0.3),
            ],
            FusionMethod::RRF,
        );
        assert_eq!(query.vectors.len(), 2);
        assert!((query.vectors[0].1 - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_rrf_single_list() {
        let list = make_results(&[(10, 0.9), (20, 0.5), (30, 0.1)]);
        let fused = fuse_results(&[list], &FusionMethod::RRF, 3);
        assert_eq!(fused.len(), 3);
        // Rank 0 should have highest RRF score.
        assert_eq!(fused[0].id, 10);
        assert_eq!(fused[1].id, 20);
        assert_eq!(fused[2].id, 30);
    }

    #[test]
    fn test_weighted_sum_single_element_lists() {
        let list_a = make_results(&[(1, 5.0)]);
        let list_b = make_results(&[(2, 3.0)]);

        let fused = fuse_results(
            &[list_a, list_b],
            &FusionMethod::WeightedSum(vec![1.0, 1.0]),
            10,
        );
        // Single-element lists: all scores normalize to 1.0.
        assert_eq!(fused.len(), 2);
        assert!((fused[0].score - 1.0).abs() < 1e-6);
        assert!((fused[1].score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_intersect_three_lists() {
        let list_a = make_results(&[(1, 0.9), (2, 0.8), (3, 0.7)]);
        let list_b = make_results(&[(2, 0.85), (3, 0.6), (4, 0.5)]);
        let list_c = make_results(&[(3, 0.95), (5, 0.4)]);

        let fused = fuse_results(&[list_a, list_b, list_c], &FusionMethod::Intersect, 10);
        // Only ID 3 is in all three lists.
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].id, 3);
        assert!((fused[0].score - 0.95).abs() < 1e-6); // best score
    }
}
