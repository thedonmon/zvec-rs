use std::collections::HashMap;

use crate::collection::SearchHit;
use super::ExtensionError;

/// Named result lists from one or more vector queries.
///
/// Keys are query/field identifiers (e.g., `"dense"`, `"sparse"`).
/// Values are search hits from each query, in relevance order.
pub type RerankInput = HashMap<String, Vec<SearchHit>>;

/// Re-ranks search results using a secondary scoring strategy.
///
/// Rerankers refine the output of one or more vector queries. They can be
/// purely algorithmic (RRF, weighted fusion) or model-based (cross-encoder APIs).
///
/// Implementations must be `Send + Sync` for use across threads.
pub trait Reranker: Send + Sync {
    /// Maximum number of results to return after re-ranking.
    fn top_n(&self) -> usize;

    /// Re-rank documents from one or more query result lists.
    ///
    /// Returns a single merged and re-scored list of [`SearchHit`], ordered by
    /// the reranker's scoring strategy, truncated to `top_n()`.
    fn rerank(&self, query_results: &RerankInput) -> Result<Vec<SearchHit>, ExtensionError>;
}

/// Async reranker for model-based reranking that requires API calls.
#[cfg(feature = "async")]
#[async_trait::async_trait]
pub trait AsyncReranker: Send + Sync {
    /// Maximum number of results to return after re-ranking.
    fn top_n(&self) -> usize;

    /// Re-rank documents from one or more query result lists.
    async fn rerank(&self, query_results: &RerankInput) -> Result<Vec<SearchHit>, ExtensionError>;
}
