use crate::sparse::SparseVector;
use super::ExtensionError;

/// Dense embedding function: maps text to a fixed-length f32 vector.
///
/// Implementations must be `Send + Sync` for use across threads
/// (e.g., shared via `Arc` with a `Collection`).
///
/// # Multimodality
///
/// This trait handles text input. For images, audio, or other modalities,
/// implement separate traits (e.g., `ImageEmbeddingFunction`) — this avoids
/// trait object issues and keeps the API type-safe.
pub trait DenseEmbeddingFunction: Send + Sync {
    /// The dimensionality of vectors produced by this function.
    fn dimension(&self) -> usize;

    /// Embed a single text input into a dense vector.
    fn embed(&self, input: &str) -> Result<Vec<f32>, ExtensionError>;

    /// Embed a batch of text inputs. Default calls `embed()` sequentially;
    /// implementations should override when the backend supports batching.
    fn embed_batch(&self, inputs: &[&str]) -> Result<Vec<Vec<f32>>, ExtensionError> {
        inputs.iter().map(|s| self.embed(s)).collect()
    }
}

/// Sparse embedding function: maps text to a [`SparseVector`].
///
/// Sparse embeddings (BM25, SPLADE, etc.) represent text as high-dimensional
/// vectors where most dimensions are zero. Only non-zero entries are stored.
pub trait SparseEmbeddingFunction: Send + Sync {
    /// Embed a single text input into a sparse vector.
    fn embed_sparse(&self, input: &str) -> Result<SparseVector, ExtensionError>;

    /// Batch sparse embedding. Default calls `embed_sparse()` sequentially.
    fn embed_sparse_batch(
        &self,
        inputs: &[&str],
    ) -> Result<Vec<SparseVector>, ExtensionError> {
        inputs.iter().map(|s| self.embed_sparse(s)).collect()
    }
}

// ---------------------------------------------------------------------------
// Async variants
// ---------------------------------------------------------------------------

/// Async dense embedding function for use in async runtimes (tokio, etc.).
///
/// This is the preferred trait for API-backed embeddings that make HTTP calls.
/// Use the sync [`DenseEmbeddingFunction`] for local/in-process embeddings.
#[cfg(feature = "async")]
#[async_trait::async_trait]
pub trait AsyncDenseEmbeddingFunction: Send + Sync {
    /// The dimensionality of vectors produced by this function.
    fn dimension(&self) -> usize;

    /// Embed a single text input into a dense vector.
    async fn embed(&self, input: &str) -> Result<Vec<f32>, ExtensionError>;

    /// Embed a batch of text inputs. Default calls `embed()` sequentially;
    /// implementations should override when the backend supports batching.
    async fn embed_batch(&self, inputs: &[&str]) -> Result<Vec<Vec<f32>>, ExtensionError> {
        let mut results = Vec::with_capacity(inputs.len());
        for input in inputs {
            results.push(self.embed(input).await?);
        }
        Ok(results)
    }
}

/// Async sparse embedding function.
#[cfg(feature = "async")]
#[async_trait::async_trait]
pub trait AsyncSparseEmbeddingFunction: Send + Sync {
    /// Embed a single text input into a sparse vector.
    async fn embed_sparse(&self, input: &str) -> Result<SparseVector, ExtensionError>;

    /// Batch sparse embedding.
    async fn embed_sparse_batch(
        &self,
        inputs: &[&str],
    ) -> Result<Vec<SparseVector>, ExtensionError> {
        let mut results = Vec::with_capacity(inputs.len());
        for input in inputs {
            results.push(self.embed_sparse(input).await?);
        }
        Ok(results)
    }
}
