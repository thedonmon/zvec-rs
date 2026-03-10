//! Extension system for AI-powered embeddings and reranking.
//!
//! This module provides pluggable traits for:
//! - **Dense embeddings** — convert text to fixed-length f32 vectors
//! - **Sparse embeddings** — convert text to sparse (index, value) vectors
//! - **Reranking** — re-score search results using a secondary signal
//!
//! # Sync vs Async
//!
//! The core traits ([`DenseEmbeddingFunction`], [`SparseEmbeddingFunction`],
//! [`Reranker`]) are synchronous — suitable for algorithmic rerankers and
//! local embedding models.
//!
//! For API-backed implementations that make HTTP calls, enable the `"async"`
//! feature to get async traits ([`AsyncDenseEmbeddingFunction`],
//! [`AsyncSparseEmbeddingFunction`], [`AsyncReranker`]) and async client
//! implementations ([`AsyncOpenAiEmbedding`], [`AsyncHttpEmbedding`]).
//!
//! # Feature flags
//!
//! | Feature | What it enables |
//! |---------|----------------|
//! | `openai` | `OpenAiEmbedding` (sync, blocking HTTP) |
//! | `http-embedding` | `HttpEmbedding` (sync, blocking HTTP) |
//! | `async` | Async traits + `AsyncOpenAiEmbedding`, `AsyncHttpEmbedding` |

mod embedding;
mod error;
mod reranker;
mod rrf_reranker;
mod weighted_reranker;

#[cfg(feature = "openai")]
mod openai;
#[cfg(feature = "http-embedding")]
mod http_embedding;

// Core sync traits (always available)
pub use embedding::{DenseEmbeddingFunction, SparseEmbeddingFunction};
pub use error::ExtensionError;
pub use reranker::{RerankInput, Reranker};
pub use rrf_reranker::RrfReranker;
pub use weighted_reranker::WeightedReranker;

// Async traits
#[cfg(feature = "async")]
pub use embedding::{AsyncDenseEmbeddingFunction, AsyncSparseEmbeddingFunction};
#[cfg(feature = "async")]
pub use reranker::AsyncReranker;

// Sync API clients (blocking HTTP — don't use inside tokio)
#[cfg(feature = "openai")]
pub use openai::OpenAiEmbedding;
#[cfg(feature = "http-embedding")]
pub use http_embedding::HttpEmbedding;

// Async API clients (preferred for async runtimes)
#[cfg(all(feature = "async", feature = "openai"))]
pub use openai::AsyncOpenAiEmbedding;
#[cfg(all(feature = "async", feature = "http-embedding"))]
pub use http_embedding::AsyncHttpEmbedding;
