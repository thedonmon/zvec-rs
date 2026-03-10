use std::sync::OnceLock;
use std::time::Duration;

use super::error::ExtensionError;

/// Parse an OpenAI-format embedding response, auto-detecting dimension.
fn parse_embedding_response(
    json: &serde_json::Value,
    dimension: &OnceLock<usize>,
) -> Result<Vec<Vec<f32>>, ExtensionError> {
    let data = json["data"]
        .as_array()
        .ok_or_else(|| ExtensionError::ParseError("missing 'data' array".into()))?;

    let mut vectors = Vec::with_capacity(data.len());
    for item in data {
        let embedding = item["embedding"]
            .as_array()
            .ok_or_else(|| ExtensionError::ParseError("missing 'embedding' array".into()))?;

        let vec: Vec<f32> = embedding
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();

        let dim = *dimension.get_or_init(|| vec.len());
        if vec.len() != dim {
            return Err(ExtensionError::DimensionMismatch {
                expected: dim,
                actual: vec.len(),
            });
        }

        vectors.push(vec);
    }

    Ok(vectors)
}

fn validate_input(input: &str) -> Result<(), ExtensionError> {
    if input.is_empty() {
        return Err(ExtensionError::InvalidInput("input text is empty".into()));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Sync implementation (blocking HTTP)
// ---------------------------------------------------------------------------

/// Synchronous generic HTTP embedding client for any OpenAI-compatible API.
///
/// Works with Ollama, vLLM, HuggingFace TEI, LiteLLM, etc.
/// Dimension is auto-detected on first call.
///
/// **Warning**: Do not use inside an async runtime. Use [`AsyncHttpEmbedding`].
///
/// # Example
/// ```no_run
/// use zvec_rs::extension::HttpEmbedding;
/// use zvec_rs::DenseEmbeddingFunction;
///
/// let embedder = HttpEmbedding::new(
///     "http://localhost:11434/v1".to_string(),
///     "nomic-embed-text".to_string(),
/// );
/// let vector = embedder.embed("hello world").unwrap();
/// ```
pub struct HttpEmbedding {
    base_url: String,
    model: String,
    api_key: Option<String>,
    client: reqwest::blocking::Client,
    dimension: OnceLock<usize>,
}

impl HttpEmbedding {
    /// Create a new HTTP embedding client.
    pub fn new(base_url: String, model: String) -> Self {
        Self {
            base_url,
            model,
            api_key: None,
            client: reqwest::blocking::ClientBuilder::new()
                .timeout(Duration::from_secs(30))
                .build()
                .unwrap_or_else(|_| reqwest::blocking::Client::new()),
            dimension: OnceLock::new(),
        }
    }

    /// Create with an API key for authenticated endpoints.
    pub fn with_api_key(base_url: String, model: String, api_key: String) -> Self {
        Self {
            api_key: Some(api_key),
            ..Self::new(base_url, model)
        }
    }

    fn call_api(&self, inputs: &[&str]) -> Result<Vec<Vec<f32>>, ExtensionError> {
        let url = format!("{}/embeddings", self.base_url);
        let body = serde_json::json!({
            "model": self.model,
            "input": inputs,
        });

        let mut req = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&body);

        if let Some(ref key) = self.api_key {
            req = req.header("Authorization", format!("Bearer {key}"));
        }

        let response = req.send().map_err(|e| ExtensionError::Network(e.to_string()))?;

        let status = response.status().as_u16();
        if status != 200 {
            let text = response.text().unwrap_or_default();
            return Err(ExtensionError::Api {
                status,
                message: text,
            });
        }

        let json: serde_json::Value = response
            .json()
            .map_err(|e| ExtensionError::ParseError(e.to_string()))?;

        parse_embedding_response(&json, &self.dimension)
    }
}

impl super::embedding::DenseEmbeddingFunction for HttpEmbedding {
    fn dimension(&self) -> usize {
        self.dimension.get().copied().unwrap_or(0)
    }

    fn embed(&self, input: &str) -> Result<Vec<f32>, ExtensionError> {
        validate_input(input)?;
        let mut results = self.call_api(&[input])?;
        results
            .pop()
            .ok_or_else(|| ExtensionError::ParseError("empty response".into()))
    }

    fn embed_batch(&self, inputs: &[&str]) -> Result<Vec<Vec<f32>>, ExtensionError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        self.call_api(inputs)
    }
}

// ---------------------------------------------------------------------------
// Async implementation
// ---------------------------------------------------------------------------

/// Async generic HTTP embedding client for any OpenAI-compatible API.
///
/// This is the **preferred** client for async applications.
///
/// # Example
/// ```no_run
/// # async fn example() {
/// use zvec_rs::extension::AsyncHttpEmbedding;
///
/// let embedder = AsyncHttpEmbedding::new(
///     "http://localhost:11434/v1".to_string(),
///     "nomic-embed-text".to_string(),
/// );
/// let vector = embedder.embed("hello world").await.unwrap();
/// # }
/// ```
#[cfg(feature = "async")]
pub struct AsyncHttpEmbedding {
    base_url: String,
    model: String,
    api_key: Option<String>,
    client: reqwest::Client,
    dimension: OnceLock<usize>,
}

#[cfg(feature = "async")]
impl AsyncHttpEmbedding {
    /// Create a new async HTTP embedding client.
    pub fn new(base_url: String, model: String) -> Self {
        Self {
            base_url,
            model,
            api_key: None,
            client: reqwest::ClientBuilder::new()
                .timeout(Duration::from_secs(30))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
            dimension: OnceLock::new(),
        }
    }

    /// Create with an API key.
    pub fn with_api_key(base_url: String, model: String, api_key: String) -> Self {
        Self {
            api_key: Some(api_key),
            ..Self::new(base_url, model)
        }
    }

    async fn call_api(&self, inputs: &[&str]) -> Result<Vec<Vec<f32>>, ExtensionError> {
        let url = format!("{}/embeddings", self.base_url);
        let body = serde_json::json!({
            "model": self.model,
            "input": inputs,
        });

        let mut req = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&body);

        if let Some(ref key) = self.api_key {
            req = req.header("Authorization", format!("Bearer {key}"));
        }

        let response = req
            .send()
            .await
            .map_err(|e| ExtensionError::Network(e.to_string()))?;

        let status = response.status().as_u16();
        if status != 200 {
            let text = response.text().await.unwrap_or_default();
            return Err(ExtensionError::Api {
                status,
                message: text,
            });
        }

        let json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| ExtensionError::ParseError(e.to_string()))?;

        parse_embedding_response(&json, &self.dimension)
    }

    /// Embed a single text input.
    pub async fn embed(&self, input: &str) -> Result<Vec<f32>, ExtensionError> {
        validate_input(input)?;
        let mut results = self.call_api(&[input]).await?;
        results
            .pop()
            .ok_or_else(|| ExtensionError::ParseError("empty response".into()))
    }

    /// Embed a batch of text inputs in a single API call.
    pub async fn embed_batch(&self, inputs: &[&str]) -> Result<Vec<Vec<f32>>, ExtensionError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        self.call_api(inputs).await
    }

    /// The dimensionality of vectors produced (0 until first call).
    pub fn dimension(&self) -> usize {
        self.dimension.get().copied().unwrap_or(0)
    }
}

#[cfg(feature = "async")]
#[async_trait::async_trait]
impl super::embedding::AsyncDenseEmbeddingFunction for AsyncHttpEmbedding {
    fn dimension(&self) -> usize {
        self.dimension.get().copied().unwrap_or(0)
    }

    async fn embed(&self, input: &str) -> Result<Vec<f32>, ExtensionError> {
        self.embed(input).await
    }

    async fn embed_batch(&self, inputs: &[&str]) -> Result<Vec<Vec<f32>>, ExtensionError> {
        self.embed_batch(inputs).await
    }
}
