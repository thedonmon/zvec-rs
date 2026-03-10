use super::error::ExtensionError;

/// Shared configuration for OpenAI-compatible embedding clients.
///
/// Works with OpenAI's API and any compatible endpoint (Ollama, vLLM,
/// Azure OpenAI, LiteLLM, etc.) by setting `base_url`.
#[derive(Clone)]
struct OpenAiConfig {
    api_key: String,
    model: String,
    dimension: usize,
    base_url: String,
}

/// Parse an OpenAI-format embedding response body.
fn parse_embedding_response(
    json: &serde_json::Value,
    expected_dim: usize,
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

        if vec.len() != expected_dim {
            return Err(ExtensionError::DimensionMismatch {
                expected: expected_dim,
                actual: vec.len(),
            });
        }
        vectors.push(vec);
    }

    Ok(vectors)
}

fn validate_inputs(inputs: &[&str]) -> Result<(), ExtensionError> {
    for (i, input) in inputs.iter().enumerate() {
        if input.is_empty() {
            return Err(ExtensionError::InvalidInput(format!(
                "input at index {i} is empty"
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Sync implementation (blocking HTTP)
// ---------------------------------------------------------------------------

/// Synchronous OpenAI-compatible embedding client using blocking HTTP.
///
/// **Warning**: Do not use this inside an async runtime (tokio) — it will
/// panic. Use [`AsyncOpenAiEmbedding`] instead.
///
/// # Example
/// ```no_run
/// use zvec_rs::extension::OpenAiEmbedding;
/// use zvec_rs::DenseEmbeddingFunction;
///
/// let embedder = OpenAiEmbedding::new(
///     "sk-...".to_string(),
///     "text-embedding-3-small".to_string(),
///     1536,
/// );
/// let vector = embedder.embed("hello world").unwrap();
/// assert_eq!(vector.len(), 1536);
/// ```
pub struct OpenAiEmbedding {
    config: OpenAiConfig,
    client: reqwest::blocking::Client,
}

impl OpenAiEmbedding {
    /// Create with default base URL (`https://api.openai.com/v1`).
    pub fn new(api_key: String, model: String, dimension: usize) -> Self {
        Self {
            config: OpenAiConfig {
                api_key,
                model,
                dimension,
                base_url: "https://api.openai.com/v1".to_string(),
            },
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Create with a custom base URL (for compatible APIs like Ollama, vLLM).
    pub fn with_base_url(
        api_key: String,
        model: String,
        dimension: usize,
        base_url: String,
    ) -> Self {
        Self {
            config: OpenAiConfig {
                api_key,
                model,
                dimension,
                base_url,
            },
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Create from environment variable `OPENAI_API_KEY`.
    pub fn from_env(model: String, dimension: usize) -> Result<Self, ExtensionError> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| ExtensionError::InvalidInput("OPENAI_API_KEY not set".into()))?;
        Ok(Self::new(api_key, model, dimension))
    }

    fn call_api(&self, inputs: &[&str]) -> Result<Vec<Vec<f32>>, ExtensionError> {
        let url = format!("{}/embeddings", self.config.base_url);
        let body = serde_json::json!({
            "model": self.config.model,
            "input": inputs,
        });

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .map_err(|e| ExtensionError::Network(e.to_string()))?;

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

        parse_embedding_response(&json, self.config.dimension)
    }
}

impl super::embedding::DenseEmbeddingFunction for OpenAiEmbedding {
    fn dimension(&self) -> usize {
        self.config.dimension
    }

    fn embed(&self, input: &str) -> Result<Vec<f32>, ExtensionError> {
        validate_inputs(&[input])?;
        let mut results = self.call_api(&[input])?;
        results
            .pop()
            .ok_or_else(|| ExtensionError::ParseError("empty response".into()))
    }

    fn embed_batch(&self, inputs: &[&str]) -> Result<Vec<Vec<f32>>, ExtensionError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        validate_inputs(inputs)?;
        self.call_api(inputs)
    }
}

// ---------------------------------------------------------------------------
// Async implementation
// ---------------------------------------------------------------------------

/// Async OpenAI-compatible embedding client.
///
/// This is the **preferred** client for use in async applications (tokio, etc.).
///
/// # Example
/// ```no_run
/// # async fn example() {
/// use zvec_rs::extension::AsyncOpenAiEmbedding;
///
/// let embedder = AsyncOpenAiEmbedding::new(
///     "sk-...".to_string(),
///     "text-embedding-3-small".to_string(),
///     1536,
/// );
/// let vector = embedder.embed("hello world").await.unwrap();
/// assert_eq!(vector.len(), 1536);
/// # }
/// ```
#[cfg(feature = "async")]
pub struct AsyncOpenAiEmbedding {
    config: OpenAiConfig,
    client: reqwest::Client,
}

#[cfg(feature = "async")]
impl AsyncOpenAiEmbedding {
    /// Create with default base URL (`https://api.openai.com/v1`).
    pub fn new(api_key: String, model: String, dimension: usize) -> Self {
        Self {
            config: OpenAiConfig {
                api_key,
                model,
                dimension,
                base_url: "https://api.openai.com/v1".to_string(),
            },
            client: reqwest::Client::new(),
        }
    }

    /// Create with a custom base URL.
    pub fn with_base_url(
        api_key: String,
        model: String,
        dimension: usize,
        base_url: String,
    ) -> Self {
        Self {
            config: OpenAiConfig {
                api_key,
                model,
                dimension,
                base_url,
            },
            client: reqwest::Client::new(),
        }
    }

    /// Create from environment variable `OPENAI_API_KEY`.
    pub fn from_env(model: String, dimension: usize) -> Result<Self, ExtensionError> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| ExtensionError::InvalidInput("OPENAI_API_KEY not set".into()))?;
        Ok(Self::new(api_key, model, dimension))
    }

    async fn call_api(&self, inputs: &[&str]) -> Result<Vec<Vec<f32>>, ExtensionError> {
        let url = format!("{}/embeddings", self.config.base_url);
        let body = serde_json::json!({
            "model": self.config.model,
            "input": inputs,
        });

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
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

        parse_embedding_response(&json, self.config.dimension)
    }

    /// Embed a single text input.
    pub async fn embed(&self, input: &str) -> Result<Vec<f32>, ExtensionError> {
        validate_inputs(&[input])?;
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
        validate_inputs(inputs)?;
        self.call_api(inputs).await
    }

    /// The dimensionality of vectors produced.
    pub fn dimension(&self) -> usize {
        self.config.dimension
    }
}

#[cfg(feature = "async")]
#[async_trait::async_trait]
impl super::embedding::AsyncDenseEmbeddingFunction for AsyncOpenAiEmbedding {
    fn dimension(&self) -> usize {
        self.config.dimension
    }

    async fn embed(&self, input: &str) -> Result<Vec<f32>, ExtensionError> {
        self.embed(input).await
    }

    async fn embed_batch(&self, inputs: &[&str]) -> Result<Vec<Vec<f32>>, ExtensionError> {
        self.embed_batch(inputs).await
    }
}
