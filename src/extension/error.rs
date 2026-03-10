/// Errors from embedding or reranking operations.
#[derive(Debug)]
pub enum ExtensionError {
    /// Input validation failure (empty text, wrong type).
    InvalidInput(String),
    /// Network or HTTP-level failure.
    Network(String),
    /// API returned an error response.
    Api { status: u16, message: String },
    /// Response could not be parsed.
    ParseError(String),
    /// Dimension mismatch between expected and actual.
    DimensionMismatch { expected: usize, actual: usize },
    /// No results to rerank.
    EmptyInput,
    /// Generic / other error.
    Other(String),
}

impl std::fmt::Display for ExtensionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Self::Network(msg) => write!(f, "network error: {msg}"),
            Self::Api { status, message } => write!(f, "API error (HTTP {status}): {message}"),
            Self::ParseError(msg) => write!(f, "parse error: {msg}"),
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
            Self::EmptyInput => write!(f, "no results to rerank"),
            Self::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for ExtensionError {}
