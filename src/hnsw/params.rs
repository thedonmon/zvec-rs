/// HNSW construction and search parameters.
#[derive(Debug, Clone)]
pub struct HnswParams {
    /// Max connections per node at layer 0.
    /// Higher = better recall, more memory, slower insert.
    /// Typical: 16-64. Default: 32.
    pub m: usize,

    /// Max connections per node at layers > 0.
    /// Usually m / 2 by convention.
    pub m_max0: usize,

    /// Size of the dynamic candidate list during construction.
    /// Higher = better graph quality, slower insert.
    /// Typical: 100-500. Default: 200.
    pub ef_construction: usize,

    /// Size of the dynamic candidate list during search.
    /// Higher = better recall, slower search.
    /// Must be >= top_k. Default: 50.
    pub ef_search: usize,

    /// Normalization factor for level generation: 1 / ln(M).
    pub(crate) ml: f64,
}

impl HnswParams {
    pub fn new(m: usize, ef_construction: usize) -> Self {
        let m_max0 = m * 2;
        Self {
            m,
            m_max0,
            ef_construction,
            ef_search: 50,
            ml: 1.0 / (m as f64).ln(),
        }
    }

    pub fn with_ef_search(mut self, ef: usize) -> Self {
        self.ef_search = ef;
        self
    }
}

impl Default for HnswParams {
    fn default() -> Self {
        Self::new(32, 200)
    }
}
