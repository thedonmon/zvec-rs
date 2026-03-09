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

    /// Use heuristic neighbor selection (Algorithm 4 from the paper).
    /// Better recall at the cost of slightly slower inserts.
    pub use_heuristic: bool,

    /// Extend candidates by adding their neighbors during selection.
    /// Only used when `use_heuristic` is true.
    pub extend_candidates: bool,
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
            use_heuristic: true,
            extend_candidates: false,
        }
    }

    pub fn with_ef_search(mut self, ef: usize) -> Self {
        self.ef_search = ef;
        self
    }

    pub fn with_heuristic(mut self, use_heuristic: bool) -> Self {
        self.use_heuristic = use_heuristic;
        self
    }

    pub fn with_extend_candidates(mut self, extend: bool) -> Self {
        self.extend_candidates = extend;
        self
    }
}

impl Default for HnswParams {
    fn default() -> Self {
        Self::new(32, 200)
    }
}
