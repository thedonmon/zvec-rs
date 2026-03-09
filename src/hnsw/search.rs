use ordered_float::OrderedFloat;

/// A search result with ID, distance/similarity score, and optional label.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: u64,
    pub score: f32,
}

impl SearchResult {
    pub fn new(id: u64, score: f32) -> Self {
        Self { id, score }
    }
}

/// Internal candidate used during search. Ordered by distance.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) struct Candidate {
    pub distance: OrderedFloat<f32>,
    pub id: u32,
}

impl Candidate {
    pub fn new(distance: f32, id: u32) -> Self {
        Self {
            distance: OrderedFloat(distance),
            id,
        }
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse ordering: smallest distance first (min-heap behavior with BinaryHeap)
        other.distance.cmp(&self.distance)
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Max-distance candidate (for the "worst in result set" heap).
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) struct FarCandidate {
    pub distance: OrderedFloat<f32>,
    pub id: u32,
}

impl FarCandidate {
    pub fn new(distance: f32, id: u32) -> Self {
        Self {
            distance: OrderedFloat(distance),
            id,
        }
    }
}

impl Ord for FarCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Normal ordering: largest distance first (max-heap with BinaryHeap)
        self.distance.cmp(&other.distance)
    }
}

impl PartialOrd for FarCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
