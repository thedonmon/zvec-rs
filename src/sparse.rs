/// A sparse vector represented as parallel arrays of indices and values.
#[derive(Clone, Debug)]
pub struct SparseVector {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

impl SparseVector {
    /// Create a new sparse vector from indices and values.
    /// Validates that both slices have the same length and sorts entries by index.
    pub fn new(indices: Vec<u32>, values: Vec<f32>) -> Result<Self, &'static str> {
        if indices.len() != values.len() {
            return Err("indices and values must have the same length");
        }
        // Sort by index using a permutation sort
        let mut pairs: Vec<(u32, f32)> = indices.into_iter().zip(values).collect();
        pairs.sort_by_key(|(idx, _)| *idx);
        // Check for duplicate indices
        for w in pairs.windows(2) {
            if w[0].0 == w[1].0 {
                return Err("duplicate indices are not allowed");
            }
        }
        let (indices, values): (Vec<u32>, Vec<f32>) = pairs.into_iter().unzip();
        Ok(Self { indices, values })
    }

    /// Number of non-zero elements.
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Sparse dot product using sorted merge.
    pub fn dot(a: &SparseVector, b: &SparseVector) -> f32 {
        let mut i = 0;
        let mut j = 0;
        let mut result = 0.0f32;
        while i < a.indices.len() && j < b.indices.len() {
            if a.indices[i] == b.indices[j] {
                result += a.values[i] * b.values[j];
                i += 1;
                j += 1;
            } else if a.indices[i] < b.indices[j] {
                i += 1;
            } else {
                j += 1;
            }
        }
        result
    }

    /// Sparse L2 squared distance using sorted merge.
    pub fn l2_squared(a: &SparseVector, b: &SparseVector) -> f32 {
        let mut i = 0;
        let mut j = 0;
        let mut result = 0.0f32;
        while i < a.indices.len() && j < b.indices.len() {
            if a.indices[i] == b.indices[j] {
                let diff = a.values[i] - b.values[j];
                result += diff * diff;
                i += 1;
                j += 1;
            } else if a.indices[i] < b.indices[j] {
                result += a.values[i] * a.values[i];
                i += 1;
            } else {
                result += b.values[j] * b.values[j];
                j += 1;
            }
        }
        // Drain remaining elements
        while i < a.indices.len() {
            result += a.values[i] * a.values[i];
            i += 1;
        }
        while j < b.indices.len() {
            result += b.values[j] * b.values[j];
            j += 1;
        }
        result
    }

    /// Sparse cosine similarity.
    pub fn cosine(a: &SparseVector, b: &SparseVector) -> f32 {
        let dot = Self::dot(a, b);
        let norm_a: f32 = a.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm_b: f32 = b.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }

    /// Convert to a dense vector of the given dimensionality.
    pub fn to_dense(&self, dims: usize) -> Vec<f32> {
        let mut dense = vec![0.0f32; dims];
        for (&idx, &val) in self.indices.iter().zip(self.values.iter()) {
            if (idx as usize) < dims {
                dense[idx as usize] = val;
            }
        }
        dense
    }

    /// Create a sparse vector from a dense vector, skipping values whose
    /// absolute value is below the given threshold.
    pub fn from_dense(dense: &[f32], threshold: f32) -> Self {
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for (i, &val) in dense.iter().enumerate() {
            if val.abs() >= threshold {
                indices.push(i as u32);
                values.push(val);
            }
        }
        Self { indices, values }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_construction() {
        let sv = SparseVector::new(vec![3, 1, 5], vec![0.3, 0.1, 0.5]).unwrap();
        assert_eq!(sv.indices, vec![1, 3, 5]);
        assert_eq!(sv.values, vec![0.1, 0.3, 0.5]);
        assert_eq!(sv.nnz(), 3);
    }

    #[test]
    fn test_mismatched_lengths() {
        let result = SparseVector::new(vec![1, 2], vec![0.1]);
        assert!(result.is_err());
    }

    #[test]
    fn test_duplicate_indices() {
        let result = SparseVector::new(vec![1, 1], vec![0.1, 0.2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty() {
        let sv = SparseVector::new(vec![], vec![]).unwrap();
        assert_eq!(sv.nnz(), 0);
    }

    #[test]
    fn test_dot_overlapping() {
        let a = SparseVector::new(vec![0, 2, 4], vec![1.0, 2.0, 3.0]).unwrap();
        let b = SparseVector::new(vec![1, 2, 4], vec![10.0, 20.0, 30.0]).unwrap();
        // overlap at 2 and 4: 2*20 + 3*30 = 40 + 90 = 130
        let dot = SparseVector::dot(&a, &b);
        assert!((dot - 130.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_non_overlapping() {
        let a = SparseVector::new(vec![0, 2], vec![1.0, 2.0]).unwrap();
        let b = SparseVector::new(vec![1, 3], vec![10.0, 20.0]).unwrap();
        let dot = SparseVector::dot(&a, &b);
        assert!((dot - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_empty() {
        let a = SparseVector::new(vec![], vec![]).unwrap();
        let b = SparseVector::new(vec![1, 3], vec![10.0, 20.0]).unwrap();
        assert!((SparseVector::dot(&a, &b) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_squared() {
        // a = [1, 0, 2], b = [0, 3, 2]
        // diff^2 = 1 + 9 + 0 = 10
        let a = SparseVector::new(vec![0, 2], vec![1.0, 2.0]).unwrap();
        let b = SparseVector::new(vec![1, 2], vec![3.0, 2.0]).unwrap();
        let dist = SparseVector::l2_squared(&a, &b);
        assert!((dist - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_squared_identical() {
        let a = SparseVector::new(vec![0, 5], vec![1.0, 2.0]).unwrap();
        let dist = SparseVector::l2_squared(&a, &a);
        assert!((dist - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = SparseVector::new(vec![0, 1], vec![1.0, 0.0]).unwrap();
        let b = SparseVector::new(vec![0, 1], vec![1.0, 0.0]).unwrap();
        let cos = SparseVector::cosine(&a, &b);
        assert!((cos - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = SparseVector::new(vec![0], vec![1.0]).unwrap();
        let b = SparseVector::new(vec![1], vec![1.0]).unwrap();
        let cos = SparseVector::cosine(&a, &b);
        assert!((cos - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_zero_vector() {
        let a = SparseVector::new(vec![], vec![]).unwrap();
        let b = SparseVector::new(vec![0], vec![1.0]).unwrap();
        assert_eq!(SparseVector::cosine(&a, &b), 0.0);
    }

    #[test]
    fn test_dense_roundtrip() {
        let dense = vec![0.0, 1.5, 0.0, 0.0, 2.5, 0.0, 3.0];
        let sparse = SparseVector::from_dense(&dense, 1e-6);
        assert_eq!(sparse.indices, vec![1, 4, 6]);
        assert_eq!(sparse.values, vec![1.5, 2.5, 3.0]);
        let back = sparse.to_dense(7);
        assert_eq!(back, dense);
    }

    #[test]
    fn test_from_dense_with_threshold() {
        let dense = vec![0.01, 0.5, 0.001, 1.0];
        let sparse = SparseVector::from_dense(&dense, 0.05);
        assert_eq!(sparse.indices, vec![1, 3]);
        assert_eq!(sparse.values, vec![0.5, 1.0]);
    }

    #[test]
    fn test_to_dense_truncates() {
        let sv = SparseVector::new(vec![0, 100], vec![1.0, 2.0]).unwrap();
        let dense = sv.to_dense(10);
        assert_eq!(dense.len(), 10);
        assert_eq!(dense[0], 1.0);
        // index 100 is beyond dims=10, so it's dropped
    }

    #[test]
    fn test_clone_debug() {
        let sv = SparseVector::new(vec![0], vec![1.0]).unwrap();
        let sv2 = sv.clone();
        assert_eq!(sv2.indices, sv.indices);
        let _debug = format!("{:?}", sv);
    }
}
