/// Quantization type selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantizationType {
    None,
    Fp16,
    Int8,
    Int4,
}

// ---------------------------------------------------------------------------
// FP16
// ---------------------------------------------------------------------------

/// Convert a single f32 to IEEE 754 half-precision (binary16).
pub fn f32_to_fp16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;

    if exponent == 255 {
        // Inf / NaN
        let fp16_mantissa = if mantissa != 0 { 0x0200 } else { 0 };
        return (sign | 0x7C00 | fp16_mantissa) as u16;
    }

    let unbiased = exponent - 127;

    if unbiased > 15 {
        // Overflow -> Inf
        return (sign | 0x7C00) as u16;
    }

    if unbiased < -24 {
        // Too small -> zero
        return sign as u16;
    }

    if unbiased < -14 {
        // Denormalized fp16
        let shift = -1 - unbiased;
        let m = (mantissa | 0x0080_0000) >> (shift + 13);
        return (sign | m) as u16;
    }

    let fp16_exp = ((unbiased + 15) as u32) << 10;
    let fp16_man = mantissa >> 13;
    (sign | fp16_exp | fp16_man) as u16
}

/// Convert a single IEEE 754 half-precision u16 back to f32.
pub fn fp16_to_f32(half: u16) -> f32 {
    let sign = ((half as u32) & 0x8000) << 16;
    let exponent = ((half >> 10) & 0x1F) as u32;
    let mantissa = (half & 0x03FF) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            // Zero
            return f32::from_bits(sign);
        }
        // Denormalized: normalize it
        let mut m = mantissa;
        let mut e: i32 = -14;
        while (m & 0x0400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x03FF; // remove implicit bit
        let f32_exp = ((e + 127) as u32) << 23;
        let f32_man = m << 13;
        return f32::from_bits(sign | f32_exp | f32_man);
    }

    if exponent == 31 {
        // Inf / NaN
        let f32_man = if mantissa != 0 { 0x0040_0000 } else { 0 };
        return f32::from_bits(sign | 0x7F80_0000 | f32_man);
    }

    let f32_exp = ((exponent as i32 - 15 + 127) as u32) << 23;
    let f32_man = mantissa << 13;
    f32::from_bits(sign | f32_exp | f32_man)
}

/// A vector quantized to IEEE 754 half-precision (16-bit) floats.
#[derive(Clone, Debug)]
pub struct Fp16Vec {
    data: Vec<u16>,
}

impl Fp16Vec {
    /// Quantize f32 data to fp16.
    pub fn from_f32(data: &[f32]) -> Self {
        Self {
            data: data.iter().map(|&v| f32_to_fp16(v)).collect(),
        }
    }

    /// Dequantize back to f32.
    pub fn to_f32(&self) -> Vec<f32> {
        self.data.iter().map(|&h| fp16_to_f32(h)).collect()
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

// ---------------------------------------------------------------------------
// INT8 (scalar quantization)
// ---------------------------------------------------------------------------

/// A vector quantized to 8-bit signed integers with affine scaling.
#[derive(Clone, Debug)]
pub struct Int8Vec {
    data: Vec<i8>,
    scale: f32,
    zero_point: f32,
}

impl Int8Vec {
    /// Quantize f32 data to int8 using min/max range scaling.
    pub fn from_f32(data: &[f32]) -> Self {
        if data.is_empty() {
            return Self {
                data: Vec::new(),
                scale: 1.0,
                zero_point: 0.0,
            };
        }
        let min = data.iter().copied().fold(f32::INFINITY, f32::min);
        let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = max - min;
        let scale = if range == 0.0 { 1.0 } else { range / 255.0 };
        let zero_point = min;

        let quantized = data
            .iter()
            .map(|&v| {
                let q = ((v - zero_point) / scale).round() as i32 - 128;
                q.clamp(-128, 127) as i8
            })
            .collect();

        Self {
            data: quantized,
            scale,
            zero_point,
        }
    }

    /// Dequantize back to f32.
    pub fn to_f32(&self) -> Vec<f32> {
        self.data
            .iter()
            .map(|&q| (q as i32 + 128) as f32 * self.scale + self.zero_point)
            .collect()
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Approximate dot product computed in the int8 domain.
    /// Accumulates in i32 then rescales.
    pub fn dot(a: &Int8Vec, b: &Int8Vec) -> f32 {
        assert_eq!(a.data.len(), b.data.len(), "vectors must have the same length");

        // Dot in quantized domain: sum of (qa + 128)(qb + 128) * scale_a * scale_b
        // plus offset corrections.
        //
        // Full expansion:
        // sum_i [ (qa_i + 128)*sa + zpa ] * [ (qb_i + 128)*sb + zpb ]
        //
        // For efficiency we compute the integer accumulation and then scale:
        let n = a.data.len();
        let mut acc: i64 = 0;
        let mut sum_a: i64 = 0;
        let mut sum_b: i64 = 0;
        for i in 0..n {
            let ai = a.data[i] as i64 + 128;
            let bi = b.data[i] as i64 + 128;
            acc += ai * bi;
            sum_a += ai;
            sum_b += bi;
        }

        let sa = a.scale;
        let sb = b.scale;
        let zpa = a.zero_point;
        let zpb = b.zero_point;
        let nf = n as f32;

        // Reconstruct: each element is (ai*sa + zpa) * (bi*sb + zpb)
        // = sa*sb*ai*bi + sa*zpb*ai + sb*zpa*bi + zpa*zpb
        sa * sb * acc as f32
            + sa * zpb * sum_a as f32
            + sb * zpa * sum_b as f32
            + zpa * zpb * nf
    }
}

// ---------------------------------------------------------------------------
// INT4 (extreme compression — 2 values per byte)
// ---------------------------------------------------------------------------

/// A vector quantized to 4-bit unsigned integers, packed two per byte.
/// High nibble stores even-indexed values, low nibble stores odd-indexed values.
#[derive(Clone, Debug)]
pub struct Int4Vec {
    data: Vec<u8>,
    len: usize,
    scale: f32,
    zero_point: f32,
}

impl Int4Vec {
    /// Quantize f32 data to 4-bit values (0..15), packed two per byte.
    pub fn from_f32(data: &[f32]) -> Self {
        if data.is_empty() {
            return Self {
                data: Vec::new(),
                len: 0,
                scale: 1.0,
                zero_point: 0.0,
            };
        }
        let min = data.iter().copied().fold(f32::INFINITY, f32::min);
        let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = max - min;
        let scale = if range == 0.0 { 1.0 } else { range / 15.0 };
        let zero_point = min;

        let nibbles: Vec<u8> = data
            .iter()
            .map(|&v| {
                let q = ((v - zero_point) / scale).round() as i32;
                q.clamp(0, 15) as u8
            })
            .collect();

        let packed_len = (nibbles.len() + 1) / 2;
        let mut packed = vec![0u8; packed_len];
        for (i, &nib) in nibbles.iter().enumerate() {
            if i % 2 == 0 {
                packed[i / 2] |= nib << 4;
            } else {
                packed[i / 2] |= nib;
            }
        }

        Self {
            data: packed,
            len: data.len(),
            scale,
            zero_point,
        }
    }

    /// Dequantize back to f32.
    pub fn to_f32(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.len);
        for i in 0..self.len {
            let byte = self.data[i / 2];
            let nibble = if i % 2 == 0 {
                (byte >> 4) & 0x0F
            } else {
                byte & 0x0F
            };
            result.push(nibble as f32 * self.scale + self.zero_point);
        }
        result
    }

    /// Number of logical elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- FP16 ---------------------------------------------------------------

    #[test]
    fn test_fp16_roundtrip() {
        let values = vec![0.0, 1.0, -1.0, 0.5, 65504.0, -65504.0, 0.001, 3.14];
        let fp16 = Fp16Vec::from_f32(&values);
        assert_eq!(fp16.len(), values.len());
        let back = fp16.to_f32();
        for (orig, recovered) in values.iter().zip(back.iter()) {
            let tol = orig.abs() * 0.002 + 1e-4; // fp16 has ~0.1% relative error
            assert!(
                (orig - recovered).abs() < tol,
                "fp16 roundtrip failed: {} -> {}",
                orig,
                recovered
            );
        }
    }

    #[test]
    fn test_fp16_zero() {
        let fp16 = Fp16Vec::from_f32(&[0.0, -0.0]);
        let back = fp16.to_f32();
        assert_eq!(back[0], 0.0);
        assert_eq!(back[1], 0.0); // sign may differ but value is 0
    }

    #[test]
    fn test_fp16_inf_nan() {
        let fp16 = Fp16Vec::from_f32(&[f32::INFINITY, f32::NEG_INFINITY, f32::NAN]);
        let back = fp16.to_f32();
        assert!(back[0].is_infinite() && back[0] > 0.0);
        assert!(back[1].is_infinite() && back[1] < 0.0);
        assert!(back[2].is_nan());
    }

    #[test]
    fn test_fp16_empty() {
        let fp16 = Fp16Vec::from_f32(&[]);
        assert_eq!(fp16.len(), 0);
        assert!(fp16.is_empty());
    }

    // -- INT8 ---------------------------------------------------------------

    #[test]
    fn test_int8_roundtrip() {
        let values: Vec<f32> = (0..256).map(|i| i as f32 / 255.0).collect();
        let int8 = Int8Vec::from_f32(&values);
        assert_eq!(int8.len(), 256);
        let back = int8.to_f32();
        for (orig, recovered) in values.iter().zip(back.iter()) {
            assert!(
                (orig - recovered).abs() < 0.01,
                "int8 roundtrip error too large: {} vs {}",
                orig,
                recovered
            );
        }
    }

    #[test]
    fn test_int8_constant_vector() {
        let values = vec![5.0; 100];
        let int8 = Int8Vec::from_f32(&values);
        let back = int8.to_f32();
        for &v in &back {
            assert!((v - 5.0).abs() < 1e-3);
        }
    }

    #[test]
    fn test_int8_single_element() {
        let int8 = Int8Vec::from_f32(&[42.0]);
        let back = int8.to_f32();
        assert!((back[0] - 42.0).abs() < 1e-3);
    }

    #[test]
    fn test_int8_empty() {
        let int8 = Int8Vec::from_f32(&[]);
        assert_eq!(int8.len(), 0);
        assert!(int8.is_empty());
    }

    #[test]
    fn test_int8_dot_product() {
        let a_vals: Vec<f32> = (0..128).map(|i| (i as f32) * 0.1).collect();
        let b_vals: Vec<f32> = (0..128).map(|i| (127 - i) as f32 * 0.1).collect();

        let exact_dot: f32 = a_vals.iter().zip(b_vals.iter()).map(|(a, b)| a * b).sum();

        let a_q = Int8Vec::from_f32(&a_vals);
        let b_q = Int8Vec::from_f32(&b_vals);
        let approx_dot = Int8Vec::dot(&a_q, &b_q);

        let relative_error = ((exact_dot - approx_dot) / exact_dot).abs();
        assert!(
            relative_error < 0.02,
            "int8 dot product relative error too large: {} (exact={}, approx={})",
            relative_error,
            exact_dot,
            approx_dot
        );
    }

    #[test]
    fn test_int8_dot_zero_vector() {
        let a = Int8Vec::from_f32(&[0.0; 64]);
        let b = Int8Vec::from_f32(&[1.0; 64]);
        let dot = Int8Vec::dot(&a, &b);
        assert!(dot.abs() < 1.0, "dot with zero vector should be near zero, got {}", dot);
    }

    // -- INT4 ---------------------------------------------------------------

    #[test]
    fn test_int4_roundtrip() {
        let values: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let int4 = Int4Vec::from_f32(&values);
        assert_eq!(int4.len(), 16);
        let back = int4.to_f32();
        for (orig, recovered) in values.iter().zip(back.iter()) {
            assert!(
                (orig - recovered).abs() < 1.0,
                "int4 roundtrip error too large: {} vs {}",
                orig,
                recovered
            );
        }
    }

    #[test]
    fn test_int4_odd_length() {
        let values = vec![1.0, 2.0, 3.0];
        let int4 = Int4Vec::from_f32(&values);
        assert_eq!(int4.len(), 3);
        let back = int4.to_f32();
        for (orig, recovered) in values.iter().zip(back.iter()) {
            assert!(
                (orig - recovered).abs() < 1.0,
                "int4 odd roundtrip: {} vs {}",
                orig,
                recovered
            );
        }
    }

    #[test]
    fn test_int4_empty() {
        let int4 = Int4Vec::from_f32(&[]);
        assert_eq!(int4.len(), 0);
        assert!(int4.is_empty());
    }

    #[test]
    fn test_int4_single_element() {
        let int4 = Int4Vec::from_f32(&[7.0]);
        assert_eq!(int4.len(), 1);
        let back = int4.to_f32();
        assert!((back[0] - 7.0).abs() < 1e-3);
    }

    #[test]
    fn test_int4_large_dimension() {
        let values: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.01).collect();
        let int4 = Int4Vec::from_f32(&values);
        assert_eq!(int4.len(), 1000);
        let back = int4.to_f32();
        let max_err: f32 = values
            .iter()
            .zip(back.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // 4-bit quantization over range [0, 9.99] -> step ~0.666
        assert!(max_err < 1.0, "int4 max error {} too large", max_err);
    }

    // -- QuantizationType ---------------------------------------------------

    #[test]
    fn test_quantization_type_clone_debug() {
        let qt = QuantizationType::Fp16;
        let qt2 = qt;
        assert_eq!(qt, qt2);
        let _s = format!("{:?}", qt);
    }
}
