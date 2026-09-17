//! TurboQuant MSE reconstruction reference: N(0,1) Lloyd-Max codebooks,
//! blockwise Walsh-Hadamard rotation, and asymmetric inner-product scoring.
//! The artifact-native implementation keeps the comparison configuration stable.

use crate::{
    codebook, dist::Gaussian, l2_norm, nearest_index, pack_indices, std_about_mean, unpack_indices,
    ItemId, MemoryBreakdown, Rotation, VectorBackend,
};

struct QuantizedVec {
    norm: f32,
    scale: f32,
    codes: Vec<u8>,
}

/// TurboQuant MSE reference backend (N(0,1) centroids).
pub struct TurboQuantBaseline {
    dim: usize,
    bits: u8,
    rotation: Rotation,
    codebook: Vec<f32>,
    entries: Vec<(ItemId, QuantizedVec)>,
}

impl TurboQuantBaseline {
    pub fn new(dim: usize, bits: u8) -> Self {
        // 4/5/6 use the published TurboQuant tables; other widths derive
        // the N(0,1) Lloyd-Max codebook with the validated generator (still
        // exactly TurboQuant's design, just not pre-tabulated).
        let codebook = match bits {
            4..=6 => codebook::gaussian(bits).to_vec(),
            _ => codebook::lloyd_max(&Gaussian, bits, 500, 1e-12).centroids,
        };
        Self {
            dim,
            bits,
            rotation: Rotation::new(dim, crate::rotation_seed()),
            codebook,
            entries: Vec::new(),
        }
    }

    fn quantize(&self, x: &[f32]) -> QuantizedVec {
        let norm = l2_norm(x);
        if norm < f32::EPSILON {
            return QuantizedVec {
                norm: 0.0,
                scale: 1.0,
                codes: pack_indices(&vec![0u16; self.dim], self.bits),
            };
        }
        let normalized: Vec<f32> = x.iter().map(|v| v / norm).collect();
        let rotated = self.rotation.apply(&normalized);
        let mut scale = std_about_mean(&rotated);
        if scale < f32::EPSILON {
            scale = 1.0;
        }
        let indices: Vec<u16> = rotated
            .iter()
            .map(|&v| nearest_index(v / scale, &self.codebook))
            .collect();
        QuantizedVec {
            norm,
            scale,
            codes: pack_indices(&indices, self.bits),
        }
    }

    fn score(&self, q_rot: &[f32], qv: &QuantizedVec, scratch: &mut [u16]) -> f32 {
        if qv.norm < f32::EPSILON {
            return 0.0;
        }
        unpack_indices(&qv.codes, self.bits, scratch);
        let mut dot = 0.0f32;
        for (c, &code) in q_rot.iter().zip(scratch.iter()) {
            dot += c * self.codebook[code as usize];
        }
        qv.scale * dot
    }
}

impl VectorBackend for TurboQuantBaseline {
    fn dimensions(&self) -> usize {
        self.dim
    }
    fn len(&self) -> usize {
        self.entries.len()
    }
    fn add(&mut self, id: ItemId, embedding: &[f32]) {
        assert_eq!(embedding.len(), self.dim);
        let q = self.quantize(embedding);
        self.entries.push((id, q));
    }
    fn reserve(&mut self, additional: usize) {
        self.entries.reserve_exact(additional);
    }
    fn add_batch(&mut self, embeddings: &[Vec<f32>]) {
        use rayon::prelude::*;
        let dim = self.dim;
        let this: &TurboQuantBaseline = self;
        let mut entries = embeddings
            .par_iter()
            .enumerate()
            .map(|(i, embedding)| {
                assert_eq!(embedding.len(), dim);
                (i as ItemId, this.quantize(embedding))
            })
            .collect();
        self.entries.append(&mut entries);
    }
    fn search(&self, query: &[f32], limit: usize) -> Vec<(ItemId, f32)> {
        let qnorm = l2_norm(query);
        if qnorm < f32::EPSILON {
            return Vec::new();
        }
        let normalized: Vec<f32> = query.iter().map(|v| v / qnorm).collect();
        let q_rot = self.rotation.apply(&normalized);
        let mut scratch = vec![0u16; self.dim];
        let mut results: Vec<(ItemId, f32)> = self
            .entries
            .iter()
            .map(|(id, qv)| (*id, self.score(&q_rot, qv, &mut scratch)))
            .collect();
        results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }
    fn mem_bytes(&self) -> usize {
        self.entries.iter().map(|(_, q)| q.codes.len() + 8).sum()
    }
    fn memory_breakdown(&self) -> MemoryBreakdown {
        MemoryBreakdown {
            code_bytes: self.mem_bytes(),
            model_bytes: self.rotation.allocated_bytes()
                + self.codebook.capacity() * std::mem::size_of::<f32>(),
            index_bytes: self.entries.capacity() * std::mem::size_of::<(ItemId, QuantizedVec)>(),
            ..MemoryBreakdown::default()
        }
    }
    fn is_approximate(&self) -> bool {
        true
    }
    fn reconstruct_unit(&self, x: &[f32]) -> Option<Vec<f32>> {
        let qv = self.quantize(x);
        if qv.norm < f32::EPSILON {
            return Some(vec![0.0; self.dim]);
        }
        let mut idx = vec![0u16; self.dim];
        unpack_indices(&qv.codes, self.bits, &mut idx);
        let rotated: Vec<f32> = idx
            .iter()
            .map(|&i| self.codebook[i as usize] * qv.scale)
            .collect();
        Some(self.rotation.apply_inverse(&rotated))
    }
}
