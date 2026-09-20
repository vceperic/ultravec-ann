//! Dataset loaders. Everything reduces to an in-memory `Dataset` of f32 vectors,
//! from which the benchmark carves a held-out query split.
//!
//! Source:
//!   * **fvecs** — the Texmex `.fvecs` format (SIFT1M/GIST1M/Deep1M): each record
//!     is `<i32 dim><dim×f32>`.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result};

use crate::next_f64;

/// An in-memory vector dataset.
pub struct Dataset {
    pub name: String,
    pub dim: usize,
    pub vectors: Vec<Vec<f32>>,
}

impl Dataset {
    pub fn len(&self) -> usize {
        self.vectors.len()
    }
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Deterministically split off `n_queries` held-out query vectors; the rest
    /// is the database. Returns `(database, queries)`. Seeded for reproducibility
    /// (research rule R10).
    pub fn split_queries(&self, n_queries: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let n = self.vectors.len();
        let n_queries = n_queries.min(n / 2);
        let mut idx: Vec<usize> = (0..n).collect();
        // Fisher-Yates with the splitmix PRNG.
        let mut state = seed;
        for i in (1..n).rev() {
            let j = (next_f64(&mut state) * (i as f64 + 1.0)) as usize;
            idx.swap(i, j.min(i));
        }
        let (q_idx, db_idx) = idx.split_at(n_queries);
        let queries = q_idx.iter().map(|&i| self.vectors[i].clone()).collect();
        let database = db_idx.iter().map(|&i| self.vectors[i].clone()).collect();
        (database, queries)
    }
}

/// Load a Texmex `.fvecs` file, capping at `max` vectors (0 = all). Streams
/// record-by-record so a `max` cap avoids reading the whole file (GIST base is
/// 3.8 GB). Each record is `<i32 dim><dim×f32>`.
pub fn load_fvecs(path: &Path, max: usize) -> Result<Dataset> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut r = std::io::BufReader::with_capacity(1 << 20, f);
    let mut vectors = Vec::new();
    let mut dim = 0usize;
    let mut hdr = [0u8; 4];
    loop {
        match r.read_exact(&mut hdr) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        let d = i32::from_le_bytes(hdr) as usize;
        anyhow::ensure!(d > 0 && d < 100_000, "implausible fvecs dim {d}");
        if dim == 0 {
            dim = d;
        }
        anyhow::ensure!(d == dim, "ragged fvecs: {d} != {dim}");
        let mut buf = vec![0u8; d * 4];
        r.read_exact(&mut buf)?;
        vectors.push(bytes_to_f32(&buf));
        if max != 0 && vectors.len() >= max {
            break;
        }
    }
    anyhow::ensure!(!vectors.is_empty(), "no vectors in {}", path.display());
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("fvecs")
        .to_string();
    let dim = pad_pow2(&mut vectors, dim);
    Ok(Dataset { name, dim, vectors })
}

/// Zero-pad every vector to the next power of two. Zeros preserve cosine exactly
/// (no contribution to norm or dot product) and make the blockwise rotation a
/// SINGLE uniform power-of-two block — avoiding the per-coordinate variance
/// skew that unequal greedy blocks (e.g. 960=512+256+128+64) cause, which breaks
/// the rescaled estimator on non-power-of-2 dims (GIST-960, OpenAI-1536, BERT-768).
/// Matches the official RaBitQ rotator, which also pads to a power of two.
pub fn pad_pow2(vectors: &mut [Vec<f32>], dim: usize) -> usize {
    let pd = dim.next_power_of_two();
    if pd != dim {
        for v in vectors.iter_mut() {
            v.resize(pd, 0.0);
        }
    }
    pd
}

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Load a Texmex `.ivecs` of per-query gold neighbour ids (each record
/// `<i32 len><len×i32>`; rows may differ in length — a relevant-set). Used by the
/// Load external per-query relevant-id sets, such as benchmark ground truth.
/// Fails visibly on a truncated record.
pub fn load_ivecs(path: &Path) -> Result<Vec<Vec<crate::ItemId>>> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut r = std::io::BufReader::with_capacity(1 << 20, f);
    let mut rows = Vec::new();
    let mut hdr = [0u8; 4];
    loop {
        match r.read_exact(&mut hdr) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        let n = i32::from_le_bytes(hdr) as usize;
        anyhow::ensure!(n < 10_000_000, "implausible ivecs row len {n}");
        let mut buf = vec![0u8; n * 4];
        r.read_exact(&mut buf)?;
        rows.push(
            buf.chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as crate::ItemId)
                .collect(),
        );
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn split_is_deterministic_and_disjoint() {
        let ds = Dataset {
            name: "t".into(),
            dim: 2,
            vectors: (0..100).map(|i| vec![i as f32, -(i as f32)]).collect(),
        };
        let (db1, q1) = ds.split_queries(20, 42);
        let (db2, q2) = ds.split_queries(20, 42);
        assert_eq!(q1.len(), 20);
        assert_eq!(db1.len(), 80);
        // Same seed → identical split.
        assert_eq!(q1, q2);
        assert_eq!(db1, db2);
    }

    #[test]
    fn fvecs_round_trip() {
        // Write a tiny .fvecs and read it back.
        let dir = std::env::temp_dir();
        let path = dir.join(format!("ultravec_test_{}.fvecs", std::process::id()));
        {
            let mut f = File::create(&path).unwrap();
            for rec in [[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]] {
                f.write_all(&3i32.to_le_bytes()).unwrap();
                for v in rec {
                    f.write_all(&v.to_le_bytes()).unwrap();
                }
            }
        }
        let ds = load_fvecs(&path, 0).unwrap();
        std::fs::remove_file(&path).ok();
        // dim 3 is padded to the next power of two (4) with a trailing zero;
        // zeros preserve cosine and give a single uniform rotation block.
        assert_eq!(ds.dim, 4);
        assert_eq!(ds.vectors.len(), 2);
        assert_eq!(ds.vectors[1], vec![4.0, 5.0, 6.0, 0.0]);
    }
}
