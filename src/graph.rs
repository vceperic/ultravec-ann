//! Minimal NSW (navigable small-world) proximity graph—the HNSW base layer—used
//! to measure how decoded-vector fidelity under distribution drift affects graph
//! topology as well as query scores. Neighbours are selected by cosine distance on
//! each codec's decoded unit vectors at insertion time.
//!
//! This controlled single-layer NSW omits hierarchy, α-pruning, and disk layout.
//! It implements greedy-search insertion, bidirectional connections, and a degree
//! cap so graph recall can be compared with exhaustive search on the same decodes.

use crate::cosine;

/// One graph node: the stored (decoded) unit vector + its neighbour ids.
struct Node {
    vec: Vec<f32>,
    nbrs: Vec<u32>,
}

/// A minimal NSW graph over unit vectors. `m` = target degree (neighbours added per
/// insert), `ef_c` = construction beam, `ef_s` = search beam. The graph stores
/// whatever vector it is handed at insert time — for the drift experiment that is the
/// codec's *reconstruction*, so edge quality reflects codec fidelity at insert time.
pub struct NswGraph {
    nodes: Vec<Node>,
    m: usize,
    ef_c: usize,
    /// Diverse entry points sampled across insertion order. Under cluster-order
    /// insertion this spreads entries across clusters, so search can reach any
    /// region, approximating the navigability role of HNSW's upper layers in this
    /// controlled single-layer graph.
    entries: Vec<u32>,
    entry_stride: usize,
}

impl NswGraph {
    pub fn new(m: usize, ef_c: usize) -> Self {
        Self {
            nodes: Vec::new(),
            m,
            ef_c,
            entries: Vec::new(),
            entry_stride: 256, // ~1 entry per 256 inserts → diverse, cheap
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Greedy beam search from the entry point: returns up to `ef` (id, cosine) by
    /// descending similarity. Visits via a fixed-width frontier (the NSW/HNSW
    /// `search-layer` routine). Empty graph → empty.
    fn search_layer(&self, q: &[f32], ef: usize) -> Vec<(u32, f32)> {
        if self.nodes.is_empty() {
            return Vec::new();
        }
        let mut visited = vec![false; self.nodes.len()];
        // Seed the frontier from ALL diverse entry points (plus node 0 as a backstop),
        // so a clustered graph is still reachable from some entry near the query.
        let mut cand: Vec<(f32, u32)> = Vec::new();
        let mut top: Vec<(f32, u32)> = Vec::new();
        let seeds = if self.entries.is_empty() {
            vec![0u32]
        } else {
            self.entries.clone()
        };
        for e in seeds {
            if !visited[e as usize] {
                visited[e as usize] = true;
                let s = cosine(q, &self.nodes[e as usize].vec);
                cand.push((s, e));
                top.push((s, e));
            }
        }
        // keep top bounded at ef
        while top.len() > ef {
            let (wi, _) = top
                .iter()
                .enumerate()
                .min_by(|a, b| a.1 .0.partial_cmp(&b.1 .0).unwrap())
                .unwrap();
            top.swap_remove(wi);
        }

        while let Some((_, cur)) = pop_max(&mut cand) {
            // worst in current top set
            let worst = top.iter().map(|x| x.0).fold(f32::INFINITY, f32::min);
            let cur_sim = cosine(q, &self.nodes[cur as usize].vec);
            if cur_sim < worst && top.len() >= ef {
                break;
            }
            for &nb in &self.nodes[cur as usize].nbrs {
                if visited[nb as usize] {
                    continue;
                }
                visited[nb as usize] = true;
                let s = cosine(q, &self.nodes[nb as usize].vec);
                let worst2 = top.iter().map(|x| x.0).fold(f32::INFINITY, f32::min);
                if top.len() < ef || s > worst2 {
                    cand.push((s, nb));
                    top.push((s, nb));
                    if top.len() > ef {
                        // drop the worst
                        let (wi, _) = top
                            .iter()
                            .enumerate()
                            .min_by(|a, b| a.1 .0.partial_cmp(&b.1 .0).unwrap())
                            .unwrap();
                        top.swap_remove(wi);
                    }
                }
            }
        }
        top.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        top.into_iter().map(|(s, id)| (id, s)).collect()
    }

    /// Insert a (decoded) unit vector. Greedy-search for the `ef_c` nearest, connect
    /// to the closest `m` bidirectionally, and degree-cap each touched node at `m` by
    /// keeping its closest neighbours (the simple HNSW prune, no α-RobustPrune).
    pub fn insert(&mut self, vec: Vec<f32>) {
        let id = self.nodes.len() as u32;
        if self.nodes.is_empty() {
            self.nodes.push(Node {
                vec,
                nbrs: Vec::new(),
            });
            self.entries.push(0);
            return;
        }
        let neigh = self.search_layer(&vec, self.ef_c);
        let chosen: Vec<u32> = neigh.iter().take(self.m).map(|(i, _)| *i).collect();
        self.nodes.push(Node {
            vec,
            nbrs: chosen.clone(),
        });
        // bidirectional + degree cap
        for &nb in &chosen {
            self.nodes[nb as usize].nbrs.push(id);
            if self.nodes[nb as usize].nbrs.len() > self.m {
                self.prune(nb);
            }
        }
        // Register a diverse entry point every `entry_stride` inserts (spreads
        // entries across insertion order = across clusters under drift order).
        if (id as usize).is_multiple_of(self.entry_stride) {
            self.entries.push(id);
        }
    }

    /// Keep node `i`'s `m` closest neighbours (cosine on stored vectors).
    fn prune(&mut self, i: u32) {
        let iv = self.nodes[i as usize].vec.clone();
        let mut scored: Vec<(f32, u32)> = self.nodes[i as usize]
            .nbrs
            .iter()
            .map(|&n| (cosine(&iv, &self.nodes[n as usize].vec), n))
            .collect();
        scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(self.m);
        self.nodes[i as usize].nbrs = scored.into_iter().map(|(_, n)| n).collect();
    }

    /// Top-`k` node ids for `q` by graph search with beam `ef_s` (≥ k).
    pub fn search(&self, q: &[f32], k: usize, ef_s: usize) -> Vec<u32> {
        let mut r = self.search_layer(q, ef_s.max(k));
        r.truncate(k);
        r.into_iter().map(|(id, _)| id).collect()
    }
}

/// Pop the max-similarity candidate (linear; the frontiers are bounded by `ef`).
fn pop_max(v: &mut Vec<(f32, u32)>) -> Option<(f32, u32)> {
    if v.is_empty() {
        return None;
    }
    let (i, _) = v
        .iter()
        .enumerate()
        .max_by(|a, b| a.1 .0.partial_cmp(&b.1 .0).unwrap())
        .unwrap();
    Some(v.swap_remove(i))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{l2_norm, next_f64};

    fn rand_unit(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut st = seed;
        (0..n)
            .map(|_| {
                let mut v: Vec<f32> = (0..dim)
                    .map(|_| {
                        let u1 = next_f64(&mut st).max(1e-12);
                        let u2 = next_f64(&mut st);
                        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
                    })
                    .collect();
                let nn = l2_norm(&v).max(f32::EPSILON);
                v.iter_mut().for_each(|x| *x /= nn);
                v
            })
            .collect()
    }

    #[test]
    fn graph_recall_beats_threshold_on_exact_vectors() {
        // On exact (un-quantized) unit vectors, the NSW graph must recover most of the
        // true top-10 — the sanity gate that the index works before we feed it codec
        // reconstructions.
        let dim = 32;
        let db = rand_unit(2000, dim, 1);
        let q = rand_unit(50, dim, 999);
        let mut g = NswGraph::new(16, 64);
        for v in &db {
            g.insert(v.clone());
        }
        let mut hit = 0usize;
        let mut total = 0usize;
        for query in &q {
            // exact top-10
            let mut ex: Vec<(f32, u32)> = db
                .iter()
                .enumerate()
                .map(|(i, v)| (cosine(query, v), i as u32))
                .collect();
            ex.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            let gold: std::collections::HashSet<u32> =
                ex.iter().take(10).map(|(_, i)| *i).collect();
            let got = g.search(query, 10, 64);
            hit += got.iter().filter(|i| gold.contains(i)).count();
            total += 10;
        }
        let recall = hit as f64 / total as f64;
        assert!(
            recall > 0.85,
            "NSW graph recall@10 {recall} below 0.85 on exact vectors"
        );
    }

    #[test]
    fn empty_graph_search_is_empty() {
        let g = NswGraph::new(8, 32);
        assert!(g.search(&[1.0, 0.0], 5, 32).is_empty());
    }
}
