#!/usr/bin/env python3
"""Faiss IVFPQ (and IVFPQ+refine) recall@10 vs QPS on SIFT-1M.

This produces the pinned mature-system reference used beside the controlled
UltraVec IVF study. Configuration: IVF nlist=1024,
PQ M=32 nbits=8 (=32 B, the 2-bit/dim analogue of the oblivious 36-40 B codecs),
20 threads, exact-cosine (IP on L2-normalized) gold, nprobe sweep.

Run: python3 scripts/faiss_ivfpq_qps.py
"""
import time
from pathlib import Path
import numpy as np
import faiss

ROOT = Path(__file__).resolve().parents[1]
SIFT = ROOT / "data" / "sift"
NLIST = 1024
M_PQ = 32          # 32 bytes = 2 bits/dim on 128-d, matched to the oblivious codecs
THREADS = 20
NPROBES = [1, 2, 4, 8, 16, 32, 64]
K = 10
QUERY_MAX = 200
MIN_TIMING_SECONDS = 0.5
SEED = 42

def read_fvecs(p):
    a = np.fromfile(p, dtype=np.int32)
    d = a[0]
    return np.ascontiguousarray(a.reshape(-1, d + 1)[:, 1:].view(np.float32))

def read_ivecs(p):
    a = np.fromfile(p, dtype=np.int32)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:]

def main():
    faiss.omp_set_num_threads(THREADS)
    DB = read_fvecs(f"{SIFT}/sift_base.fvecs")
    Q = read_fvecs(f"{SIFT}/sift_query.fvecs")[:QUERY_MAX]
    faiss.normalize_L2(DB); faiss.normalize_L2(Q)
    d = DB.shape[1]
    print(f"SIFT-1M: {DB.shape[0]} base, {Q.shape[0]} queries, dim {d}, {THREADS} threads")

    # exact-cosine gold on the normalized vectors (IP), top-K
    print("computing exact gold ...", flush=True)
    flat = faiss.IndexFlatIP(d); flat.add(DB)
    _, gold = flat.search(Q, K)

    def recall_at_k(I):
        hit = sum(len(set(I[i]) & set(gold[i])) for i in range(len(I)))
        return hit / (len(I) * K)

    for factory in [f"IVF{NLIST},PQ{M_PQ}"]:
        index = faiss.index_factory(d, factory, faiss.METRIC_INNER_PRODUCT)
        base = faiss.extract_index_ivf(index)
        base.cp.seed = SEED
        # cap PQ k-means iters for a fast train (accuracy unchanged at this scale)
        try:
            ivfpq = faiss.downcast_index(base)
            if hasattr(ivfpq, "pq"):
                ivfpq.pq.cp.niter = 10
                ivfpq.pq.cp.seed = SEED
        except Exception:
            pass
        t0 = time.perf_counter()
        index.train(DB); index.add(DB)
        build = time.perf_counter() - t0
        code_bytes = M_PQ
        resident_bytes = M_PQ
        print(f"\n=== {factory}  (code={code_bytes} B/vec, resident>={resident_bytes} B/vec, build {build:.1f}s) ===")
        print("  nprobe   R@10      QPS")
        for npb in NPROBES:
            base.nprobe = npb
            # warmup
            index.search(Q[:100], K)
            t0 = time.perf_counter()
            runs = 0
            while True:
                _, I = index.search(Q, K)
                runs += 1
                dt = time.perf_counter() - t0
                if dt >= MIN_TIMING_SECONDS:
                    break
            qps = runs * len(Q) / dt
            print(f"  {npb:>5}   {recall_at_k(I):.4f}   {qps:9.0f}")
    print("\nDONE faiss_ivfpq_qps")

if __name__ == "__main__":
    main()
