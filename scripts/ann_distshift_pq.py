#!/usr/bin/env python3
"""Isolated Faiss PQ worker for the ANN distribution-shift experiment.

Trains a plain PQ on normalized vectors, decodes the database, normalizes those
reconstructions, and scores them by cosine against full-precision normalized queries
and exact-cosine gold. Subprocess isolation ensures a Faiss
assert/segfault is reported as a failed leg, not a grid kill.

argv: train.fvecs db.fvecs query.fvecs M nbits K gold.ivecs
"""
import sys, struct, json
import numpy as np
import faiss

faiss.omp_set_num_threads(4)


def rd(p):
    o = []
    with open(p, "rb") as f:
        while True:
            h = f.read(4)
            if len(h) < 4:
                break
            d = struct.unpack("<i", h)[0]
            o.append(np.frombuffer(f.read(4 * d), dtype="<f4"))
    return np.ascontiguousarray(np.array(o, dtype=np.float32))


def rdi(p):
    o = []
    with open(p, "rb") as f:
        while True:
            h = f.read(4)
            if len(h) < 4:
                break
            d = struct.unpack("<i", h)[0]
            o.append(np.frombuffer(f.read(4 * d), dtype="<i4").tolist())
    return o


def main():
    train_p, db_p, q_p, M, nbits, K, gold_p = sys.argv[1:8]
    M, nbits, K = int(M), int(nbits), int(K)
    train = rd(train_p); DB = rd(db_p); Q = rd(q_p); gold = rdi(gold_p)
    d = DB.shape[1]
    # Plain PQ training followed by flat cosine scoring of its reconstruction. No
    # IVF: recall is the codec's, not confounded by coarse-quantizer pruning.
    index = faiss.IndexPQ(d, M, nbits)
    index.pq.cp.niter = 25            # full k-means (small corpus; seconds)
    index.train(train)
    index.add(DB)
    decoded = np.ascontiguousarray(index.reconstruct_n(0, len(DB)), dtype=np.float32)
    faiss.normalize_L2(decoded)
    faiss.normalize_L2(Q)
    flat = faiss.IndexFlatIP(d)
    flat.add(decoded)
    _, I = flat.search(Q, K)
    r10 = []
    for qi in range(len(Q)):
        g = set(gold[qi][:K])
        if not g:
            continue
        ranked = [int(x) for x in I[qi] if x >= 0][:K]
        r10.append(len(g & set(ranked)) / K)
    bpv = float(index.sa_code_size())
    print(json.dumps({"r10": float(np.mean(r10)), "bytes": bpv, "nq": len(r10)}))


if __name__ == "__main__":
    main()
