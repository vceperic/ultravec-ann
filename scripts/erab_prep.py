#!/usr/bin/env python3
"""Prepare normalized SIFT, one-cell IVF assignment, and exact-cosine gold."""
import numpy as np
import os
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "sift"
OUT = Path(os.environ.get("ULTRAVEC_ERAB_DIR", ROOT / "data" / "erab"))
N = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
Q = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
K = 10
os.makedirs(OUT, exist_ok=True)


def rd(p, limit):
    path = Path(p)
    with path.open("rb") as handle:
        raw_dim = handle.read(4)
    if len(raw_dim) != 4:
        raise ValueError(f"empty vecs file: {path}")
    dim = struct.unpack("<i", raw_dim)[0]
    if dim <= 0:
        raise ValueError(f"invalid vector dimension {dim}: {path}")
    count = min(limit, path.stat().st_size // (4 * (dim + 1)))
    raw = np.fromfile(path, dtype="<i4", count=count * (dim + 1))
    rows = raw.reshape(count, dim + 1)
    if not np.all(rows[:, 0] == dim):
        raise ValueError(f"inconsistent vecs row dimensions: {path}")
    return np.ascontiguousarray(rows[:, 1:].view("<f4"))


def wf(p, arr):
    with open(p, "wb") as f:
        for v in arr:
            f.write(struct.pack("<i", len(v)))
            f.write(v.astype("<f4").tobytes())


base = rd(SRC / "sift_base.fvecs", N)
query = rd(SRC / "sift_query.fvecs", Q)
base /= np.linalg.norm(base, axis=1, keepdims=True) + 1e-12
query /= np.linalg.norm(query, axis=1, keepdims=True) + 1e-12
gold = np.zeros((len(query), K), dtype=np.int32)
for i in range(0, len(query), 100):
    qb = query[i:i + 100]
    score = qb @ base.T
    candidates = np.argpartition(-score, K - 1, axis=1)[:, :K]
    candidate_scores = np.take_along_axis(score, candidates, axis=1)
    order = np.argsort(-candidate_scores, axis=1)
    gold[i:i + 100] = np.take_along_axis(candidates, order, axis=1)
wf(OUT / "base.fvecs", base)
wf(OUT / "query.fvecs", query)
with (OUT / "gold.ivecs").open("wb") as f:
    for r in gold:
        f.write(struct.pack("<i", K))
        f.write(r.astype("<i4").tobytes())

ivf = OUT / "ivfA"
ivf.mkdir(exist_ok=True)
wf(ivf / "centroids_1.fvecs", base.mean(axis=0, keepdims=True))
with (ivf / "cids_1.ivecs").open("wb") as handle:
    row = struct.pack("<ii", 1, 0)
    for _ in range(len(base)):
        handle.write(row)
print(f"prep: normalized base {base.shape} query {query.shape} cosine gold top-{K}, one IVF cell -> {OUT}")
