#!/usr/bin/env python3
"""Bootstrap CI for the PCA-head hybrid vs E-RaBitQ-estimator recall@10 gap
on normalized SIFT-128 (100k base / 1000 queries). E-RaBitQ's per-query top-10 IDs
are precomputed (ivfA/erabIVF_n1_b2.ivecs, its real asymmetric estimator); the PCA-head hybrid's top-10 are
computed fresh from the rust `recon --backend dehub` (fixed r=8, matching the paper's A/B).
Both compress only the database and use the original full-precision normalized
queries against exact-cosine gold. Bootstrap-resample the queries to obtain a
95% confidence interval on the freshly measured gap.

Run: python3 scripts/erab_bootstrap_ci.py
"""
import os, struct, subprocess, tempfile, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DD = Path(os.environ.get("ULTRAVEC_ERAB_DIR", ROOT / "data" / "erab"))
BIN = ROOT / "target" / "release" / "ultravec"
TEMP = Path(tempfile.mkdtemp(prefix="ultravec-erab-ci-"))
B = 2000   # bootstrap resamples
K = 10

def rdi(p):
    a = np.fromfile(p, dtype=np.int32); d = a[0]
    return a.reshape(-1, d + 1)[:, 1:]

def rdf(p):
    a = np.fromfile(p, dtype=np.int32); d = a[0]
    return np.ascontiguousarray(a.reshape(-1, d + 1)[:, 1:].view(np.float32))

def recon_dehub(src, out, bits):
    env = dict(os.environ, ULTRAVEC_TRELLIS_MEM=os.environ.get("ULTRAVEC_TRELLIS_MEM", "12"), ULTRAVEC_DEHUB_R="8")
    p = subprocess.run([BIN, "recon", "--backend", "dehub", "--fit", f"{DD}/base.fvecs",
                        "--dataset", src, "--bits", str(bits), "--out", out],
                       capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-500:])
    return rdf(out)

gold = rdi(f"{DD}/gold.ivecs")                 # exact-cosine top-10
nq = len(gold)
def per_query_recall(got):
    return np.array([len(set(got[i]) & set(gold[i])) / K for i in range(len(got))])

def recon_dehub_bits(bits):
    DBn = recon_dehub(f"{DD}/base.fvecs", TEMP / "dehub_db.fvecs", bits)
    Q = rdf(f"{DD}/query.fvecs")
    DBn /= np.linalg.norm(DBn, axis=1, keepdims=True) + 1e-12
    Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12
    got = np.zeros((nq, K), dtype=np.int64)
    for i in range(0, nq, 256):
        query = Q[i:i+256]
        score = query @ DBn.T
        got[i:i+256] = np.argpartition(-score, K - 1, axis=1)[:, :K]
    return per_query_recall(got)

print("bits | PCA-head | E-RaBitQ | gap pp | 95% CI (bootstrap)")
for bits in [1, 2, 3, 4]:
    erab = rdi(f"{DD}/ivfA/erabIVF_n1_b{bits}.ivecs")
    erab_pq = per_query_recall(erab)
    dehub_pq = recon_dehub_bits(bits)
    dm, em = dehub_pq.mean(), erab_pq.mean(); gap = (dm - em) * 100
    rng = np.random.default_rng(42)
    gaps = np.array([(dehub_pq[idx].mean() - erab_pq[idx].mean()) * 100
                     for idx in (rng.integers(0, nq, nq) for _ in range(B))])
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    print(f"  {bits}  | {dm:.4f} | {em:.4f}   | {gap:+.2f}  | [{lo:+.1f}, {hi:+.1f}]", flush=True)
print("DONE erab_bootstrap_ci")
