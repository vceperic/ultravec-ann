#!/usr/bin/env python3
"""Warm batch-fitted PCA-head and plain-trellis reference against Faiss OPQ/PQ.

PQ/OPQ use M=source_dim/2 subquantizers at 8 bits. The PCA head is fitted with
that source-dimension budget, while plain trellis storage is reported from its actual
padded dimension, emission rate, start state, and rescale scalar. These are therefore
nearby-budget reference points rather than an asserted byte match. Reconstruction uses exact
cosine ground truth and trellis memory M=12.
Run: ULTRAVEC_TRELLIS_MEM=12 python3 scripts/dehub_vs_strong_matched.py
"""
import os, re, struct, subprocess, tempfile, numpy as np, faiss
from pathlib import Path

try:
    from scripts.storage_accounting import trellis_serialized_bytes
except ModuleNotFoundError:  # direct `python scripts/...py` invocation
    from storage_accounting import trellis_serialized_bytes

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "target" / "release" / "ultravec"
TEMP = Path(tempfile.mkdtemp(prefix="ultravec-dehub-strong-"))
MEM = os.environ.get("ULTRAVEC_TRELLIS_MEM", "12")
faiss.omp_set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "20")))


def rd(p, maxn=None):
    o = []
    with open(p, "rb") as f:
        while True:
            h = f.read(4)
            if len(h) < 4: break
            d = struct.unpack("<i", h)[0]
            o.append(np.frombuffer(f.read(4 * d), dtype="<f4"))
            if maxn and len(o) >= maxn: break
    return np.ascontiguousarray(np.array(o), dtype=np.float32)


def wr(p, a):
    a = np.ascontiguousarray(a, dtype="<f4")
    with open(p, "wb") as f:
        for v in a:
            f.write(struct.pack("<i", v.shape[0])); f.write(v.tobytes())


def recon(src, bk, bits, out, extra=None, env_extra=None):
    env = dict(os.environ, ULTRAVEC_TRELLIS_MEM=MEM, **(env_extra or {}))
    cmd = [BIN, "recon", "--dataset", src, "--backend", bk, "--bits", str(bits), "--out", out]
    if extra: cmd += extra
    r = subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return rd(out), r.stderr.decode("utf8", "ignore")


def gfid(a, b):
    an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
    bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
    return float(np.nanmean((an * bn).sum(1)))


def topk(Q, D, k):
    dn = np.ascontiguousarray(D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9), dtype=np.float32)
    qn = np.ascontiguousarray(Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9), dtype=np.float32)
    index = faiss.IndexFlatIP(dn.shape[1])
    index.add(dn)
    return index.search(qn, k)[1]


def recalls_at(q, R, gold):
    got = topk(q, R, 100)
    return tuple(
        float(
            np.mean(
                [
                    len(set(got[i, :k].tolist()) & gold[k][i]) / k
                    for i in range(len(q))
                ]
            )
        )
        for k in (1, 10, 100)
    )


def _faiss_fit(db, M, opq, train_n, niter):
    """One PQ/OPQ fit. `train_n=None` passes the whole base; `niter=None` keeps
    Faiss's defaults (25 k-means iterations, 50 OPQ rotations)."""
    d = db.shape[1]; n = len(db)
    if train_n is not None and n > train_n:
        tr = np.ascontiguousarray(db[np.random.RandomState(42).choice(n, train_n, replace=False)])
    else:
        tr = db
    pq = faiss.IndexPQ(d, M, 8)
    if niter is not None:
        pq.pq.cp.niter = niter
    if opq:
        om = faiss.OPQMatrix(d, M)
        if niter is not None:
            om.niter = niter
        index = faiss.IndexPreTransform(om, pq)
    else:
        index = pq
    index.train(tr); index.add(db)
    return index.reconstruct_n(0, n)


# Training budgets for the fitted reference. The first uses Faiss's defaults on
# the whole base; the second uses a 40,000-vector draw and 12 iterations. Their
# ordering varies by corpus: Faiss caps k-means at 256 points per centroid, so
# passing the full base makes it subsample 65,536 points, where a 40,000-vector
# draw is used whole.
FITTED_BUDGETS = ((None, None), (40000, 12))


def faiss_recon(db, M, opq, score=None):
    """Fit the warm reference under each budget and keep the better one.

    The PCA-head residual hybrid fits on the entire base and searches a 28-point (rank, rate)
    grid on held-out data. The fitted reference is therefore evaluated under both
    budgets above, and the stronger result is retained for each corpus.

    `score` maps a reconstruction to a scalar to maximize; without it the first
    budget is used, which keeps the function usable outside the recall harness.
    """
    fits = [_faiss_fit(db, M, opq, train_n, niter) for train_n, niter in FITTED_BUDGETS]
    if score is None:
        return fits[0]
    best = max(fits, key=score)
    return best


DOMAINS = [
    ("GIST-960", ROOT / "data/gist/gist_base.fvecs", ROOT / "data/gist/gist_query.fvecs", 100000, 1000),
    ("SIFT",     ROOT / "data/sift/sift_base.fvecs", ROOT / "data/sift/sift_query.fvecs", 100000, 1000),
    ("dbpedia",  ROOT / "data/dbpedia/dbpedia_base.fvecs", ROOT / "data/dbpedia/dbpedia_query.fvecs", 100000, 1000),
]
DB = TEMP / "db.fvecs"
print(f"Warm reference: batch-fitted PCA-head, plain and centered trellis vs Faiss "
      f"OPQ/PQ | trellis M={MEM}")
print("Shared fitted state is outside the byte column for every fitted arm: PQ/OPQ "
      "codebooks, the PCA-head projection, and the centered arm's single mean vector "
      f"(4*dim bytes).\n")
for nm, dbp, qp, mx, qmx in DOMAINS:
    if not (os.path.exists(dbp) and os.path.exists(qp)):
        raise FileNotFoundError(f"missing required {nm} inputs: {dbp} or {qp}")
    db = rd(dbp, mx); q = rd(qp, qmx); dim = q.shape[1]
    db = db[:, :dim] if db.shape[1] != dim else db
    gold_ids = topk(q, db, 100)
    g1 = [set(r[:1].tolist()) for r in gold_ids]
    g10 = [set(r[:10].tolist()) for r in gold_ids]
    g100 = [set(r.tolist()) for r in gold_ids]
    wr(DB, db)
    B = dim // 2
    trellis_bytes = trellis_serialized_bytes(dim, 4, int(MEM))
    dh, derr = recon(DB, "dehub", 4, TEMP / "dehub.fvecs", extra=["--budget-bytes", str(B)]); dh = dh[:, :dim]
    tr4, _ = recon(DB, "trellis", 4, TEMP / "trellis.fvecs"); tr4 = tr4[:, :dim]
    # The warm regime is the one where fitting is allowed, and every other column in
    # this table fits something: PQ and OPQ train codebooks, the PCA head fits a
    # projection. Leaving UltraVec calibration-free here compares a codec that learns
    # nothing against three that do. The centered arm is the cheapest thing it can
    # learn -- one corpus mean -- and it is charged the same way the others are: the
    # per-vector record is unchanged, and the shared state sits outside the byte
    # column exactly as PQ's codebook does.
    tr4c, _ = recon(DB, "trellis", 4, TEMP / "trellis_centered.fvecs",
                    env_extra={"ULTRAVEC_CENTER": "1"}); tr4c = tr4c[:, :dim]
    m = re.search(r"r=(\d+) b=(\d+) \((\d+)B used\)", derr); cfg = f"r={m.group(1)},b={m.group(2)},{m.group(3)}B" if m else "?"
    pick10 = lambda R: recalls_at(q, R, {1: g1, 10: g10, 100: g100})[1]
    pqf = faiss_recon(db, B, opq=False, score=pick10)
    rows = []
    if B <= 160:  # OPQ rotation only tractable+meaningful at coarse M; at sub_dim=2 (large M) OPQ~PQ
        rows.append(("faiss-OPQ (data-dep, strong)", B, faiss_recon(db, B, opq=True, score=pick10)))
    else:
        rows.append((f"faiss-OPQ SKIP (M={B}: OPQ~PQ at sub_dim 2)", B, None))
    dehub_bytes = int(m.group(3)) if m else B
    rows += [("faiss-PQ  (data-dep)", B, pqf),
             ("trellis@4 (obl, no head)", trellis_bytes, tr4),
             ("trellis@4 centered (fitted mean)", trellis_bytes, tr4c),
             (f"PCA-head@budget (data-dep; {cfg})", dehub_bytes, dh)]
    gold = {1: g1, 10: g10, 100: g100}
    recall_cache = {id(pqf): recalls_at(q, pqf, gold)}
    ref10 = recall_cache[id(pqf)][1]  # always compare vs PQ (present on every corpus)
    print(f"=== {nm}  (n={len(db)}, dim={dim}, queries={len(q)}, budget={B}B) ===")
    print(f"  {'method':34s} {'bytes':>6s} {'g':>7s} {'R@1':>7s} {'R@10':>7s} {'R@100':>7s}")
    for name, by, R in rows:
        if R is None:
            print(f"  {name:34s}"); continue
        if id(R) not in recall_cache:
            recall_cache[id(R)] = recalls_at(q, R, gold)
        r1, r10, r100 = recall_cache[id(R)]
        tag = f"  (vs PQ: {(r10-ref10)*100:+.1f}pp R@10)" if not name.startswith("faiss-PQ") else ""
        print(f"  {name:34s} {by:6d} {gfid(db,R):7.4f} {r1:7.4f} {r10:7.4f} {r100:7.4f}{tag}", flush=True)
    print(flush=True)
print("DONE")
