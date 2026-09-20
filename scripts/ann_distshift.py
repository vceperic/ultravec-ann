#!/usr/bin/env python3
"""X2 -- ANN distribution-shift robustness: a learned PQ codebook trained on a
DIFFERENT region of the corpus (off-distribution) vs the same PQ trained in-corpus
vs the codebook-free oblivious trellis (no fitted corpus state).

Construction of a genuine shift (no synthetic data; real SIFT1M descriptors):
  1. k-means(2) on the SIFT base -> two clusters A, B (disjoint regions of the
     descriptor manifold). This is a real covariate shift: the two clusters have
     different per-dimension means/spreads (reported).
  2. Corpus B (the deployed/shifted index)  = base[label==B].
     Training pool A (off-distribution)      = base[label==A].
  3. Queries = a held-out sample drawn from cluster B (so the query distribution is
     B, matched to the DB). Normalize A, DB_B, and queries for directional retrieval.
     Gold = exact-cosine top-10 within DB_B, recomputed for this database.

Three codecs, ONE (DB_B, queries_B, gold_B), all scored by exact cosine over the
reconstruction / PQ-decoded DB -- a matched comparison in which only the codec
differs:
  - transferred-PQ : faiss PQ trained on pool A, encodes+searches DB_B   (the shift)
  - in-corpus-PQ   : faiss PQ trained on DB_B,  encodes+searches DB_B    (PQ's best case)
  - trellis        : ultravec recon --backend trellis --bits 2 on normalized DB_B,
                     then cosine-rank with the original fp32 normalized queries_B;
                     data-free and without fitted corpus state.

PQ is run at M=16 x 8-bit = 16 B/vector and M=32 x 8-bit = 32 B/vector. The
trellis has a 32-byte two-bit payload, plus its start state and rescale scalar for a
38-byte serialized record. These points diagnose transfer loss; they are not claimed
as an exact byte match. Recall@10 is reported for every leg.

The oblivious trellis keeps identical codec parameters across the shift. The script
reports the measured PQ degradation without assuming its direction or size.

Usage: ann_distshift.py [--base data/sift/sift_base.fvecs]
                        [--mem 10] [--bits 2] [--seed 42]
"""
import argparse, os, struct, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

try:
    from scripts.storage_accounting import trellis_serialized_bytes
except ModuleNotFoundError:  # direct `python scripts/...py` invocation
    from storage_accounting import trellis_serialized_bytes

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "target" / "release" / "ultravec"
K = 10


# ---------- fvecs IO (LE float32, dim-prefixed) ----------
def rd(p, maxn=None):
    o = []
    with open(p, "rb") as f:
        while True:
            h = f.read(4)
            if len(h) < 4:
                break
            d = struct.unpack("<i", h)[0]
            o.append(np.frombuffer(f.read(4 * d), dtype="<f4"))
            if maxn and len(o) >= maxn:
                break
    return np.ascontiguousarray(np.array(o), dtype=np.float32)


def wr(p, arr):
    a = np.ascontiguousarray(arr, dtype="<f4")
    with open(p, "wb") as f:
        for v in a:
            f.write(struct.pack("<i", v.shape[0]))
            f.write(v.tobytes())


def recon(src, backend, bits, out, mem):
    """Use the UltraVec CLI to reconstruct unit directions with the given codec."""
    env = dict(os.environ, ULTRAVEC_TRELLIS_MEM=str(mem))
    p = subprocess.run([BIN, "recon", "--dataset", src, "--backend", backend,
                        "--bits", str(bits), "--out", out],
                       capture_output=True, text=True, env=env)
    if p.returncode != 0:
        sys.exit(f"recon failed ({backend} {bits}b on {src}):\n{p.stderr}\n{p.stdout}")
    return out


# ---------- exact-cosine recall@10 (the matched metric for every leg) ----------
def normalize(a):
    return np.ascontiguousarray(a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12),
                                dtype=np.float32)


def cosine_topk(db, q, k):
    """Indices of the exact top-k highest-cosine db rows for each query (blocked)."""
    out = []
    for i in range(0, len(q), 256):
        qb = q[i:i + 256]
        score = qb @ db.T
        idx = np.argpartition(-score, k - 1, axis=1)[:, :k]
        out.append(idx)
    return np.concatenate(out, 0)


def recall_at_k(got, gold_sets, k):
    return float(np.mean([len(set(got[i].tolist()) & gold_sets[i]) / k
                          for i in range(len(got))]))


def trellis_recall(db_path, queries, bits, mem, gold_sets, tag, temp_dir):
    """Reconstruct the DB and cosine-rank from full-precision normalized queries."""
    dbh = recon(db_path, "trellis", bits, temp_dir / f"tr_db_{tag}.fvecs", mem)
    DB = normalize(rd(dbh))
    got = cosine_topk(DB, queries, K)
    return recall_at_k(got, gold_sets, K), trellis_serialized_bytes(DB.shape[1], bits, mem)


def pq_recall(train, db, q, m, nbits, gold_sets, tag, temp_dir):
    """Train Faiss PQ, then cosine-rank its normalized decoded database.
    Isolated subprocess so a faiss assert can't take the grid down."""
    train_path = temp_dir / f"train_{tag}.fvecs"
    db_path = temp_dir / f"db_{tag}.fvecs"
    query_path = temp_dir / f"query_{tag}.fvecs"
    gold_path = temp_dir / f"gold_{tag}.ivecs"
    wr(train_path, train)
    wr(db_path, db)
    wr(query_path, q)
    import json
    cmd = [sys.executable, str(ROOT / "scripts" / "ann_distshift_pq.py"),
           str(train_path), str(db_path), str(query_path), str(m), str(nbits), str(K)]
    # pass gold inline via a temp ivecs
    gold_arr = np.array([sorted(s) for s in gold_sets], dtype=np.int32)
    with gold_path.open("wb") as f:
        for row in gold_arr:
            f.write(struct.pack("<i", len(row)))
            f.write(row.tobytes())
    cmd.append(str(gold_path))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"faiss PQ worker failed:\n{p.stderr}\n{p.stdout}")
    line = [l for l in p.stdout.splitlines() if l.startswith("{")][-1]
    r = json.loads(line)
    return r["r10"], r["bytes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "data" / "sift" / "sift_base.fvecs"))
    # Default from the environment so a tier's operating point is honored; an
    # explicit --mem still takes precedence.
    ap.add_argument("--mem", type=int,
                    default=int(os.environ.get("ULTRAVEC_TRELLIS_MEM", 12)))
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nq", type=int, default=1000, help="held-out queries drawn from cluster B")
    a = ap.parse_args()

    from sklearn.cluster import MiniBatchKMeans

    base = rd(a.base)
    print(f"loaded base {base.shape} from {a.base}", flush=True)

    # 1. k-means(2): split the descriptor manifold into two real regions.
    km = MiniBatchKMeans(n_clusters=2, random_state=a.seed, n_init=10, batch_size=4096)
    lab = km.fit_predict(base)
    nA, nB = int((lab == 0).sum()), int((lab == 1).sum())
    # Make A = the SMALLER cluster (the off-distribution training pool), B = larger (the
    # deployed index). Either assignment is a valid shift; fix it deterministically.
    if nA > nB:
        lab = 1 - lab
        nA, nB = nB, nA
    A = base[lab == 0]
    B = base[lab == 1]

    # Quantify the shift: per-dim mean separation between A and B, normalized by B's std.
    muA, muB, sdB = A.mean(0), B.mean(0), B.std(0) + 1e-9
    shift = float(np.mean(np.abs(muA - muB) / sdB))
    # centroid cosine (how far apart the two regions point)
    cca = float(np.dot(muA, muB) / (np.linalg.norm(muA) * np.linalg.norm(muB) + 1e-9))
    print(f"cluster A (train pool) n={nA}  cluster B (deployed index) n={nB}", flush=True)
    print(f"shift: mean |mu_A-mu_B|/std_B = {shift:.3f}  centroid-cosine(A,B) = {cca:.4f}",
          flush=True)

    # 2. queries from cluster B (held out from the DB so a query is never its own NN).
    rng = np.random.default_rng(a.seed)
    nq = min(a.nq, nB // 4)
    qidx = rng.choice(nB, size=nq, replace=False)
    mask = np.ones(nB, dtype=bool); mask[qidx] = False
    DB_B = np.ascontiguousarray(B[mask])
    Q_B = np.ascontiguousarray(B[qidx])
    print(f"DB_B {DB_B.shape}  Q_B {Q_B.shape} (queries drawn from cluster B)", flush=True)

    # 3. Directional retrieval protocol: normalize all codec inputs and exact gold.
    A_codec, DB_codec, Q_codec = normalize(A), normalize(DB_B), normalize(Q_B)
    gold = cosine_topk(DB_codec, Q_codec, K)
    gold_sets = [set(row.tolist()) for row in gold]

    temporary = tempfile.TemporaryDirectory(prefix="ultravec-distshift-")
    temp_dir = Path(temporary.name)
    db_b_path = temp_dir / "db_b.fvecs"
    wr(db_b_path, DB_codec)

    print("\n--- cosine recall@10 under distribution shift (DB=cluster B, queries=cluster B) ---",
          flush=True)
    results = {}

    # Trellis: reconstruct only DB_B; queries remain full precision, as in the PQ legs.
    tr_r10, tr_bytes = trellis_recall(db_b_path, Q_codec, a.bits, a.mem,
                                      gold_sets, "B", temp_dir)
    results["trellis_2bit"] = (tr_r10, tr_bytes)
    print(f"trellis @{a.bits}-bit (oblivious)      : R@10 = {tr_r10:.4f}  ({tr_bytes:.0f} B/vec)",
          flush=True)

    # PQ legs at 16 B and 32 B; the trellis directional record is 38 B.
    for (m, nb, label) in [(16, 8, "M16x8 (16 B/vec)"), (32, 8, "M32x8 (32 B/vec)")]:
        inc_r10, inc_bytes = pq_recall(DB_codec, DB_codec, Q_codec, m, nb, gold_sets, f"inc_{m}", temp_dir)
        tr_r10b, tr_bytes2 = pq_recall(A_codec, DB_codec, Q_codec, m, nb, gold_sets, f"trf_{m}", temp_dir)
        results[f"pq_incorpus_{label}"] = (inc_r10, inc_bytes)
        results[f"pq_transferred_{label}"] = (tr_r10b, tr_bytes2)
        drop = inc_r10 - tr_r10b
        print(f"PQ {label}:", flush=True)
        print(f"    in-corpus   (train on B)        : R@10 = {inc_r10:.4f}  ({inc_bytes:.0f} B/vec)",
              flush=True)
        print(f"    transferred (train on A, off-dist): R@10 = {tr_r10b:.4f}  ({tr_bytes2:.0f} B/vec)",
              flush=True)
        print(f"    => transfer degradation          : {drop:+.4f}  ({100*drop/max(inc_r10,1e-9):+.1f}% rel)",
              flush=True)

    # summary line
    print("\n=== SUMMARY ===", flush=True)
    print(f"shift_strength(mean|dmu|/stdB)={shift:.3f}  centroid_cos={cca:.4f}  "
          f"nA={nA} nB={nB} nq={nq} seed={a.seed} mem={a.mem} bits={a.bits}", flush=True)
    for k, (v, b) in results.items():
        print(f"  {k:42s} R@10={v:.4f}  ({b:.0f} B/vec)", flush=True)


if __name__ == "__main__":
    main()
