#!/usr/bin/env python3
"""What does the cheapest possible fit cost, and what does it buy?

Centering lifts UltraVec above the pinned upstream library, but a mean is corpus
state: a centered codec is not calibration-free and must not be reported as though it
were. The question that decides whether the variant is worth having is therefore how
little data buys it. A mean estimated from a handful of vectors is a different
proposition from one that needs the corpus, and it is directly comparable to the
calibration cost this paper already measures for PQ.

The draw is a random variable exactly as PQ's calibration set is, so each finite K is
repeated over the same six seeds the cold-start study uses and reported with a 95% t
interval. K=0 is the plain calibration-free codec and needs no draw; K=full uses the
whole base and is the ceiling. Both are deterministic and reported without an
interval.

Footing is identical to the parity check: normalized SIFT-128, 1,000 queries,
exact-cosine gold, two bits per dimension.

Run: python3 scripts/center_sample_efficiency.py
"""
import math
import os
import statistics
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DD = Path(os.environ.get("ULTRAVEC_ERAB_DIR", ROOT / "data" / "erab"))
BIN = Path(os.environ.get("ULTRAVEC_BIN", ROOT / "target" / "release" / "ultravec"))
TEMP = Path(tempfile.mkdtemp(prefix="ultravec-center-k-"))
K = 10
BITS = int(os.environ.get("ULTRAVEC_CENTER_BITS", "2"))
DRAWS = [int(x) for x in os.environ.get("ULTRAVEC_CENTER_DRAWS", "8,16,32,64,256").split(",")]
SEEDS = [42, 43, 44, 45, 46, 47]
T_CRITICAL = 2.571  # t(0.975, df=5), the six-draw convention used for cold start


def read_ivecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    dim = raw[0]
    return raw.reshape(-1, dim + 1)[:, 1:]


def read_fvecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    dim = raw[0]
    return np.ascontiguousarray(raw.reshape(-1, dim + 1)[:, 1:].view(np.float32))


def recall_at(draw: int, seed: int, queries: np.ndarray, gold: np.ndarray) -> float:
    out = TEMP / f"trellis_k{draw}_s{seed}.fvecs"
    env = dict(os.environ)
    env["ULTRAVEC_TRELLIS_MEM"] = env.get("ULTRAVEC_TRELLIS_MEM", "12")
    env["ULTRAVEC_CENTER"] = "0" if draw == 0 else "1"
    env["ULTRAVEC_CENTER_SEED"] = str(seed)
    if draw > 0:
        env["ULTRAVEC_CENTER_K"] = str(draw)
    else:
        env.pop("ULTRAVEC_CENTER_K", None)
    proc = subprocess.run(
        [str(BIN), "recon", "--backend", "trellis", "--dataset", str(DD / "base.fvecs"),
         "--bits", str(BITS), "--out", str(out)],
        capture_output=True, text=True, env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"recon failed (K={draw}, seed={seed}):\n{proc.stderr[-600:]}")
    database = read_fvecs(out)
    database /= np.linalg.norm(database, axis=1, keepdims=True) + 1e-12
    hits = 0.0
    for start in range(0, len(queries), 256):
        block = queries[start:start + 256]
        got = np.argpartition(-(block @ database.T), K - 1, axis=1)[:, :K]
        for i, row in enumerate(got):
            hits += len(set(row) & set(gold[start + i])) / K
    return hits / len(queries)


def main() -> None:
    gold = read_ivecs(DD / "gold.ivecs")
    queries = read_fvecs(DD / "query.fvecs")
    queries = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12)
    upstream_ids = read_ivecs(DD / "ivfA" / f"erabIVF_n1_b{BITS}.ivecs")
    upstream = float(np.mean(
        [len(set(upstream_ids[i]) & set(gold[i])) / K for i in range(len(gold))]
    ))

    print(f"Centering sample efficiency — normalized SIFT-128, {len(gold)} queries, "
          f"{BITS} bits, exact-cosine gold, {len(SEEDS)} draws")
    print(f"Pinned upstream E-RaBitQ on the same footing: {upstream:.4f}")
    print()
    print("| K | R@10 | CI half | vs upstream pp | vs uncentered pp |")
    print("|---|---|---|---|---|")

    plain = recall_at(0, SEEDS[0], queries, gold)
    print(f"| 0 | {plain:.4f} | 0.0000 | {(plain - upstream) * 100:+.2f} | +0.00 |")

    for draw in DRAWS:
        values = [recall_at(draw, seed, queries, gold) for seed in SEEDS]
        mean = statistics.fmean(values)
        half = T_CRITICAL * statistics.stdev(values) / math.sqrt(len(values))
        print(f"| {draw} | {mean:.4f} | {half:.4f} | {(mean - upstream) * 100:+.2f} "
              f"| {(mean - plain) * 100:+.2f} |")

    ceiling = recall_at(-1, SEEDS[0], queries, gold)
    print(f"| full | {ceiling:.4f} | 0.0000 | {(ceiling - upstream) * 100:+.2f} "
          f"| {(ceiling - plain) * 100:+.2f} |")
    print()
    print("DONE center_sample_efficiency")


if __name__ == "__main__":
    main()
