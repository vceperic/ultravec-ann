#!/usr/bin/env python3
"""Parity check: artifact-native E-RaBitQ against the pinned upstream library.

The flat comparison table reports an artifact-native implementation of the published
E-RaBitQ construction. A reviewer cannot tell from the manuscript whether that
implementation is competitive with the real library, and the paper's other upstream
experiment changes metric, scorer and search path at the same time, so it cannot
settle the question either. This driver removes every difference except the
implementation.

Everything is shared: the same normalized SIFT-128 base (100k) and queries (1,000)
under `data/erab/`, the same exact-cosine `gold.ivecs`, the same emitted rate, and
the same Recall@10 definition. The upstream side is the retained
`ivfA/erabIVF_n1_b{bits}.ivecs` top-10 output -- `n1` is a single-list IVF, so it is
an exhaustive scan with the library's native asymmetric estimator, not an
approximate index. The artifact side reconstructs with `recon --backend rabitq` and
scores exact cosine over the normalized reconstructions. The only remaining variable
is whose code produced the codes.

UltraVec is reported on the identical footing so that the headline comparison can be
read against the upstream baseline rather than only against the in-repo one.

Run: python3 scripts/erab_parity.py
"""
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DD = Path(os.environ.get("ULTRAVEC_ERAB_DIR", ROOT / "data" / "erab"))
BIN = Path(os.environ.get("ULTRAVEC_BIN", ROOT / "target" / "release" / "ultravec"))
TEMP = Path(tempfile.mkdtemp(prefix="ultravec-erab-parity-"))
BOOTSTRAP = 2000
K = 10
RATES = (1, 2, 3, 4)
SEED = 42


def read_ivecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    dim = raw[0]
    return raw.reshape(-1, dim + 1)[:, 1:]


def read_fvecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    dim = raw[0]
    return np.ascontiguousarray(raw.reshape(-1, dim + 1)[:, 1:].view(np.float32))


def reconstruct(backend: str, bits: int, centered: bool = False) -> np.ndarray:
    """Database reconstruction from the artifact's own codec at `bits`."""
    out = TEMP / f"{backend}_b{bits}{'_centered' if centered else ''}.fvecs"
    env = dict(os.environ)
    env.setdefault("ULTRAVEC_TRELLIS_MEM", "12")
    env["ULTRAVEC_CENTER"] = "1" if centered else "0"
    proc = subprocess.run(
        [
            str(BIN), "recon",
            "--backend", backend,
            "--dataset", str(DD / "base.fvecs"),
            "--bits", str(bits),
            "--out", str(out),
        ],
        capture_output=True, text=True, env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"recon --backend {backend} --bits {bits} failed:\n{proc.stderr[-800:]}")
    return read_fvecs(out)


def top10(reconstruction: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Exact-cosine top-10 over normalized reconstructions -- the shared scorer."""
    database = reconstruction / (np.linalg.norm(reconstruction, axis=1, keepdims=True) + 1e-12)
    got = np.zeros((len(queries), K), dtype=np.int64)
    for start in range(0, len(queries), 256):
        block = queries[start:start + 256]
        score = block @ database.T
        got[start:start + 256] = np.argpartition(-score, K - 1, axis=1)[:, :K]
    return got


def per_query_recall(got: np.ndarray, gold: np.ndarray) -> np.ndarray:
    return np.array([len(set(got[i]) & set(gold[i])) / K for i in range(len(got))])


def main() -> None:
    gold = read_ivecs(DD / "gold.ivecs")
    queries = read_fvecs(DD / "query.fvecs")
    queries = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12)
    if len(queries) != len(gold):
        raise ValueError(f"query/gold mismatch: {len(queries)} vs {len(gold)}")

    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(gold), size=(BOOTSTRAP, len(gold)))

    print(f"E-RaBitQ implementation parity — normalized SIFT-128, "
          f"{len(gold)} queries, exact-cosine gold, seed {SEED}")
    print("All rows share inputs, queries, gold and rate; only the implementation differs.")
    print()
    def interval(delta: np.ndarray) -> str:
        resampled = np.sort(delta[draws].mean(axis=1))
        return f"[{resampled[int(0.025 * BOOTSTRAP)]:+.2f}, {resampled[int(0.975 * BOOTSTRAP)]:+.2f}]"

    print("| bits | upstream | artifact | vs up. | 95% CI | art.+center | vs up. "
          "| 95% CI | UltraVec | vs up. | UV+center | vs up. | 95% CI |")
    print("|" + "---|" * 13)
    for bits in RATES:
        upstream = per_query_recall(read_ivecs(DD / "ivfA" / f"erabIVF_n1_b{bits}.ivecs"), gold)
        plain = per_query_recall(top10(reconstruct("rabitq", bits), queries), gold)
        centered = per_query_recall(top10(reconstruct("rabitq", bits, centered=True), queries), gold)
        trellis = per_query_recall(top10(reconstruct("trellis", bits), queries), gold)
        trellis_c = per_query_recall(top10(reconstruct("trellis", bits, centered=True), queries), gold)

        dp = (plain - upstream) * 100.0
        dc = (centered - upstream) * 100.0
        dt = (trellis - upstream) * 100.0
        dtc = (trellis_c - upstream) * 100.0
        print(
            f"| {bits} | {upstream.mean():.4f} | {plain.mean():.4f} | {dp.mean():+.2f} "
            f"| {interval(dp)} | {centered.mean():.4f} | {dc.mean():+.2f} "
            f"| {interval(dc)} | {trellis.mean():.4f} | {dt.mean():+.2f} "
            f"| {trellis_c.mean():.4f} | {dtc.mean():+.2f} | {interval(dtc)} |"
        )
    print()
    print("DONE erab_parity")


if __name__ == "__main__":
    main()
