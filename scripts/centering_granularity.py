#!/usr/bin/env python3
"""How fine should centering be, and what does each granularity cost?

Centering against a single corpus mean is the largest accuracy lever measured anywhere
in this artifact (+8.55 Recall@10 points at two bits, `center-sample-efficiency`). Its
generalization -- k-means centroids, each vector coded as the residual to its own
cluster centre, which is the regime RaBitQ ships in -- is fully implemented behind
`ULTRAVEC_IVF_NLIST`. This script is what measures it.

Two things make the sweep worth reporting rather than just adopting:

  1. It is SHARED. Every codec in the field receives the same centroids, so a gain here
     is a property of the preprocessing and not of the trellis, exactly like the
     rotation-round sweep. The per-codec columns are what make that checkable.

  2. It is NOT free, and the byte column now says so. Centering stores a per-vector
     residual norm, and more than one centroid additionally stores an assignment index
     (`crate::centering_bytes`). Until this campaign neither was charged, so every
     centered number in the paper was quoted at a rate it did not pay. The record grows
     by 4 bytes at `nlist=1` and 5-6 bytes beyond it, against a 32-byte two-bit payload.

The `nlist=0` and `nlist=1` rows are included deliberately. A single global mean is
reported elsewhere as a large GAIN, and a spot check through this path showed it as a
large LOSS; running both here on one footing is what turns that into evidence instead
of a discrepancy between two harnesses.

Run: python3 scripts/centering_granularity.py
"""
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("ULTRAVEC_BIN", ROOT / "target" / "release" / "ultravec"))
BASE = ROOT / "data" / "sift" / "sift_base.fvecs"
QUERY = ROOT / "data" / "sift" / "sift_query.fvecs"
BITS = 2
MAX, QUERY_MAX, SEED = 100_000, 1000, 42
# 0 = uncentered, 1 = single global mean, the rest are k-means centroid counts. 1024 on
# a 100k base leaves ~98 vectors per cluster, which is where estimating a centroid
# starts to fit noise rather than structure -- included so the turn is visible.
GRANULARITIES = [0, 1, 16, 64, 256, 1024]
ROW = re.compile(
    r"^\|\s*([a-z0-9_]+)\s*\|\s*(\d)\s*\|\s*[\d.]+\s*\|\s*([\d.]+)\s*\|"
    r"\s*[\d.]+\s*\|\s*\S+\s*\|\s*[\d.]+\s*\|\s*(\d+)\s*\|"
)


def run(nlist: int) -> dict[str, tuple[float, int]]:
    """(Recall@10, code bytes) per codec for one centering granularity."""
    env = dict(os.environ, ULTRAVEC_TRELLIS_MEM=os.environ.get("ULTRAVEC_TRELLIS_MEM", "12"))
    env.pop("ULTRAVEC_CENTER", None)
    env.pop("ULTRAVEC_IVF_NLIST", None)
    if nlist == 1:
        env["ULTRAVEC_CENTER"] = "1"
    elif nlist > 1:
        env["ULTRAVEC_IVF_NLIST"] = str(nlist)
    completed = subprocess.run(
        [
            str(BIN), "bench",
            "--dataset", str(BASE), "--query-file", str(QUERY),
            "--query-max", str(QUERY_MAX), "--max", str(MAX),
            "--bits", str(BITS), "--metric", "ip", "--seed", str(SEED), "--sota3",
        ],
        capture_output=True, text=True, env=env,
    )
    if completed.returncode:
        raise RuntimeError(f"bench failed (nlist={nlist}):\n{completed.stderr[-2000:]}")
    out: dict[str, tuple[float, int]] = {}
    for line in completed.stdout.splitlines():
        match = ROW.match(line)
        if match and int(match.group(2)) == BITS:
            out[match.group(1)] = (float(match.group(3)), int(match.group(4)))
    if not out:
        raise RuntimeError(f"bench produced no parsable rows (nlist={nlist})")
    return out


def main() -> None:
    print(
        f"# Centering granularity — SIFT-{MAX // 1000}k, {BITS}-bit, {QUERY_MAX} queries, "
        f"raw-IP gold, seed {SEED}, M=12"
    )
    print()
    print(
        "Shared preprocessing: every codec receives the same centroids, so a gain here "
        "belongs to the transform and not to any one codec. Cells are Recall@10 / code "
        "bytes; the byte column includes the per-vector residual norm and centroid index "
        "that centering requires."
    )
    print()
    results = {n: run(n) for n in GRANULARITIES}
    codecs = sorted(set().union(*(set(r) for r in results.values())))
    header = " | ".join(
        "uncentered" if n == 0 else ("global mean" if n == 1 else f"nlist={n}")
        for n in GRANULARITIES
    )
    print(f"| codec | {header} |")
    print("|---|" + "---|" * len(GRANULARITIES))
    for codec in codecs:
        cells = []
        for n in GRANULARITIES:
            got = results[n].get(codec)
            cells.append("---" if got is None else f"{got[0]:.4f} / {got[1]}")
        print(f"| {codec} | " + " | ".join(cells) + " |")
    print()
    # The question the sweep exists to answer: does finer centering change the ORDERING,
    # or does it lift the whole field? Reported as the trellis's margin over the best
    # comparator at each granularity, which is the quantity a reader cares about.
    print("| granularity | trellis | best other | margin pp |")
    print("|---|---|---|---|")
    for n in GRANULARITIES:
        row = results[n]
        if "trellis" not in row:
            continue
        others = {k: v[0] for k, v in row.items() if k != "trellis"}
        if not others:
            continue
        best = max(others.values())
        label = "uncentered" if n == 0 else ("global mean" if n == 1 else f"nlist={n}")
        print(f"| {label} | {row['trellis'][0]:.4f} | {best:.4f} | "
              f"{(row['trellis'][0] - best) * 100:+.2f} |")
    print()
    print("DONE centering_granularity")


if __name__ == "__main__":
    main()
