#!/usr/bin/env python3
"""Cold-start control: is the PQ crossover about calibration scarcity or about 256-way
subquantizers?

The cold-start table fixes PQ at 256 centroids per subspace. Below 256 calibration
vectors that configuration cannot even fill its codebook with distinct centroids, so
"PQ needs roughly 256 vectors" and "a 256-centroid PQ needs roughly 256 vectors" are
not distinguishable from it. This driver holds the emitted rate fixed and varies the
subquantizer width instead: at two bits per dimension, an 8-bit subquantizer means
32 subspaces of 256 centroids, a 4-bit one means 64 subspaces of 16, and a 2-bit one
means 128 subspaces of 4. Every arm spends the same bits per vector; they differ only
in how much there is to estimate.

Reports mean Recall@10 and a 95% t-interval over the same six calibration draws the
main study uses, so the arms are comparable to it and to each other.

Both cold-start corpora are swept. Running SIFT alone would leave the GloVe arm of
the main study resting on the single 256-centroid configuration this control exists
to distrust: on SIFT the low-cardinality arms cross plain UltraVec at K=64 rather
than 256, and there is no reason to assume GloVe behaves differently just because it
was not measured. GloVe's transformed D=32 admits the same three widths, since a
c-bit subquantizer needs c/2-dimensional subspaces dividing D and c a whole multiple
of the two emitted bits per dimension.

Run: python3 scripts/coldstart_pq_width.py
"""
import math
import os
import re
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("ULTRAVEC_BIN", ROOT / "target" / "release" / "ultravec"))
SEEDS = [42, 43, 44, 45, 46, 47]
CALIB = "32,64,128,256,512,1000,2000,5000,20000,0"
KS = ["32", "64", "128", "256", "512", "1000", "2000", "5000", "20000", "full"]
# (name, base, query, evaluation-base size, transformed D). The base sizes and query
# counts are the ones the corresponding main cold-start arm uses -- SIFT-50k from
# `coldstart_seed_ci.py`, GloVe-59.5k from `coldstart_glove_seed_ci.py` -- so a width
# row here is comparable to the study it qualifies rather than to a fresh setup.
CORPORA = [
    ("SIFT", ROOT / "data" / "sift" / "sift_base.fvecs",
     ROOT / "data" / "sift" / "sift_query.fvecs", 50000, 128),
    ("GloVe", ROOT / "data" / "glove" / "glove_base.fvecs",
     ROOT / "data" / "glove" / "glove_query.fvecs", 59500, 32),
]
# Subquantizer bits at 2 emitted bits per dimension. Centroids are 2^c and the
# subspace count is D/(c/2); both are derived per corpus rather than tabulated,
# because hardcoding SIFT's 32/64/128 was what made this driver single-corpus.
WIDTH_BITS = [8, 4, 2]


def widths_for(dim: int) -> list[tuple[int, int, int]]:
    """(subquantizer bits, centroids, subspaces) for every width this D supports."""
    supported = []
    for code_bits in WIDTH_BITS:
        sub_dim = code_bits // 2
        if sub_dim and dim % sub_dim == 0:
            supported.append((code_bits, 1 << code_bits, dim // sub_dim))
    return supported


def run(seed: int, code_bits: int, corpus: tuple) -> tuple[dict[str, float], float]:
    _, base, query, size, _ = corpus
    env = dict(os.environ)
    env["ULTRAVEC_TRELLIS_MEM"] = env.get("ULTRAVEC_TRELLIS_MEM", "12")
    env["ULTRAVEC_PQ_CODE_BITS"] = str(code_bits)
    completed = subprocess.run(
        [
            str(BIN), "coldstart",
            "--dataset", str(base),
            "--query-file", str(query),
            "--query-max", "200", "--max", str(size), "--bits", "2",
            "--calib", CALIB, "--seed", str(seed),
        ],
        capture_output=True, text=True, env=env,
    )
    if completed.returncode:
        raise RuntimeError(
            f"coldstart failed (seed {seed}, code_bits {code_bits}):\n{completed.stderr[-2000:]}"
        )
    pq = {}
    for line in completed.stdout.splitlines():
        match = re.match(r"\|\s*(\d+|full)\s*\|\s*([0-9.]+)\s*\|", line)
        if match:
            pq[match.group(1)] = float(match.group(2))
    trellis = re.search(r"trellis\s+([0-9.]+)", completed.stdout)
    missing = sorted(set(KS) - set(pq))
    if missing or trellis is None:
        raise RuntimeError(
            f"incomplete output (seed {seed}, code_bits {code_bits}); missing={missing}"
        )
    return pq, float(trellis.group(1))


def interval(values: list[float]) -> tuple[float, float]:
    """Mean and 95% t half-width over the calibration draws."""
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, 0.0
    # t(0.975, df=5) = 2.571 -- the same six-draw convention as the main study.
    half = 2.571 * statistics.stdev(values) / math.sqrt(len(values))
    return mean, half


def main() -> None:
    print(f"Cold-start PQ width control — 200 queries, 2 bits/dim, "
          f"{len(SEEDS)} calibration draws")
    print("Every arm emits the same bits per vector; only the factorization differs.")
    print()
    for corpus in CORPORA:
        name, _, _, size, dim = corpus
        # One header per corpus. The width blocks keep their existing `--- ` prefix
        # and ordering, so a consumer that already parses them keeps working; the
        # corpus line is what tells it which block belongs to which base.
        print(f"=== {name}-{size // 1000}k, transformed D={dim}")
        print()
        for code_bits, centroids, subspaces in widths_for(dim):
            per_k: dict[str, list[float]] = {k: [] for k in KS}
            trellis_runs: list[float] = []
            for seed in SEEDS:
                pq, trellis = run(seed, code_bits, corpus)
                trellis_runs.append(trellis)
                for k in KS:
                    per_k[k].append(pq[k])
            trellis_mean = statistics.fmean(trellis_runs)
            print(f"--- {centroids} centroids/subspace, {subspaces} subspaces "
                  f"({code_bits}-bit subquantizer); calibration-free trellis = {trellis_mean:.4f}")
            print("| K | PQ R@10 | 95% CI half | vs trellis pp | leader |")
            print("|---|---|---|---|---|")
            for k in KS:
                mean, half = interval(per_k[k])
                delta = (mean - trellis_mean) * 100.0
                leader = "PQ" if delta > 0 else "UltraVec"
                print(f"| {k} | {mean:.4f} | {half:.4f} | {delta:+.2f} | {leader} |")
            print()
    print("DONE coldstart_pq_width")


if __name__ == "__main__":
    main()
