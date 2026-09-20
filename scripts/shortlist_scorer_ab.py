#!/usr/bin/env python3
"""Which stage-1 scorer should the shortlist arm use?

The two-stage companion prunes each posting list with a 1-bit sign code before the
trellis reranks the survivors. Two choices in that prune were implemented but not yet
measured, so the constructor defaults were deciding them. This A/B measures both, so
the reported arm is a measured choice rather than an inherited default:

  1. ASYMMETRIC vs SYMMETRIC. The asymmetric scorer keeps the query in full precision
     and computes `<q_r, sign(u_r)>`; the symmetric one binarizes the query too and
     ranks by Hamming distance, discarding every query magnitude. Symmetric was the
     default.
  2. RESIDUAL-NORM WEIGHT. Under centering a stored vector is `u_i = c + ||u_i-c||*r_i`
     and only `r_i` is sign-coded, so the quantity that ranks candidates carries a
     per-vector factor `||u_i-c||` that stage 1 ignored. Uncentered the factor is 1.0
     for every entry, so this is a centering-only question -- and the systems tier
     centers.

The list geometry is what makes this measurable, so the sweep uses `nlist` chosen to
match the production mean list length (1,000,000/1024 = 977 there, 100,000/100 = 1,000
here) rather than the sqrt-N default. A single loose cluster is not a model of an IVF
posting list: with vectors spread over the sphere the centroid collapses toward the
origin, every residual norm is ~1, and the residual-norm question becomes vacuous.

Reports Recall@10 and QPS per (scorer, weight, C), so the two levers can be read
independently and neither is presented without its cost.

Run: python3 scripts/shortlist_scorer_ab.py
"""
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("ULTRAVEC_BIN", ROOT / "target" / "release" / "ultravec"))
BASE = ROOT / "data" / "sift" / "sift_base.fvecs"
QUERY = ROOT / "data" / "sift" / "sift_query.fvecs"
SHORTLISTS = [128, 256, 512]
NPROBE = 8
# 100k/100 = 1,000 vectors per list, against 977 in the reported 1M/1024 configuration.
MAX, NLIST, QUERY_MAX, BITS, SEED = 100_000, 100, 200, 2, 42
ROW = re.compile(
    r"^\|\s*(\S+)\s*\|[^|]*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*(\d+)\s*\|"
)


def run(scorer: str, rnorm: str) -> dict[int, tuple[float, int]]:
    """(Recall@10, QPS) per shortlist size for one (scorer, weight) combination.

    Throughput is reported because the two scorers are not a strict ordering: the
    asymmetric one does O(D) float work per entry where the symmetric one does a
    popcount, so it buys shortlist quality with scan speed. Reporting only recall at a
    fixed C would present one point of a trade-off as a verdict -- and C is a free
    parameter, so the honest comparison is the frontier the two trace together.
    """
    env = dict(
        os.environ,
        ULTRAVEC_TRELLIS_MEM=os.environ.get("ULTRAVEC_TRELLIS_MEM", "12"),
        # The reported arm is centered, and the residual-norm weight only exists under
        # centering, so measuring uncentered would answer a question nobody asked.
        ULTRAVEC_IVF_CENTER="1",
    )
    completed = subprocess.run(
        [
            str(BIN), "ivf",
            "--dataset", str(BASE), "--query-file", str(QUERY),
            "--max", str(MAX), "--query-max", str(QUERY_MAX), "--bits", str(BITS),
            "--nlist", str(NLIST), "--nprobe", str(NPROBE),
            "--shortlist", ",".join(str(c) for c in SHORTLISTS),
            "--codecs", "trellis_shortlist",
            "--shortlist-scorer", scorer, "--shortlist-rnorm", rnorm,
            "--reps", "1", "--seed", str(SEED),
        ],
        capture_output=True, text=True, env=env,
    )
    if completed.returncode:
        raise RuntimeError(
            f"ivf failed (scorer {scorer}, rnorm {rnorm}):\n{completed.stderr[-2000:]}"
        )
    out: dict[int, tuple[float, int]] = {}
    for line in completed.stdout.splitlines():
        match = ROW.match(line)
        if match and int(match.group(2)) == NPROBE:
            size = int(match.group(1).rsplit("_c", 1)[1])
            out[size] = (float(match.group(3)), int(match.group(4)))
    missing = sorted(set(SHORTLISTS) - set(out))
    if missing:
        raise RuntimeError(
            f"incomplete output (scorer {scorer}, rnorm {rnorm}); missing C={missing}"
        )
    return out


def main() -> None:
    print(
        f"# Shortlist stage-1 scorer — SIFT-{MAX // 1000}k, {BITS}-bit, nlist {NLIST} "
        f"(mean list {MAX // NLIST}), centered, n_probe={NPROBE}, {QUERY_MAX} queries, seed {SEED}"
    )
    print()
    print("Recall@10 after the trellis rerank, so both stages are included.")
    print()
    print("Cells are Recall@10 / QPS.")
    print()
    print("| scorer | residual-norm weight | " + " | ".join(f"C={c}" for c in SHORTLISTS) + " |")
    print("|---|---|" + "---|" * len(SHORTLISTS))
    results: dict[tuple[str, str], dict[int, tuple[float, int]]] = {}
    for scorer in ("asym", "hamming"):
        for rnorm in ("1", "0"):
            got = run(scorer, rnorm)
            results[(scorer, rnorm)] = got
            label = "on" if rnorm == "1" else "off"
            print(
                f"| {scorer} | {label} | "
                + " | ".join(f"{got[c][0]:.4f} / {got[c][1]}" for c in SHORTLISTS)
                + " |"
            )
    print()
    # The two levers, isolated. Reported as explicit deltas because the paper states
    # them as separate claims and a reader should not have to subtract table cells.
    print("| lever | " + " | ".join(f"C={c}" for c in SHORTLISTS) + " |")
    print("|---|" + "---|" * len(SHORTLISTS))
    asym_gain = [
        (results[("asym", "1")][c][0] - results[("hamming", "1")][c][0]) * 100 for c in SHORTLISTS
    ]
    print("| asymmetric minus symmetric (pp) | " + " | ".join(f"{g:+.2f}" for g in asym_gain) + " |")
    weight_gain = [
        (results[("asym", "1")][c][0] - results[("asym", "0")][c][0]) * 100 for c in SHORTLISTS
    ]
    print("| residual-norm weight, asym (pp) | " + " | ".join(f"{g:+.2f}" for g in weight_gain) + " |")
    weight_gain_ham = [
        (results[("hamming", "1")][c][0] - results[("hamming", "0")][c][0]) * 100 for c in SHORTLISTS
    ]
    print("| residual-norm weight, symmetric (pp) | " + " | ".join(f"{g:+.2f}" for g in weight_gain_ham) + " |")
    print()
    print("| lever | " + " | ".join(f"C={c}" for c in SHORTLISTS) + " |")
    print("|---|" + "---|" * len(SHORTLISTS))
    speed = [
        results[("asym", "1")][c][1] / results[("hamming", "1")][c][1] for c in SHORTLISTS
    ]
    print("| asymmetric QPS relative to symmetric | "
          + " | ".join(f"{s:.2f}x" for s in speed) + " |")
    print()
    print("DONE shortlist_scorer_ab")


if __name__ == "__main__":
    main()
