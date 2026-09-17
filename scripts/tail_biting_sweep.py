#!/usr/bin/env python3
"""Tail-biting across the three corpora: what the start-state byte costs, and where.

Tail-biting constrains the Viterbi path to end in the state it starts from. The
trellis state is the last M emitted bits, so a wrapped path's start is a suffix of
its own codes: derivable rather than stored, which drops the ceil(M/8)-byte start
field and puts the record at byte parity with RaBitQ.

The price is that the wrap pins the last ceil(M/b) emissions, a fraction M/(D*b) of
the vector that shrinks with dimension. On SIFT (D=128, 2 bits) that is 6.25% of the
payload against a 5.26% byte saving; on GIST (D=1024) 0.78%; on DBpedia (D=1536)
0.52%. This sweep measures all three so the lever is reported where it pays and
where it does not, at the reported operating point (M=16) and across the same five
rotation draws as the rotation study, because two-bit GIST is decided by the draw.

Every row is UltraVec against itself; no comparator is touched.
"""
import os
import re
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("ULTRAVEC_BIN", ROOT / "target" / "release" / "ultravec"))
ROW = re.compile(
    r"^\|\s*([a-z0-9_]+)\s*\|\s*(\d)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
    r"\s*[^|]*\|\s*[^|]*\|\s*(\d+)\s*\|"
)
ROTATIONS = [42, 7, 13, 29, 101]
CORPORA = [
    ("SIFT", ROOT / "data" / "sift" / "sift_base.fvecs", ROOT / "data" / "sift" / "sift_query.fvecs", 128),
    ("GIST", ROOT / "data" / "gist" / "gist_base.fvecs", ROOT / "data" / "gist" / "gist_query.fvecs", 1024),
    ("DBpedia", ROOT / "data" / "dbpedia" / "dbpedia_base.fvecs", ROOT / "data" / "dbpedia" / "dbpedia_query.fvecs", 1536),
]
BITS = 2
MEM = int(os.environ.get("ULTRAVEC_TRELLIS_MEM", "16"))


def run(base: Path, query: Path, rotation: int, tail: bool) -> tuple[float, float, int]:
    """(Recall@10, Recall@100, code bytes) for the trellis row of one bench run."""
    env = dict(os.environ)
    env["ULTRAVEC_TRELLIS_MEM"] = str(MEM)
    env["ULTRAVEC_ROTATION_SEED"] = str(rotation)
    env["ULTRAVEC_TRELLIS_TAILBITE"] = "1" if tail else "0"
    env["ULTRAVEC_TRELLIS_TAILBITE_K"] = "1"
    completed = subprocess.run(
        [str(BIN), "bench", "--dataset", str(base), "--query-file", str(query),
         "--query-max", "1000", "--max", "100000", "--bits", str(BITS),
         "--metric", "ip", "--seed", "42", "--sota3"],
        capture_output=True, text=True, env=env,
    )
    if completed.returncode:
        raise RuntimeError(f"bench failed (rotation {rotation}, tail={tail}):\n{completed.stderr[-2000:]}")
    for line in completed.stdout.splitlines():
        m = ROW.match(line)
        if m and m.group(1) == "trellis":
            return float(m.group(4)), float(m.group(5)), int(m.group(6))
    raise RuntimeError(f"no trellis row (rotation {rotation}, tail={tail})")


def main() -> None:
    print(f"Tail-biting sweep — 100k base, 1,000 queries, {BITS} bits, trellis memory M={MEM}, "
          f"{len(ROTATIONS)} rotation draws {ROTATIONS}")
    print("Free start against tail-biting; every row is UltraVec against itself.\n")
    summary = []
    for name, base, query, dim in CORPORA:
        pinned = -(-MEM // BITS)
        print(f"--- {name} (D={dim}: wrap pins {pinned}/{dim} emissions = {100*pinned/dim:.2f}% of payload)")
        print("| draw | free R@10 | tail R@10 | delta pp | free R@100 | tail R@100 | free B | tail B |")
        print("|---|---|---|---|---|---|---|---|")
        deltas = []
        fb = tb = 0
        for rot in ROTATIONS:
            f10, f100, fb = run(base, query, rot, False)
            t10, t100, tb = run(base, query, rot, True)
            d = (t10 - f10) * 100
            deltas.append(d)
            print(f"| {rot} | {f10:.4f} | {t10:.4f} | {d:+.2f} | {f100:.4f} | {t100:.4f} | {fb} | {tb} |")
        mean = statistics.fmean(deltas)
        sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
        print(f"mean delta {mean:+.2f} pp (sd {sd:.2f}); bytes {fb} -> {tb}\n")
        summary.append((name, dim, fb, tb, mean, min(deltas), max(deltas)))
    print("=== summary ===")
    print("| corpus | D | free B | tail B | mean delta pp | min | max |")
    print("|---|---|---|---|---|---|---|")
    for name, dim, fb, tb, mean, lo, hi in summary:
        print(f"| {name} | {dim} | {fb} | {tb} | {mean:+.2f} | {lo:+.2f} | {hi:+.2f} |")
    print("\nDONE tail_biting_sweep")


if __name__ == "__main__":
    main()
