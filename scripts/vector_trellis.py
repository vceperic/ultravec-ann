#!/usr/bin/env python3
"""Is the scalar trellis the right member of its own family?

Every block-based comparator in the flat table is a *vector* quantizer: PQ, E8 and
BlockQuant all quantize a group of coordinates jointly, and that joint treatment is
where their shape gain comes from. UltraVec is reported only in its scalar form,
`V=1`, one coordinate per trellis step -- even though the implementation has carried
a `V` knob throughout. So the manuscript compares a scalar codec against vector
codecs and never asks whether its own vector form is better.

At `V` coordinates per step the emission alphabet is `bits*V` bits wide and there
are `dim/V` steps, so the emitted payload is `dim*bits` bits for any `V`: the rate is
fixed and only the shape of the quantization cell changes. Larger `V` tiles space
with rounder Voronoi regions, which is the same argument that makes a lattice beat a
product code at matched rate.

What it costs is table and encode time. The reconstruction table holds
`2^(mem + bits*V) * V` floats, so each step of `V` multiplies it by `2^bits`, and the
Viterbi column grows with it. `bits*V <= 16` is asserted in the codec, which bounds
the sweep: at two bits `V` reaches 8, at four bits only 4.

Reports recall at matched emitted rate, the serialized record, and encode
throughput, so the trade is visible rather than asserted.

Run: python3 scripts/vector_trellis.py
"""
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("ULTRAVEC_BIN", ROOT / "target" / "release" / "ultravec"))
BASE = ROOT / "data" / "erab" / "base.fvecs"
QUERY = ROOT / "data" / "erab" / "query.fvecs"
MEM = "12"
# (bits, V values). bits*V <= 16 is asserted by the codec, and the table is
# 2^(mem+bits*V) floats, so the ladder stops where a run would need gigabytes.
GRID = [(2, [1, 2, 4]), (3, [1, 2]), (4, [1, 2])]
RECALL_ROW = re.compile(r"^\|\s*trellis\s*\|\s*(\d)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|")
ENCODE = re.compile(r"ENCODE\s+([0-9.]+)s\s+\(([0-9]+)\s+vec/s\)")


def env_for(v: int) -> dict:
    env = dict(os.environ)
    env["ULTRAVEC_TRELLIS_MEM"] = MEM
    env["ULTRAVEC_TRELLIS_V"] = str(v)
    return env


def recall(bits: int, v: int) -> tuple[float, float, float, int]:
    """Recall@1/10/100 and serialized bytes for one (bits, V) cell."""
    done = subprocess.run(
        [
            str(BIN), "bench",
            "--dataset", str(BASE), "--query-file", str(QUERY),
            "--query-max", "1000", "--max", "100000", "--bits", str(bits),
            "--seed", "42", "--sota3",
        ],
        capture_output=True, text=True, env=env_for(v),
    )
    if done.returncode:
        raise RuntimeError(f"bench failed (bits={bits}, V={v}):\n{done.stderr[-2000:]}")
    for line in done.stdout.splitlines():
        m = RECALL_ROW.match(line)
        if m and int(m.group(1)) == bits:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            # trellis | bits | r1 | r10 | r100 | mse | ... | code B | resident B
            return float(cells[2]), float(cells[3]), float(cells[4]), int(cells[-2])
    raise RuntimeError(f"no trellis row parsed (bits={bits}, V={v}):\n{done.stdout[-2000:]}")


def encode_rate(bits: int, v: int) -> int:
    """Best-of-3 encode throughput, matching scripts/encode_cost.py's convention."""
    best = 0
    for _ in range(3):
        done = subprocess.run(
            [
                str(BIN), "recon", "--dataset", str(BASE),
                "--backend", "trellis", "--bits", str(bits), "--decode-bench", "1",
            ],
            capture_output=True, text=True, env=env_for(v),
        )
        if done.returncode:
            raise RuntimeError(f"recon failed (bits={bits}, V={v}):\n{done.stderr[-2000:]}")
        m = ENCODE.search(done.stdout)
        if not m:
            raise RuntimeError(f"no ENCODE line (bits={bits}, V={v})")
        best = max(best, int(m.group(2)))
    return best


def main() -> None:
    print(f"Vector trellis: recall and cost against V at a fixed emitted rate "
          f"(normalized SIFT-128, 100,000 base, 1,000 queries, mem={MEM}, seed 42)")
    print("The emitted payload is dim*bits bits for every V; only the cell shape, the "
          "table size and the encode cost change.")
    print()
    print("| bits | V | table entries | R@1 | R@10 | R@100 | code B | encode vec/s |")
    print("|---|---|---|---|---|---|---|---|")
    rows: dict[tuple[int, int], float] = {}
    for bits, vs in GRID:
        for v in vs:
            started = time.monotonic()
            r1, r10, r100, code_b = recall(bits, v)
            rate = encode_rate(bits, v)
            rows[(bits, v)] = r10
            entries = 1 << (int(MEM) + bits * v)
            print(f"| {bits} | {v} | {entries:,} | {r1:.4f} | {r10:.4f} | {r100:.4f} "
                  f"| {code_b} | {rate:,} |",
                  flush=True)
            _ = time.monotonic() - started
    print()
    print("=== V>1 minus V=1, Recall@10 (percentage points) ===")
    print("| bits | V | delta pp |")
    print("|---|---|---|")
    for bits, vs in GRID:
        for v in vs:
            if v == 1:
                continue
            print(f"| {bits} | {v} | {(rows[(bits, v)] - rows[(bits, 1)]) * 100:+.2f} |")
    print()
    print("DONE vector_trellis")


if __name__ == "__main__":
    main()
