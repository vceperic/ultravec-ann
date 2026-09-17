#!/usr/bin/env python3
"""What trellis memory costs at index-build time.

Trellis memory is the paper's fixed-emission-rate quality lever: raising it
improves direction fidelity without changing a single emitted bit. The price is
paid entirely by the encoder, whose exact-Viterbi cost is O(D * 2^M * 2^b) per
vector, and a codec sold on being deployable without calibration owes a number for
that -- calibration-free means the corpus can be encoded immediately, not cheaply.

The manuscript previously asserted a "3.3 times" encode-time ratio between M=14 and
M=12 with no retained measurement behind it. This driver produces one, produces the
step to the reported M=16 operating point beside it, and produces the whole grid, so the memory sweep's recall column
(`appendix-ablations` section 4, manuscript Table "memory-sweep") has a
throughput column beside it at matching (M, rate) cells.

Timing uses the existing `recon --decode-bench` path, whose ENCODE timer wraps
`add_batch` alone: dataset load and decode are outside it. Encoding is rayon-parallel,
so the figure is whole-machine ingestion throughput at the recorded thread count, not
single-core cost.

Run: python3 scripts/encode_cost.py
"""
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("ULTRAVEC_BIN", ROOT / "target" / "release" / "ultravec"))
# The same normalized SIFT-100k base `appendix_ablations.py` sweeps, so a cell here
# lines up with the recall cell at the same (M, bits) rather than describing a
# different corpus.
BASE = ROOT / "data" / "erab" / "base.fvecs"
MEMORIES = [4, 6, 8, 10, 12, 14, 16]
# The memory steps the manuscript quotes by name. M=16 is the reported operating
# point, so its price belongs in the same grid as the quality it buys.
HEADLINE_STEPS = ((14, 12), (16, 14))
RATES = [2, 3, 4]
# Best-of-N over a repeated window, because a single window on a shared-tenancy host
# reports interference as though it were cost.
#
# MIN_RUNS is a floor on the *count*, not on cumulative time, and that distinction is
# the whole point. A "repeat until 2s have elapsed" rule gives every cheap cell five
# samples and every expensive cell exactly one -- and the expensive cells are the ones
# the manuscript's ratio is read from. It mattered: under that rule the headline
# M=14/M=12 ratio measured 5.24x, and at three samples per cell it is 4.73x.
#
# (Two anomalies that earlier versions of this comment recorded as properties of the
# encoder -- a 9.85x step at 4-bit M=10/M=8 and an 8.51x step at 2-bit M=14/M=12 --
# were neither. The first was a scalar add-compare-select beside a vectorized metric;
# the second was a per-vector zeroed allocation of the backpointer array. With both
# fixed every step sits near the 4x the recursion implies. The lesson the comment is
# worth keeping for: a ratio that reproduces is not thereby a property of the
# algorithm -- it can just as easily be a property of the implementation.)
MIN_RUNS = 3
MIN_SECONDS = 2.0
MAX_RUNS = 5

ENCODE = re.compile(r"ENCODE\s+([0-9.]+)s\s+\(([0-9]+)\s+vec/s\)")


def encode_once(memory: int, bits: int) -> tuple[float, int, int]:
    """One timed encode of the full base. Returns (seconds, vectors/s, vectors)."""
    env = dict(os.environ)
    env["ULTRAVEC_TRELLIS_MEM"] = str(memory)
    completed = subprocess.run(
        [
            str(BIN), "recon",
            "--dataset", str(BASE),
            "--backend", "trellis",
            "--bits", str(bits),
            "--decode-bench", "1",
        ],
        capture_output=True, text=True, env=env,
    )
    if completed.returncode:
        raise RuntimeError(
            f"recon failed (M={memory}, {bits} bit):\n{completed.stderr[-2000:]}"
        )
    match = ENCODE.search(completed.stdout)
    count = re.search(r":\s*(\d+)\s+vecs", completed.stdout)
    if not match or not count:
        raise RuntimeError(
            f"no ENCODE line parsed (M={memory}, {bits} bit):\n{completed.stdout[-2000:]}"
        )
    return float(match.group(1)), int(match.group(2)), int(count.group(1))


def measure(memory: int, bits: int) -> tuple[float, int, int, int, float]:
    """Best-of-N. Returns (best seconds, best vectors/s, runs, vectors, spread).

    `spread` is worst/best over the samples. It is reported rather than discarded
    because it is the only thing that distinguishes a clean measurement from one
    taken while a peer workload had the machine, and this host is explicitly
    shared-tenancy.
    """
    timings: list[float] = []
    best_seconds, best_rate, vectors = float("inf"), 0, 0
    spent = 0.0
    while len(timings) < MAX_RUNS and (len(timings) < MIN_RUNS or spent < MIN_SECONDS):
        started = time.monotonic()
        seconds, rate, vectors = encode_once(memory, bits)
        spent += time.monotonic() - started
        timings.append(seconds)
        if seconds < best_seconds:
            best_seconds, best_rate = seconds, rate
    return best_seconds, best_rate, len(timings), vectors, max(timings) / min(timings)


def main() -> None:
    seconds: dict[tuple[int, int], float] = {}
    rates: dict[tuple[int, int], int] = {}
    spreads: dict[tuple[int, int], float] = {}
    vectors = 0
    for memory in MEMORIES:
        for bits in RATES:
            best_seconds, best_rate, count, vectors, spread = measure(memory, bits)
            seconds[(memory, bits)] = best_seconds
            rates[(memory, bits)] = best_rate
            spreads[(memory, bits)] = spread
            print(
                f"# M={memory} {bits}-bit: {best_seconds:.3f}s "
                f"({best_rate} vec/s), best of {count}, spread {spread:.2f}x",
                flush=True,
            )
    print()
    print(
        f"Encode throughput versus trellis memory at a fixed emitted rate — "
        f"{vectors} normalized SIFT-128 vectors, exact Viterbi, "
        f"{os.environ.get('RAYON_NUM_THREADS', 'default')} threads"
    )
    print(
        "The emitted payload is identical along each column; only the encoder's "
        "search cost changes. Best of up to "
        f"{MAX_RUNS} timed encodes per cell."
    )
    print()
    print("| M | 2-bit vec/s | 3-bit vec/s | 4-bit vec/s |")
    print("|---|---|---|---|")
    for memory in MEMORIES:
        cells = " | ".join(f"{rates[(memory, bits)]}" for bits in RATES)
        print(f"| {memory} | {cells} |")
    print()

    # The ratios the manuscript quotes when it prices the memory lever. Emitted for
    # every consecutive step so the shape of the cost curve is visible rather than
    # just the one cell the prose happens to name.
    print("=== consecutive memory-step encode-time ratios ===")
    print("| step | 2 bits | 3 bits | 4 bits |")
    print("|---|---|---|---|")
    for lower, upper in zip(MEMORIES, MEMORIES[1:]):
        cells = " | ".join(
            f"{seconds[(upper, bits)] / seconds[(lower, bits)]:.2f}" for bits in RATES
        )
        print(f"| M={upper} / M={lower} | {cells} |")
    print()
    for upper, lower in HEADLINE_STEPS:
        ratio = seconds[(upper, 2)] / seconds[(lower, 2)]
        print(f"M={upper} over M={lower} at two bits: {ratio:.2f}x encode time")
    print()

    # Worst spread over the grid, so a reader can see at a glance whether the run was
    # clean. Exact Viterbi predicts 2^(M2-M1) for a memory step; a cell whose own
    # samples disagree by more than the effect being measured is not evidence.
    worst = max(spreads.items(), key=lambda item: item[1])
    print(
        f"sample spread: worst {worst[1]:.2f}x at M={worst[0][0]} {worst[0][1]}-bit; "
        f"median {sorted(spreads.values())[len(spreads) // 2]:.2f}x "
        f"over {len(spreads)} cells, minimum {MIN_RUNS} samples each"
    )
    print()
    print("DONE encode_cost")


if __name__ == "__main__":
    main()
