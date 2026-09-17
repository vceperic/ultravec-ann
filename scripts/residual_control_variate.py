#!/usr/bin/env python3
"""Should UltraVec spend a bit of base fidelity on an unbiased residual correction?

The codec scores with the rescaled estimator. This script measures a second estimator
the codec implements but does not use by default: quantize as usual, then store the
residual's norm plus the SIGN
bits of an independent SRHT sketch of it, and correct the inner product by
`sqrt(pi/2)/m * ||r|| * <sketch(q), signs>`, which is unbiased for the residual's
contribution. `residual.rs` gives the same construction on a scalar base, where the
budget is meant to balance: one bit per dimension less in the base pays for one bit per
dimension of sketch.

That framing is what makes this a real question rather than a free lunch, and the knob
does NOT enforce it -- `ULTRAVEC_TRELLIS_RESID` leaves the base rate alone, so turning
it on at b bits simply spends more bytes. The arms below therefore drop the base rate
by one and pass `m = dim`:

    control  b bits, no correction
    variant  b-1 bits + dim sign bits + one f32 residual norm

which leaves the variant 4 bytes per vector heavier (the norm). That is disclosed in
the table rather than rounded away: the variant has to win by more than those 4 bytes
are worth, and at these rates 4 bytes is 2-7% of the record.

The prior is not neutral. The same technique on TurboQuant's scalar base -- the `TQ-IP`
column of the flat table -- gains 4.8 points at 2 bits and loses 7.7 and 13.3 at 3 and
4, because the correction's variance scales with the residual it is correcting. A
coarse base has a large residual and the correction is worth its bits; a fine base does
not. UltraVec's remaining losses are all at one bit, which is the regime where this
should help if it ever does.

Protocol is the flat comparison's own -- raw inner-product gold, 100k base, 1,000
queries, seed 42 -- so a row here is directly comparable to Table `tab:common-r10`.

Run: python3 scripts/residual_control_variate.py
"""
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("ULTRAVEC_BIN", ROOT / "target" / "release" / "ultravec"))
BASE = ROOT / "data" / "sift" / "sift_base.fvecs"
QUERY = ROOT / "data" / "sift" / "sift_query.fvecs"
# SIFT is 128-dimensional and already a power of two, so the transformed dimension --
# and therefore the sketch width for a one-bit-per-dimension budget -- is 128.
DIM = 128
PAIRS = [(2, 1), (3, 2), (4, 3)]  # (control bits, variant base bits)
ROW = re.compile(
    r"^\|\s*trellis\s*\|\s*(\d)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
    r"\s*\S+\s*\|\s*([\d.]+)\s*\|\s*(\d+)\s*\|"
)


def run(bits: list[int], resid: int) -> dict[int, tuple[float, float, float, int]]:
    """(R@1, R@10, R@100, code bytes) per rate for the trellis arm."""
    env = dict(os.environ, ULTRAVEC_TRELLIS_MEM=os.environ.get("ULTRAVEC_TRELLIS_MEM", "12"))
    if resid:
        env["ULTRAVEC_TRELLIS_RESID"] = str(resid)
    else:
        env.pop("ULTRAVEC_TRELLIS_RESID", None)
    completed = subprocess.run(
        [
            str(BIN), "bench",
            "--dataset", str(BASE), "--query-file", str(QUERY),
            "--query-max", "1000", "--max", "100000",
            "--bits", ",".join(str(b) for b in bits),
            "--metric", "ip", "--seed", "42", "--sota3",
        ],
        capture_output=True, text=True, env=env,
    )
    if completed.returncode:
        raise RuntimeError(f"bench failed (resid={resid}):\n{completed.stderr[-2000:]}")
    out: dict[int, tuple[float, float, float, int]] = {}
    for line in completed.stdout.splitlines():
        match = ROW.match(line)
        if match:
            out[int(match.group(1))] = (
                float(match.group(2)), float(match.group(3)),
                float(match.group(4)), int(match.group(6)),
            )
    missing = sorted(set(bits) - set(out))
    if missing:
        raise RuntimeError(f"incomplete output (resid={resid}); missing bits={missing}")
    return out


def main() -> None:
    print(
        f"# Residual control variate on the trellis base — SIFT-100k, 1,000 queries, "
        f"raw-IP gold, seed 42, M=12, uncentered"
    )
    print()
    print(
        "Bit-matched by construction: the variant drops one bit per dimension from the "
        "base and spends it on the sketch. The 4-byte residual norm is extra and is "
        "reported, not hidden."
    )
    print()
    rates = sorted({b for b, _ in PAIRS} | {v for _, v in PAIRS})
    control = run(rates, 0)
    variant = run(rates, DIM)

    print("## Bit-matched: one bit per dimension of base traded for the sketch")
    print()
    print("| control bits | variant base bits | control B | variant B | dB | "
          "control R@10 | variant R@10 | delta pp |")
    print("|---|---|---|---|---|---|---|---|")
    for cb, vb in PAIRS:
        _, c_r10, _, c_bytes = control[cb]
        _, v_r10, _, v_bytes = variant[vb]
        print(
            f"| {cb} | {vb} | {c_bytes} | {v_bytes} | {v_bytes - c_bytes:+d} | "
            f"{c_r10:.4f} | {v_r10:.4f} | {(v_r10 - c_r10) * 100:+.2f} |"
        )
    print()
    print("## Same base rate: what the correction is worth for its own bytes")
    print()
    print("| bits | plain B | corrected B | dB | plain R@10 | corrected R@10 | delta pp |")
    print("|---|---|---|---|---|---|---|")
    for b in rates:
        _, c_r10, _, c_bytes = control[b]
        _, v_r10, _, v_bytes = variant[b]
        print(
            f"| {b} | {c_bytes} | {v_bytes} | {v_bytes - c_bytes:+d} | "
            f"{c_r10:.4f} | {v_r10:.4f} | {(v_r10 - c_r10) * 100:+.2f} |"
        )
    print()
    # R@1 and R@100 too: the correction changes the estimator, and an estimator change
    # can move the head of the ranking and its tail in opposite directions.
    print("| control bits | R@1 c/v | R@100 c/v |")
    print("|---|---|---|")
    for cb, vb in PAIRS:
        c_r1, _, c_r100, _ = control[cb]
        v_r1, _, v_r100, _ = variant[vb]
        print(f"| {cb} | {c_r1:.4f} / {v_r1:.4f} | {c_r100:.4f} / {v_r100:.4f} |")
    print()
    print("Bit-matched pairs above; the same-rate table pairs each rate with itself.")
    print()
    print("DONE residual_control_variate")


if __name__ == "__main__":
    main()
