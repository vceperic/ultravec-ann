#!/usr/bin/env python3
"""Rotation-to-rotation variability for the narrow-margin cells.

Every codec in the flat comparison is a randomized construction sharing one seeded
sign flip, and the reported cells are a single draw from that distribution. The
paired query bootstrap therefore covers query sampling and nothing else, which is
adequate for a wide margin and not adequate for a cell decided by around one point.
GIST and DBpedia--OpenAI are exactly those cells: at two-bit GIST the interval
against E8 already includes zero.

This driver re-runs the flat comparison under several rotation seeds
(`ULTRAVEC_ROTATION_SEED`, which moves every codec together) and reports each
codec's mean and spread across draws, plus UltraVec's margin over its closest
comparator per draw. A margin that survives every draw is a property of the method;
one that changes sign is a property of the seed, and the manuscript should say so.

`--seed 42` is held fixed throughout: the query split and database subsample must not
move, or rotation variance and query variance would be confounded.

Run: python3 scripts/rotation_seed_ci.py
"""
import argparse
import math
import os
import re
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("ULTRAVEC_BIN", ROOT / "target" / "release" / "ultravec"))
ROTATIONS = [42, 7, 13, 29, 101]
# Rates are per corpus because the two corpora need different coverage and exact
# Viterbi encode cost grows as 2^bits, so sweeping every rate everywhere buys little
# and costs a great deal.
#
# GIST is swept at every reported rate. Its 2-bit cell was already decided by the
# draw, and its 3- and 4-bit margins over the best comparator (+0.87 and +0.46 pp)
# are at or below the +-0.52 pp draw interval measured at 2 bits -- so claiming them
# on the query bootstrap alone asserts exactly what this experiment showed to be
# unreliable one rate below.
#
# DBpedia stays at 1-2 bits. It is the low-variance corpus (0.3-0.8 pp spread per
# codec against GIST's 0.7-3.0), and both swept cells lead in all five draws, so its
# 3- and 4-bit margins of +1.22 and +1.04 pp are not the ones in doubt.
CORPORA = [
    ("GIST", ROOT / "data" / "gist" / "gist_base.fvecs",
     ROOT / "data" / "gist" / "gist_query.fvecs", (1, 2, 3, 4)),
    ("DBpedia", ROOT / "data" / "dbpedia" / "dbpedia_base.fvecs",
     ROOT / "data" / "dbpedia" / "dbpedia_query.fvecs", (1, 2)),
]
ROW = re.compile(r"^\|\s*([a-z0-9_]+)\s*\|\s*(\d)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|")


# t(0.975, df) for the draw counts this driver can produce. Hardcoding 2.776 was
# correct only for the published five draws; with `--draws` it would report an interval
# that is simply the wrong width, which is worse than a wide one.
_T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    24: 2.064, 29: 2.045, 39: 2.023, 49: 2.010,
}


def t_multiplier(n: int) -> float:
    """Two-sided 95% t multiplier for `n` observations (df = n-1)."""
    df = max(1, n - 1)
    if df in _T975:
        return _T975[df]
    # Above the table, interpolate down toward the normal limit rather than guess high.
    larger = [k for k in sorted(_T975) if k > df]
    return _T975[larger[0]] if larger else 1.960


def run(base: Path, query: Path, rotation: int, bits: tuple[int, ...]) -> dict[tuple[str, int], float]:
    """Recall@10 per (codec, bits) for one rotation draw."""
    env = dict(os.environ)
    env["ULTRAVEC_TRELLIS_MEM"] = env.get("ULTRAVEC_TRELLIS_MEM", "12")
    env["ULTRAVEC_ROTATION_SEED"] = str(rotation)
    completed = subprocess.run(
        [
            str(BIN), "bench",
            "--dataset", str(base), "--query-file", str(query),
            "--query-max", "1000", "--max", "100000",
            "--bits", ",".join(str(b) for b in bits),
            "--metric", "ip", "--seed", "42", "--sota3",
        ],
        capture_output=True, text=True, env=env,
    )
    if completed.returncode:
        raise RuntimeError(f"bench failed (rotation {rotation}):\n{completed.stderr[-2000:]}")
    out: dict[tuple[str, int], float] = {}
    for line in completed.stdout.splitlines():
        match = ROW.match(line)
        if match:
            out.setdefault((match.group(1), int(match.group(2))), float(match.group(4)))
    if not out:
        raise RuntimeError(f"bench produced no parsable rows (rotation {rotation})")
    return out


def main() -> None:
    # The rate and corpus lists are restrictable because the study is re-run at a second
    # state memory for the one-bit cells only: sweeping GIST's four rates at M=14 costs
    # hours and answers a question the M=12 run already settled.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates", help="comma-separated bit rates to restrict the sweep to")
    parser.add_argument("--corpora", help="comma-separated corpus names to restrict the sweep to")
    parser.add_argument(
        "--draws", type=int,
        help="number of rotation draws (default 5). The t-interval on the mean margin "
             "narrows as t(0.975,n-1)/sqrt(n), so 5->15 draws is about 2.2x tighter: "
             "that is the binding constraint on the narrow GIST cells, not query count.",
    )
    args = parser.parse_args()
    if args.draws:
        if args.draws < 2:
            raise SystemExit("--draws must be at least 2 for an interval")
        # Extend deterministically from the published five so the original draws stay a
        # prefix: a reader comparing against the retained bundle sees the same first
        # five seeds, and the extra ones are reproducible from this rule alone.
        extra = [1_000_003 * (i + 1) % 99_991 for i in range(args.draws)]
        seen, merged = set(), []
        for seed in ROTATIONS + extra:
            if seed not in seen:
                seen.add(seed)
                merged.append(seed)
            if len(merged) == args.draws:
                break
        globals()["ROTATIONS"] = merged
    corpora = CORPORA
    if args.corpora:
        wanted = {name.strip().lower() for name in args.corpora.split(",")}
        corpora = [c for c in corpora if c[0].lower() in wanted]
        if not corpora:
            raise SystemExit(f"no corpus matched {args.corpora!r}")
    if args.rates:
        rates = tuple(int(r) for r in args.rates.split(",") if r.strip())
        corpora = [
            (name, base, query, tuple(r for r in existing if r in rates))
            for name, base, query, existing in corpora
        ]
        corpora = [c for c in corpora if c[3]]
        if not corpora:
            raise SystemExit(f"no corpus supports rates {args.rates!r}")
    globals()["CORPORA"] = corpora
    # State memory belongs in the header: a rotation study is only evidence for the
    # cells reported at the same M, and a bundle that does not say which memory it
    # swept can be read against a table it does not cover.
    print(f"Rotation-draw variability — 100k base, 1,000 queries, query seed 42, "
          f"{len(ROTATIONS)} rotation draws {ROTATIONS}, "
          f"trellis memory M={os.environ.get('ULTRAVEC_TRELLIS_MEM', '12')}")
    print("Every codec shares the drawn rotation, so a draw moves the whole field together.")
    print("Rates swept: " + "; ".join(
        f"{name} {','.join(str(b) for b in rates)} bit" for name, _, _, rates in CORPORA
    ))
    print()
    summary: list[tuple[str, int, float, float, float, float, str]] = []
    for name, base, query, rates in CORPORA:
        draws = [run(base, query, rotation, rates) for rotation in ROTATIONS]
        codecs = sorted({codec for draw in draws for codec, _ in draw})
        for bits in rates:
            present = [c for c in codecs if all((c, bits) in d for d in draws)]
            if not present:
                continue
            print(f"--- {name}, {bits} bit")
            print("| codec | mean R@10 | min | max | spread pp |")
            print("|---|---|---|---|---|")
            for codec in present:
                values = [d[(codec, bits)] for d in draws]
                print(f"| {codec} | {statistics.fmean(values):.4f} | {min(values):.4f} "
                      f"| {max(values):.4f} | {(max(values) - min(values)) * 100:.2f} |")
            print()
            # Two margin statistics, because they answer different questions and only
            # one of them matches the protocol Section 5.4 declares.
            #
            # PER-COMPARATOR is the one the manuscript's own convention implies: the
            # paired bootstrap requires the interval against every comparator
            # INDIVIDUALLY to exclude zero (an intersection-union test), so the
            # rotation axis should ask the same thing of the draw distribution.
            #
            # FIELD-MAX subtracts the best comparator WITHIN each draw. That is a
            # different and much harder bar: with a field whose means cluster inside a
            # tenth of a point but whose draws spread over two, the maximum of seven
            # noisy values sits well above any single comparator's mean, so the
            # statistic charges UltraVec for the SIZE of the field rather than for
            # anything about the codec -- adding an eighth mediocre-but-noisy
            # comparator would lower the reported margin without any measurement
            # changing. It is retained and printed because it is the conservative
            # reading, but the per-comparator rows are the ones the manuscript's
            # protocol licenses.
            others = [c for c in present if c != "trellis"]
            per_comparator: list[tuple[str, float, float, float, float, str]] = []
            for comparator in others:
                values = [
                    (draw[("trellis", bits)] - draw[(comparator, bits)]) * 100.0
                    for draw in draws
                ]
                mean_c = statistics.fmean(values)
                half_c = (
                    t_multiplier(len(values)) * statistics.stdev(values) / math.sqrt(len(values))
                    if len(values) > 1 else 0.0
                )
                verdict_c = ("leads-every-draw" if min(values) > 0 else
                             "trails-every-draw" if max(values) < 0 else "sign-changes")
                per_comparator.append(
                    (comparator, mean_c, half_c, min(values), max(values), verdict_c)
                )
            if per_comparator:
                print(f"--- {name}, {bits} bit, margin against each comparator")
                print("| comparator | margin pp | CI half | min | max | verdict |")
                print("|---|---|---|---|---|---|")
                for comparator, mean_c, half_c, low_c, high_c, verdict_c in per_comparator:
                    print(f"| {comparator} | {mean_c:+.2f} | {half_c:.2f} | {low_c:+.2f} "
                          f"| {high_c:+.2f} | {verdict_c} |")
                narrowest = min(per_comparator, key=lambda row: row[1])
                separated = all(row[1] - row[2] > 0 for row in per_comparator)
                print(f"closest comparator {narrowest[0]} at {narrowest[1]:+.2f} pp; "
                      f"every comparator interval excludes zero: {separated}")
                print()

            margins = [
                (draw[("trellis", bits)] - max(draw[(c, bits)] for c in present if c != "trellis")) * 100.0
                for draw in draws
                if ("trellis", bits) in draw and any(c != "trellis" for c in present)
            ]
            if margins:
                mean = statistics.fmean(margins)
                half = (t_multiplier(len(margins)) * statistics.stdev(margins) / math.sqrt(len(margins))
                        if len(margins) > 1 else 0.0)
                verdict = ("leads-every-draw" if min(margins) > 0 else
                           "trails-every-draw" if max(margins) < 0 else "sign-changes")
                summary.append((name, bits, mean, half, min(margins), max(margins), verdict))

    # Machine-readable summary, one row per cell, in the shape the manuscript table
    # uses. Emitted last so a gate reads it without walking the per-codec blocks.
    print("=== margin summary ===")
    print("| corpus | bits | margin pp | CI half | min | max | verdict |")
    print("|---|---|---|---|---|---|---|")
    for name, bits, mean, half, low, high, verdict in summary:
        print(f"| {name} | {bits} | {mean:+.2f} | {half:.2f} | {low:+.2f} | {high:+.2f} | {verdict} |")
    print()
    print("DONE rotation_seed_ci")


if __name__ == "__main__":
    main()
