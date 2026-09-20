#!/usr/bin/env python3
"""Measure direction fidelity as a function of trellis memory, at a fixed emission rate.

The manuscript's central empirical observation is that trellis memory raises
reconstruction-direction fidelity `g` *without changing the emitted rate*. The
per-corpus diagnostics in `mechanism-inputs` use one campaign operating point, and
the appendix memory sweep reports Recall@10. This stage supplies the complementary
`g`-against-M measurement by sweeping M at a fixed two bits per transformed
dimension on SIFT-100k -- the same corpus, query set and seed as the appendix
ablations -- and reports the trellis row's `g` at each M.

The sweep runs through M=16, the memory the flat comparison reports, so the
mechanism table covers the operating point rather than stopping below it.

The emitted rate is constant across the sweep by construction: M changes the
encoder's state count, not the number of emitted bits. The stored record grows by
`ceil(M/8)` bytes for the start state, which is reported so the reader can see it
is not a rate change in disguise.

E-RaBitQ is re-reported at every M as a fixed reference. Its flat line confirms
that only the trellis configuration changes across the sweep.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "target" / "release" / "ultravec"
OUTPUT = ROOT / "results" / "generated" / "mechanism-msweep"

# SIFT-100k at two bits: matches the appendix ablation setting so the g sweep and
# the recall sweep in tab:memory-sweep describe the same experiment.
DATASET = "data/sift/sift_base.fvecs"
QUERIES = "data/sift/sift_query.fvecs"
MAXIMUM = 100_000
QUERY_MAX = 200
# `diag` loads QUERY_MAX queries, then scores SAMPLE_Q of them against a seeded
# SAMPLE_DB subsample. The appendix recall ablations use 1,000 queries over the
# complete 100,000-vector base; both settings are stated in the output.
SAMPLE_DB = 1000
SAMPLE_Q = 100
BITS = 2
SEED = 42
MEMORIES = (2, 4, 6, 8, 10, 12, 14, 16)


def parse_diag(text: str) -> dict[str, dict[str, float]]:
    """{backend: {m, g, kappa, sigma_pred, sigma_emp, bias, mse}} from a diag table."""
    out: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|") or set(line) <= set("|-: "):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 9 or cells[0] == "backend":
            continue
        try:
            out[cells[0]] = dict(
                m=float(cells[2]), g=float(cells[3]), kappa=float(cells[4]),
                sigma_pred=float(cells[5]), sigma_emp=float(cells[6]),
                bias=float(cells[7].replace("+", "")), mse=float(cells[8]))
        except ValueError:
            continue
    return out


def main() -> int:
    if not BIN.is_file():
        raise FileNotFoundError(f"release binary is missing: {BIN}")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    rows: list[tuple[int, dict[str, float]]] = []
    rabitq_g: list[float] = []

    for memory in MEMORIES:
        env["ULTRAVEC_TRELLIS_MEM"] = str(memory)
        argv = [
            str(BIN), "diag",
            "--dataset", DATASET, "--query-file", QUERIES,
            "--query-max", str(QUERY_MAX), "--max", str(MAXIMUM),
            "--sample-db", str(SAMPLE_DB), "--sample-q", str(SAMPLE_Q),
            "--bits", str(BITS), "--seed", str(SEED),
        ]
        print(f"[msweep] M={memory}", flush=True)
        completed = subprocess.run(argv, cwd=ROOT, env=env, text=True,
                                   capture_output=True, check=False)
        if completed.returncode:
            raise RuntimeError(f"{' '.join(argv)} failed:\n{completed.stderr[-2000:]}")
        # Normalize the artifact root to "." exactly as reproduce.py does for the
        # stdout it captures. This tier writes its own per-M logs rather than going
        # through record_bundle, so without this the absolute checkout path (and
        # whatever the working directory happens to be called) is committed verbatim.
        (OUTPUT / f"diag-sift-b{BITS}-m{memory}.txt").write_text(
            completed.stdout.replace(str(ROOT), "."), encoding="utf-8")

        parsed = parse_diag(completed.stdout)
        if "trellis" not in parsed:
            raise RuntimeError(f"no trellis row in diag output at M={memory}")
        observed = int(parsed["trellis"]["m"])
        if observed != memory:
            raise RuntimeError(
                f"ULTRAVEC_TRELLIS_MEM={memory} but the diag table reports M={observed}; "
                "the sweep is not actually varying trellis memory")
        rows.append((memory, parsed["trellis"]))
        if "rabitq" in parsed:
            rabitq_g.append(parsed["rabitq"]["g"])

    # Emitted rate is D*bits regardless of M; only the start state changes size.
    emitted_bits = BITS
    print()
    print("# Trellis memory sweep — direction fidelity at a fixed emitted rate")
    print()
    print(f"- SIFT-100k, {SAMPLE_Q} sampled queries against a seeded {SAMPLE_DB}-vector "
          f"database subsample, {BITS} bits/dim, seed {SEED}")
    print(f"- same corpus, rate and seed as the appendix ablations, which instead score "
          f"1,000 queries over the full base")
    print(f"- emitted bits per transformed dimension is {emitted_bits} at every M; "
          "only the start-state field (ceil(M/8) B) changes size")
    print()
    print("| M | start-state B | mean g | kappa | sigma_pred | sigma_emp | recon MSE |")
    print("|---|---|---|---|---|---|---|")
    for memory, r in rows:
        print(f"| {memory} | {math.ceil(memory / 8)} | {r['g']:.4f} | {r['kappa']:.4f} | "
              f"{r['sigma_pred']:.5f} | {r['sigma_emp']:.5f} | {r['mse']:.5f} |")
    print()

    first, last = rows[0][1]["g"], rows[-1][1]["g"]
    monotone = all(b[1]["g"] >= a[1]["g"] - 1e-9 for a, b in zip(rows, rows[1:]))
    print(f"- g rises from {first:.4f} at M={rows[0][0]} to {last:.4f} at "
          f"M={rows[-1][0]} (delta {last - first:+.4f}); monotone non-decreasing: {monotone}")
    print(f"- estimator std falls from {rows[0][1]['sigma_emp']:.5f} to "
          f"{rows[-1][1]['sigma_emp']:.5f} over the same sweep")
    if rabitq_g:
        spread = max(rabitq_g) - min(rabitq_g)
        print(f"- E-RaBitQ reference (no memory lever): g = {rabitq_g[0]:.4f}, "
              f"spread across the sweep {spread:.4f} — flat, as expected")
    # The bundle is accepted only when the measured fidelity sequence satisfies
    # the monotonic fixed-emission-rate claim it is designed to test.
    if not monotone:
        raise SystemExit(
            "fidelity is not non-decreasing in trellis memory over "
            f"M={[m for m, _ in rows]}: g = {[round(r['g'], 4) for _, r in rows]}. "
            "The claim this bundle evidences does not hold on this run.")
    if last <= first:
        raise SystemExit(
            f"fidelity did not rise across the sweep: {first:.4f} at M={rows[0][0]} "
            f"to {last:.4f} at M={rows[-1][0]}.")

    print()
    print("DONE mechanism_msweep")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
