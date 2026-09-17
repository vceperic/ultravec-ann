#!/usr/bin/env python3
"""Generate the per-corpus diagnostic and recall tables used by mechanism_pvalues.py."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

import evidence_paths
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "target" / "release" / "ultravec"
OUTPUT = ROOT / "results" / "generated" / "mechanism-inputs"
CORPORA = {
    "siftsmall": ("data/siftsmall/siftsmall_base.fvecs", "data/siftsmall/siftsmall_query.fvecs", 10_000, 100),
    "sift": ("data/sift/sift_base.fvecs", "data/sift/sift_query.fvecs", 100_000, 200),
    "glove": ("data/glove/glove_base.fvecs", "data/glove/glove_query.fvecs", 59_500, 200),
    "gist": ("data/gist/gist_base.fvecs", "data/gist/gist_query.fvecs", 100_000, 200),
}


def run(argv: list[str], output: Path, env: dict[str, str], scratch: str = "") -> None:
    print(f"[mechanism] generating {output.name}", flush=True)
    completed = subprocess.run(argv, cwd=ROOT, env=env, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"{' '.join(argv)} failed:\n{completed.stderr[-2000:]}")
    # Normalize before the log is committed, exactly as reproduce.py does for the
    # stdout it captures. This tier writes its own logs rather than going through
    # record_bundle, so without this the absolute checkout path -- and the randomly
    # named scratch directory the binary echoes back -- are baked into 36 tracked
    # evidence files. scripts/test_bundle_hygiene.py fails the build on a regression.
    output.write_text(
        evidence_paths.redact(
            completed.stdout, ROOT, ((scratch, evidence_paths.SCRATCH),) if scratch else ()
        ),
        encoding="utf-8",
    )


def main() -> int:
    if not BIN.is_file():
        raise FileNotFoundError(f"release binary is missing: {BIN}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("ULTRAVEC_TRELLIS_MEM", "12")
    # This tier runs `bench` at its own (smaller) query counts purely to obtain
    # diagnostics. `bench` also writes results/flat-<corpus>-<metric>.md as a
    # side effect, keyed only by dataset + metric, which would overwrite the
    # canonical 1,000-query files the manuscript is validated against. Send those
    # side-effect summaries to a scratch directory; this tier's real outputs are
    # the per-corpus logs written by run() above.
    scratch = tempfile.mkdtemp(prefix="ultravec-mechanism-results-")
    env["ULTRAVEC_RESULTS_DIR"] = scratch
    for corpus, (base, query, maximum, query_max) in CORPORA.items():
        for bits in (1, 2, 3, 4):
            common = [
                "--dataset", base, "--query-file", query, "--query-max", str(query_max),
                "--max", str(maximum), "--bits", str(bits), "--seed", "42",
            ]
            run([str(BIN), "diag", *common], OUTPUT / f"diag-{corpus}-b{bits}.txt", env, scratch)
            run(
                [str(BIN), "bench", *common, "--metric", "ip", "--sota3"],
                OUTPUT / f"recall-{corpus}-b{bits}.txt",
                env,
                scratch,
            )
    base, query, maximum, query_max = CORPORA["sift"]
    common = [
        "--dataset", base, "--query-file", query, "--query-max", str(query_max),
        "--max", str(maximum), "--bits", "2", "--seed", "42",
    ]
    for label, biased in (("unbiased", False), ("biased", True)):
        control_env = dict(env)
        if biased:
            control_env["ULTRAVEC_BIASED_ESTIMATOR"] = "1"
        run([str(BIN), "diag", *common], OUTPUT / f"control_{label}.txt", control_env, scratch)
        run(
            [str(BIN), "bench", *common, "--metric", "ip", "--sota3"],
            OUTPUT / f"control_{label}_recall.txt",
            control_env,
            scratch,
        )
    print(f"mechanism inputs written to {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
