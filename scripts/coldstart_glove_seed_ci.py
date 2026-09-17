#!/usr/bin/env python3
"""Multi-seed CI for the public GloVe-25-angular cold-start arm.

Runs `coldstart --seed S` for six seeds, each reseeding the PQ calibration draw,
and reports mean ± 95% CI on PQ recall@10 per K versus the flat trellis.

Run: python3 scripts/coldstart_glove_seed_ci.py
"""
import math
import os
import re
import subprocess
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "target" / "release" / "ultravec"
G = ROOT / "data" / "glove"
SEEDS = [42, 43, 44, 45, 46, 47]
KS = ["32", "64", "128", "256", "512", "1000", "2000", "5000", "20000", "full"]
CALIB = "32,64,128,256,512,1000,2000,5000,20000,0"

def run(seed):
    env = dict(os.environ, ULTRAVEC_TRELLIS_MEM=os.environ.get("ULTRAVEC_TRELLIS_MEM", "12"))
    completed = subprocess.run(
        [BIN, "coldstart", "--dataset", f"{G}/glove_base.fvecs",
         "--query-file", f"{G}/glove_query.fvecs", "--query-max", "200",
         "--max", "59500", "--bits", "2", "--calib", CALIB, "--seed", str(seed)],
        capture_output=True, text=True, env=env)
    if completed.returncode:
        raise RuntimeError(
            f"coldstart failed for seed {seed} ({completed.returncode}):\n{completed.stderr[-2000:]}"
        )
    out = completed.stdout
    pq = {}
    for line in out.splitlines():
        m = re.match(r"\|\s*(\d+|full)\s*\|\s*([0-9.]+)\s*\|", line)
        if m:
            pq[m.group(1)] = float(m.group(2))
    tm = re.search(r"trellis\s+([0-9.]+)", out)
    missing = sorted(set(KS) - set(pq))
    if missing or tm is None:
        raise RuntimeError(f"incomplete coldstart output for seed {seed}; missing rows={missing}, trellis={tm}")
    return pq, float(tm.group(1))

rows = {}; tvals = []
for s in SEEDS:
    pq, tr = run(s); tvals.append(tr)
    for k, v in pq.items():
        rows.setdefault(k, []).append(v)
    print(f"seed {s}: trellis {tr:.3f} | " + " ".join(f"{k}={pq.get(k,float('nan')):.3f}" for k in KS), flush=True)

tflat = st.mean(tvals)
print(f"\n# trellis flat: {tflat:.3f} (sd {st.pstdev(tvals):.4f})")
print("\n## PQ recall@10 vs K — mean ± 95%CI over 6 calibration seeds, vs flat trellis")
print("| K | PQ mean | 95% CI | PQ - trellis |")
print("|---|---|---|---|")
never = True
for k in KS:
    g = rows.get(k, [])
    if not g:
        continue
    m = st.mean(g); sd = st.stdev(g) if len(g) > 1 else 0.0
    se = sd / math.sqrt(len(g)); hw = 2.571 * se
    if m > tflat:
        never = False
    print(f"| {k} | {m:.3f} | [{m-hw:.3f}, {m+hw:.3f}] | {m-tflat:+.3f} |")
print(f"\n# PQ NEVER catches the trellis at any K: {never}")
print("DONE coldstart_glove_seed_ci")
