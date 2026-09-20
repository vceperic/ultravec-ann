#!/usr/bin/env python3
"""Multi-seed CI for the cold-start calibration study. The random element is the PQ
calibration-set DRAW: for each of 6 seeds, run `coldstart --seed S` (which reseeds the
K-sized PQ calibration subsample), parse PQ recall@10 at each K, and report mean ± 95% CI
per K. The sweep locates the crossover between fitted PQ and the calibration-free
trellis on a 50,000-vector SIFT base.

Run: python3 scripts/coldstart_seed_ci.py
"""
import math
import os
import re
import subprocess
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "target" / "release" / "ultravec"
SIFT = ROOT / "data" / "sift"
SEEDS = [42, 43, 44, 45, 46, 47]
KS = ["32", "64", "128", "256", "512", "1000", "2000", "5000", "20000", "full"]
CALIB = "32,64,128,256,512,1000,2000,5000,20000,0"   # 0 = full; matches tab:coldstart's K grid

def run(seed):
    env = dict(os.environ, ULTRAVEC_TRELLIS_MEM=os.environ.get("ULTRAVEC_TRELLIS_MEM", "12"))
    completed = subprocess.run(
        [BIN, "coldstart", "--dataset", f"{SIFT}/sift_base.fvecs",
         "--query-file", f"{SIFT}/sift_query.fvecs", "--query-max", "200",
         "--max", "50000", "--bits", "2", "--calib", CALIB, "--seed", str(seed)],
        capture_output=True, text=True, env=env)
    if completed.returncode:
        raise RuntimeError(
            f"coldstart failed for seed {seed} ({completed.returncode}):\n{completed.stderr[-2000:]}"
        )
    out = completed.stdout
    # rows like: | 64 | 0.578 | -0.059 | ... ; trellis flat printed in the header line
    pq = {}
    for line in out.splitlines():
        m = re.match(r"\|\s*(\d+|full)\s*\|\s*([0-9.]+)\s*\|", line)
        if m:
            pq[m.group(1)] = float(m.group(2))
    tm = re.search(r"trellis\s+([0-9.]+)", out)
    missing = sorted(set(KS) - set(pq))
    if missing or tm is None:
        raise RuntimeError(f"incomplete coldstart output for seed {seed}; missing rows={missing}, trellis={tm}")
    trellis = float(tm.group(1))
    return pq, trellis

rows = {}
trellis_vals = []
for s in SEEDS:
    pq, tr = run(s)
    trellis_vals.append(tr)
    for k, v in pq.items():
        rows.setdefault(k, []).append(v)
    print(f"seed {s}: trellis {tr:.3f} | " + " ".join(f"{k}={pq.get(k,float('nan')):.3f}" for k in KS), flush=True)

tr_flat = st.mean(trellis_vals)
print(f"\n# trellis flat (oblivious, calibration-free): {tr_flat:.3f} "
      f"(sd {st.pstdev(trellis_vals):.4f} across seeds — should be ~0, deterministic)")
print("\n## PQ recall@10 vs K — mean ± 95%CI over 6 calibration seeds, vs flat trellis")
print("| K | PQ mean | 95% CI | PQ - trellis |")
print("|---|---|---|---|")
for k in KS:
    g = rows.get(k, [])
    if not g:
        continue
    m = st.mean(g); sd = st.stdev(g) if len(g) > 1 else 0.0
    se = sd / math.sqrt(len(g)); hw = 2.571 * se
    print(f"| {k} | {m:.3f} | [{m-hw:.3f}, {m+hw:.3f}] | {m-tr_flat:+.3f} |")
print("\nDONE coldstart_seed_ci")
