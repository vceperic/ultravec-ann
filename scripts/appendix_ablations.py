#!/usr/bin/env python3
"""Journal appendix ablations: isolate each trellis design choice and measure
Recall@10 against fp32 cosine gold on SIFT-100k. Only database vectors are
compressed; queries remain full precision in every arm.
  1. computed inverse-CDF code vs byte-sum 1MAD   (ULTRAVEC_TRELLIS_CODE=1mad)
  2. free-start vs fixed-start                     (ULTRAVEC_TRELLIS_FIXEDSTART=1)
  2b. tail-biting vs free start                    (ULTRAVEC_TRELLIS_TAILBITE=1)
  3. beam-width sweep at M=12, and 3b the same at high MEM (ULTRAVEC_TRELLIS_BEAM)
  4. per-bit x per-M recall table
  5. rotation-round sweep                       (ULTRAVEC_ROTATION_ROUNDS)
  6. anisotropic branch-metric sweep            (ULTRAVEC_TRELLIS_ANISO)
Run from the artifact root with the release binary built.
"""
import os, re, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from vector_io import rd, recon, norm

BIN = ROOT / "target" / "release" / "ultravec"
DD = Path(os.environ.get("ULTRAVEC_ERAB_DIR", ROOT / "data" / "erab"))
TEMP = Path(tempfile.mkdtemp(prefix="ultravec-ablations-"))


DBx = rd(f"{DD}/base.fvecs")
Qx = rd(f"{DD}/query.fvecs")
DB_UNIT, Q_UNIT = norm(DBx), norm(Qx)
gold = [np.argpartition(-(Q_UNIT[i] @ DB_UNIT.T), 9)[:10] for i in range(len(Q_UNIT))]


def geom(Rdb):
    DBn = norm(Rdb)
    got = [np.argpartition(-(Q_UNIT[i] @ DBn.T), 9)[:10] for i in range(len(Q_UNIT))]
    return float(np.mean([len(set(got[i].tolist()) & set(gold[i].tolist())) / 10 for i in range(len(got))]))


def recon_env(bits, env_extra, mem=12):
    env = dict(os.environ, ULTRAVEC_TRELLIS_MEM=str(mem), **env_extra)
    out_db = TEMP / "db.fvecs"
    subprocess.run([BIN, "recon", "--backend", "trellis", "--dataset", f"{DD}/base.fvecs",
                    "--bits", str(bits), "--out", out_db], check=True, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return geom(rd(out_db))


print(f"# Journal appendix ablations — SIFT-100k ({len(DBx)} db / {len(Qx)} q), Recall@10 vs fp32 cosine\n")

print("## 1. Computed inverse-CDF code vs byte-sum 1MAD (the codebook lever), M=12")
for b in (2, 3, 4):
    g_comp = recon_env(b, {})
    g_mad = recon_env(b, {"ULTRAVEC_TRELLIS_CODE": "1mad"})
    print(f"  {b}-bit: computed {g_comp:.4f}  vs  1MAD {g_mad:.4f}   (computed - 1MAD = {g_comp-g_mad:+.4f})")

print("\n## 2. Free-start vs fixed-start (the free-start lever), M=12")
for b in (2, 3, 4):
    g_free = recon_env(b, {})
    g_fix = recon_env(b, {"ULTRAVEC_TRELLIS_FIXEDSTART": "1"})
    print(f"  {b}-bit: free {g_free:.4f}  vs  fixed {g_fix:.4f}   (free - fixed = {g_free-g_fix:+.4f})")

print("\n## 2b. Tail-biting vs free start (the start-state byte), M=12")
# Tail-biting pins the path to end in its start state, so the start is the code
# stream's own last M bits and the record drops the ceil(M/8)-byte start field.
# It pins the last ceil(M/b) emissions, which on SIFT (D=128) is 6% of the vector
# against a 5% byte saving; on GIST and DBpedia the same two bytes cost a fraction
# of a point. SIFT is reported here because it is the WORST case for the lever.
for b in (2, 3, 4):
    g_free = recon_env(b, {})
    g_tail = recon_env(b, {"ULTRAVEC_TRELLIS_TAILBITE": "1", "ULTRAVEC_TRELLIS_TAILBITE_K": "1"})
    print(f"  {b}-bit: free {g_free:.4f}  vs  tail-biting {g_tail:.4f}   (tail - free = {g_tail-g_free:+.4f})")

print("\n## 3. Beam-width sweep at M=12, 2-bit (exact vs beam-search encode)")
for beam in (0, 4, 8, 16):
    g = recon_env(2, {} if beam == 0 else {"ULTRAVEC_TRELLIS_BEAM": str(beam)})
    print(f"  beam={'exact' if beam==0 else beam}: {g:.4f}")

# Beam encode costs O(D*W*2^b) and is independent of 2^M, so state memories beyond
# the exact-search range are computationally reachable only this way. The manuscript
# argues from this grid that the beam, not the memory, is what binds at these widths;
# it is measured here rather than asserted. The `W=<w>  M=<m>:` shape is deliberate --
# it collides with neither the `beam=<w>:` sweep above nor the three-column M grid below.
print("\n## 3b. Beam width at high state memory, 2-bit (beam cost is independent of 2^M)")
for width in (64, 128):
    for m in (14, 16, 18, 20, 22):
        g = recon_env(2, {"ULTRAVEC_TRELLIS_BEAM": str(width)}, mem=m)
        print(f"  W={width}  M={m}: {g:.4f}")

print("\n## 4. Per-bit x per-M recall table (the fidelity knob)")
print(f"  {'M':>3} " + " ".join(f"{b}-bit".rjust(8) for b in (2, 3, 4)))
# The ladder runs through M=14. The start state fits in two bytes for M=9..16,
# so the M=10, 12, and 14 rows have the same serialized record size.
for m in (4, 6, 8, 10, 12, 14):
    row = [recon_env(b, {}, mem=m) for b in (2, 3, 4)]
    print(f"  {m:>3} " + " ".join(f"{g:8.4f}" for g in row))

print("\n## 5. Rotation rounds at M=12, 2-bit (shared preprocessing, all codecs)")
for rounds in (1, 2, 3, 4):
    g = recon_env(2, {"ULTRAVEC_ROTATION_ROUNDS": str(rounds)})
    print(f"  rounds={rounds}: {g:.4f}")

# The section header says "all codecs" but the loop above measures only the trellis,
# via `recon`. The manuscript nevertheless claims the extra mixing helps the
# comparators MORE than it helps us -- PVQ +5.5, E8 +2.5 against the trellis's +0.4,
# narrowing the margin over the field from 5.7 to 5.1 -- which is the argument that the
# three-round default was adopted against our own interest. That claim had no evidence
# behind it. This measures it: same rounds sweep, but through `bench`, which runs the
# whole field and reports each codec's own recall.
print("\n## 5b. Rotation rounds per codec at M=12, 2-bit (the shared-preprocessing claim)")
_ROT_ROW = re.compile(r"^\|\s*([a-z0-9_]+)\s*\|\s*(\d)\s*\|\s*[\d.]+\s*\|\s*([\d.]+)\s*\|")
_per_round: dict[int, dict[str, float]] = {}
for rounds in (1, 2, 3, 4):
    env = dict(os.environ, ULTRAVEC_TRELLIS_MEM="12", ULTRAVEC_ROTATION_ROUNDS=str(rounds))
    completed = subprocess.run(
        [str(BIN), "bench", "--dataset", f"{DD}/base.fvecs", "--query-file", f"{DD}/query.fvecs",
         "--query-max", "1000", "--max", "100000", "--bits", "2", "--seed", "42", "--sota3"],
        capture_output=True, text=True, env=env,
    )
    if completed.returncode:
        raise RuntimeError(f"bench failed (rounds {rounds}):\n{completed.stderr[-2000:]}")
    row = {}
    for line in completed.stdout.splitlines():
        m = _ROT_ROW.match(line)
        if m and int(m.group(2)) == 2:
            row[m.group(1)] = float(m.group(3))
    if not row:
        raise RuntimeError(f"bench produced no parsable rows (rounds {rounds})")
    _per_round[rounds] = row
_codecs = sorted(set().union(*(set(r) for r in _per_round.values())))
print("| codec | " + " | ".join(f"r={r}" for r in (1, 2, 3, 4)) + " | r=3 minus r=1 |")
print("|---|" + "---|" * 5)
for c in _codecs:
    vals = [_per_round[r].get(c) for r in (1, 2, 3, 4)]
    cells = " | ".join("---" if v is None else f"{v:.4f}" for v in vals)
    delta = "---" if (vals[0] is None or vals[2] is None) else f"{(vals[2] - vals[0]) * 100:+.2f}"
    print(f"| {c} | {cells} | {delta} |")
# The margin the claim is really about: trellis minus the best comparator, per round.
print()
print("| rounds | trellis | best other | margin pp |")
print("|---|---|---|---|")
for r in (1, 2, 3, 4):
    row = _per_round[r]
    if "trellis" not in row:
        continue
    others = {k: v for k, v in row.items() if k != "trellis"}
    if not others:
        continue
    best = max(others.values())
    print(f"| {r} | {row['trellis']:.4f} | {best:.4f} | {(row['trellis'] - best) * 100:+.2f} |")

print("\n## 6. Anisotropic branch metric at M=12, 2-bit (w_i = 1 + lambda*x_i^2)")
for lam in (0.0, 0.5, 1.0, 2.0):
    g = recon_env(2, {"ULTRAVEC_TRELLIS_ANISO": str(lam)})
    print(f"  lambda={lam}: {g:.4f}")
