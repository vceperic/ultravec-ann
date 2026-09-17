#!/usr/bin/env python3
"""Is the trellis-memory operating point the same at every rate?

The memory sweep that selects M=12 is run at two bits, where returns have flattened:
M=14 buys 0.15 points there for 3.3 times the encode time. That sweep cannot say
whether 12 is also right at one bit, and one bit is where the codec is weakest and
where its fixed overhead is largest -- 6 of a 22-byte record, against 6 of 70 at four
bits.

The start state occupies ceil(M/8) bytes, so memory comes in byte tiers: every M from
9 to 16 costs the same two bytes. Within a tier, more memory is free in record size
and paid for only in encode time. This sweeps that tier at one bit, and includes M=8
-- the top of the one-byte tier -- to price the opposite move of trading state away
to save a byte.

Footing matches the parity and centering studies: normalized SIFT-128, 1,000 queries,
exact-cosine gold, uncentered.

Run: python3 scripts/rate_memory_sweep.py
"""
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DD = Path(os.environ.get("ULTRAVEC_ERAB_DIR", ROOT / "data" / "erab"))
BIN = Path(os.environ.get("ULTRAVEC_BIN", ROOT / "target" / "release" / "ultravec"))
TEMP = Path(tempfile.mkdtemp(prefix="ultravec-rate-mem-"))
K = 10
BITS = 1
DIM = 128
MEMORIES = [8, 12, 14, 16]
REFERENCE = 12


def read_ivecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    dim = raw[0]
    return raw.reshape(-1, dim + 1)[:, 1:]


def read_fvecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    dim = raw[0]
    return np.ascontiguousarray(raw.reshape(-1, dim + 1)[:, 1:].view(np.float32))


def recall(memory: int, queries: np.ndarray, gold: np.ndarray) -> float:
    out = TEMP / f"m{memory}.fvecs"
    env = dict(os.environ)
    env["ULTRAVEC_TRELLIS_MEM"] = str(memory)
    env["ULTRAVEC_CENTER"] = "0"
    proc = subprocess.run(
        [str(BIN), "recon", "--backend", "trellis", "--dataset", str(DD / "base.fvecs"),
         "--bits", str(BITS), "--out", str(out)],
        capture_output=True, text=True, env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"recon failed (M={memory}):\n{proc.stderr[-600:]}")
    database = read_fvecs(out)
    database /= np.linalg.norm(database, axis=1, keepdims=True) + 1e-12
    hits = 0.0
    for start in range(0, len(queries), 256):
        got = np.argpartition(-(queries[start:start + 256] @ database.T), K - 1, axis=1)[:, :K]
        for i, row in enumerate(got):
            hits += len(set(row) & set(gold[start + i])) / K
    return hits / len(queries)


def main() -> None:
    gold = read_ivecs(DD / "gold.ivecs")
    queries = read_fvecs(DD / "query.fvecs")
    queries = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12)

    payload = DIM * BITS // 8
    print(f"Trellis memory at one bit — normalized SIFT-128, {len(gold)} queries, "
          "exact-cosine gold, uncentered")
    print(f"record = {payload} B payload + ceil(M/8) B start state + 4 B rescale")
    print()
    measured = {m: recall(m, queries, gold) for m in MEMORIES}
    reference = measured[REFERENCE]
    print("| M | state B | record B | R@10 | vs M=12 pp |")
    print("|---|---|---|---|---|")
    for memory in MEMORIES:
        state = -(-memory // 8)
        print(f"| {memory} | {state} | {payload + state + 4} | {measured[memory]:.4f} "
              f"| {(measured[memory] - reference) * 100:+.2f} |")
    print()
    print("DONE rate_memory_sweep")


if __name__ == "__main__":
    main()
