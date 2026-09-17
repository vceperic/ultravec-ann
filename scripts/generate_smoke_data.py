#!/usr/bin/env python3
"""Generate deterministic, non-claim synthetic vectors for the smoke tier."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vector_io import wr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    rng = np.random.default_rng(42)
    base = rng.normal(size=(128, 32)).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    query = base[:16] + rng.normal(scale=0.02, size=(16, 32)).astype(np.float32)
    query /= np.linalg.norm(query, axis=1, keepdims=True)
    destination = ROOT / "data" / "smoke"
    destination.mkdir(parents=True, exist_ok=True)
    wr(destination / "base.fvecs", base)
    wr(destination / "query.fvecs", query)
    print(f"wrote deterministic smoke corpus: {len(base)} base / {len(query)} query / dim {base.shape[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
