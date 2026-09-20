#!/usr/bin/env python3
"""Convert ANN-Benchmarks GloVe HDF5 arrays to portable Texmex fvecs files."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from vector_io import wr

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "source" / "glove-25-angular.hdf5",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "glove")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.input, "r") as source:
        if "train" not in source or "test" not in source:
            raise ValueError(f"{args.input} must contain train and test arrays")
        base = np.asarray(source["train"], dtype=np.float32)
        query = np.asarray(source["test"], dtype=np.float32)
    if base.ndim != 2 or query.ndim != 2 or base.shape[1] != query.shape[1]:
        raise ValueError(f"invalid train/test shapes: {base.shape}, {query.shape}")
    wr(args.output / "glove_base.fvecs", base)
    wr(args.output / "glove_query.fvecs", query)
    print(f"GloVe: wrote {base.shape} base and {query.shape} query vectors to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
