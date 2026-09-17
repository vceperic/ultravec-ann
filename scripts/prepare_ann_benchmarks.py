#!/usr/bin/env python3
"""Convert pinned ANN-Benchmarks SIFT/GIST HDF5 sources to Texmex fvecs.

The conversion is chunked so the 3.8 GiB GIST source never needs to be held in
memory.  SIFTsmall is a declared deterministic view: the first 10,000 SIFT base
vectors and first 100 published test queries.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def write_fvecs(source: h5py.Dataset, output: Path, limit: int | None = None) -> tuple[int, int]:
    total = len(source) if limit is None else min(len(source), limit)
    if len(source.shape) != 2:
        raise ValueError(f"{source.name} is not a matrix: {source.shape}")
    dim = int(source.shape[1])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        for start in range(0, total, 4096):
            values = np.asarray(source[start : min(total, start + 4096)], dtype="<f4")
            packed = np.empty((len(values), dim + 1), dtype="<f4")
            packed.view("<i4")[:, 0] = dim
            packed[:, 1:] = values
            packed.tofile(handle)
    return total, dim


def convert(name: str) -> None:
    source = ROOT / "data" / "source" / f"{name}-{'128' if name == 'sift' else '960'}-euclidean.hdf5"
    output = ROOT / "data" / name
    with h5py.File(source, "r") as bundle:
        if bundle.attrs.get("distance") != "euclidean":
            raise ValueError(f"unexpected distance metadata in {source}")
        base_shape = write_fvecs(bundle["train"], output / f"{name}_base.fvecs")
        query_shape = write_fvecs(bundle["test"], output / f"{name}_query.fvecs")
        print(f"{name}: base={base_shape}, query={query_shape}")
        if name == "sift":
            small = ROOT / "data" / "siftsmall"
            small_base = write_fvecs(bundle["train"], small / "siftsmall_base.fvecs", 10_000)
            small_query = write_fvecs(bundle["test"], small / "siftsmall_query.fvecs", 100)
            print(f"siftsmall derived view: base={small_base}, query={small_query}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("sift", "gist", "all"), default="all", nargs="?")
    args = parser.parse_args()
    for name in (("sift", "gist") if args.dataset == "all" else (args.dataset,)):
        convert(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
