#!/usr/bin/env python3
"""Convert the three pinned DBpedia-OpenAI parquet shards to Texmex fvecs.

The downloader and ``datasets/manifest.toml`` pin Hugging Face snapshot
af9b8869cc2d8debbd254d77737865bb09a2067f.  This converter performs no network
access and deterministically uses the first 100,000 rows as base and the next
1,000 as held-out queries.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "train-00000-of-00026-3c7b99d1c7eda36e.parquet",
    "train-00001-of-00026-2b24035a6390fdcb.parquet",
    "train-00002-of-00026-b05ce48965853dad.parquet",
)
DIM = 1536


def write_fvecs(path: Path, values: np.ndarray) -> None:
    values = np.ascontiguousarray(values, dtype="<f4")
    packed = np.empty((len(values), values.shape[1] + 1), dtype="<f4")
    packed.view("<i4")[:, 0] = values.shape[1]
    packed[:, 1:] = values
    path.parent.mkdir(parents=True, exist_ok=True)
    packed.tofile(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=int, default=100_000)
    parser.add_argument("--queries", type=int, default=1_000)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "dbpedia")
    args = parser.parse_args()
    needed = args.base + args.queries
    chunks: list[np.ndarray] = []
    count = 0
    for filename in SOURCES:
        source = ROOT / "data" / "source" / filename
        table = pq.read_table(source, columns=["openai"])
        column = table.column("openai").combine_chunks()
        flat = column.flatten().to_numpy(zero_copy_only=False)
        values = np.asarray(flat, dtype=np.float32).reshape(-1, DIM)
        chunks.append(values)
        count += len(values)
        if count >= needed:
            break
    values = np.concatenate(chunks, axis=0)[:needed]
    if len(values) != needed or not np.isfinite(values).all():
        raise ValueError(f"expected {needed} finite vectors, got {len(values)}")
    write_fvecs(args.output / "dbpedia_base.fvecs", values[: args.base])
    write_fvecs(args.output / "dbpedia_query.fvecs", values[args.base :])
    print(f"DBpedia-OpenAI: base=({args.base}, {DIM}), query=({args.queries}, {DIM})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
