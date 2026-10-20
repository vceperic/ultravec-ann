#!/usr/bin/env python3
"""Verify every pinned source and derived public-data object, then write the full-tier gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SOURCE_MANIFEST = ROOT / "datasets" / "manifest.toml"
DERIVED_MANIFEST = ROOT / "datasets" / "derived.toml"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def check(path: Path, size: int, wanted: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != size:
        raise ValueError(f"{path}: expected {size} bytes, got {path.stat().st_size}")
    actual = digest(path)
    if actual != wanted:
        raise ValueError(f"{path}: expected SHA-256 {wanted}, got {actual}")


def main() -> int:
    with SOURCE_MANIFEST.open("rb") as handle:
        sources = tomllib.load(handle)
    with DERIVED_MANIFEST.open("rb") as handle:
        derived = tomllib.load(handle)
    source_ids = {row["id"] for row in sources["dataset"]}
    for row in sources["dataset"]:
        expected = row.get("expected", [])
        if len(expected) != 1:
            raise ValueError(f"{row['id']}: expected exactly one source object")
        check(DATA / expected[0], int(row["bytes"]), row["sha256"])
    for row in derived["file"]:
        unknown = set(row["sources"]) - source_ids
        if unknown:
            raise ValueError(f"{row['path']}: unknown sources {sorted(unknown)}")
        check(DATA / row["path"], int(row["bytes"]), row["sha256"])
    record = {
        "schema_version": 1,
        "source_manifest_sha256": digest(SOURCE_MANIFEST),
        "derived_manifest_sha256": digest(DERIVED_MANIFEST),
        "sources_verified": len(sources["dataset"]),
        "derived_files_verified": len(derived["file"]),
    }
    gate = DATA / ".full-inputs-verified"
    gate.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"verified {record['sources_verified']} sources and {record['derived_files_verified']} derived files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
