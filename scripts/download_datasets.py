#!/usr/bin/env python3
"""Checksum-mandatory dataset downloader for the ANN artifact."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile
import tomllib
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "datasets" / "manifest.toml"
DATA = ROOT / "data"
STAMPS = DATA / ".dataset-sha256"
PLACEHOLDER = "REQUIRED_BEFORE_RELEASE"


class DatasetError(RuntimeError):
    pass


def load() -> dict[str, dict]:
    with MANIFEST.open("rb") as handle:
        manifest = tomllib.load(handle)
    if manifest.get("schema_version") != 1:
        raise DatasetError("unsupported dataset manifest schema")
    rows = manifest.get("dataset", [])
    datasets = {row["id"]: row for row in rows}
    if len(datasets) != len(rows):
        raise DatasetError("duplicate dataset id")
    return datasets


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def validate_row(row: dict) -> None:
    url = str(row.get("url", ""))
    wanted = str(row.get("sha256", ""))
    if not row.get("enabled"):
        raise DatasetError(f"{row['id']}: download disabled pending checksum/license review")
    if urllib.parse.urlparse(url).scheme != "https":
        raise DatasetError(f"{row['id']}: only HTTPS downloads are allowed")
    if wanted == PLACEHOLDER or len(wanted) != 64 or any(c not in "0123456789abcdef" for c in wanted.lower()):
        raise DatasetError(f"{row['id']}: valid SHA-256 is required before download")
    if row.get("archive") not in {"none", "tar.gz"}:
        raise DatasetError(f"{row['id']}: archive mode requires a curated downloader")


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts or member.issym() or member.islnk() or member.isdev():
                raise DatasetError(f"unsafe archive member: {member.name}")
        bundle.extractall(destination, filter="data")


def verify(row: dict) -> None:
    validate_row(row)
    missing = [rel for rel in row.get("expected", []) if not (DATA / rel).is_file()]
    if missing:
        raise DatasetError(f"{row['id']}: missing staged files: {', '.join(missing)}")
    stamp = STAMPS / row["id"]
    if not stamp.is_file() or stamp.read_text(encoding="ascii").strip() != row["sha256"].lower():
        raise DatasetError(f"{row['id']}: checksum provenance stamp is absent or stale; fetch through this tool")
    expected = row.get("expected", [])
    if len(expected) == 1:
        source = DATA / expected[0]
        wanted_bytes = row.get("bytes")
        if wanted_bytes is not None and source.stat().st_size != int(wanted_bytes):
            raise DatasetError(
                f"{row['id']}: byte-size mismatch; expected {wanted_bytes}, got {source.stat().st_size}"
            )
        actual = digest(source)
        if actual != row["sha256"].lower():
            raise DatasetError(f"{row['id']}: staged source checksum mismatch; expected {row['sha256']}, got {actual}")
    print(f"{row['id']}: staged files and source archive checksum are verified")


def adopt(row: dict) -> None:
    """Verify an independently downloaded source object and create its provenance stamp."""
    validate_row(row)
    expected = row.get("expected", [])
    if len(expected) != 1:
        raise DatasetError(f"{row['id']}: adopt requires exactly one expected source object")
    source = DATA / expected[0]
    if not source.is_file():
        raise DatasetError(f"{row['id']}: source object is missing: {source}")
    wanted_bytes = row.get("bytes")
    if wanted_bytes is not None and source.stat().st_size != int(wanted_bytes):
        raise DatasetError(
            f"{row['id']}: byte-size mismatch; expected {wanted_bytes}, got {source.stat().st_size}"
        )
    actual = digest(source)
    if actual != row["sha256"].lower():
        raise DatasetError(f"{row['id']}: checksum mismatch; expected {row['sha256']}, got {actual}")
    STAMPS.mkdir(parents=True, exist_ok=True)
    (STAMPS / row["id"]).write_text(actual + "\n", encoding="ascii")
    verify(row)


def fetch(row: dict) -> None:
    validate_row(row)
    DATA.mkdir(parents=True, exist_ok=True)
    suffix = ".tar.gz" if row["archive"] == "tar.gz" else ".download"
    # Staged inside data/ rather than $TMPDIR. `Path.replace` below is rename(2),
    # which raises EXDEV across filesystems -- the common case for a fresh clone,
    # where /tmp is frequently tmpfs or a separate mount -- and it would fail only
    # after the whole object had been downloaded and verified. Staging here also
    # keeps the 3.8 GB GIST object off tmpfs.
    with tempfile.TemporaryDirectory(prefix="ultravec-download-", dir=DATA) as temp:
        incoming = Path(temp) / f"{row['id']}{suffix}"
        request = urllib.request.Request(row["url"], headers={"User-Agent": "ultravec-artifact/0.1"})
        with urllib.request.urlopen(request, timeout=60) as source, incoming.open("wb") as target:
            shutil.copyfileobj(source, target)
        actual = digest(incoming)
        if actual != row["sha256"].lower():
            raise DatasetError(f"{row['id']}: checksum mismatch; expected {row['sha256']}, got {actual}")
        if row["archive"] == "tar.gz":
            safe_extract(incoming, DATA)
        else:
            expected = row.get("expected", [])
            target = DATA / expected[0] if len(expected) == 1 else DATA / Path(urllib.parse.urlparse(row["url"]).path).name
            target.parent.mkdir(parents=True, exist_ok=True)
            incoming.replace(target)
        STAMPS.mkdir(parents=True, exist_ok=True)
        (STAMPS / row["id"]).write_text(actual + "\n", encoding="ascii")
    verify(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "verify", "fetch", "adopt"))
    parser.add_argument("--dataset")
    args = parser.parse_args()
    try:
        datasets = load()
        if args.action == "list":
            for ident, row in datasets.items():
                state = "enabled" if row.get("enabled") else "GATED"
                print(f"{ident:10} {state:7} {row.get('description', '')}")
            return 0
        if not args.dataset or args.dataset not in datasets:
            raise DatasetError("--dataset must name an id shown by the list action")
        if args.action == "verify":
            verify(datasets[args.dataset])
        elif args.action == "adopt":
            adopt(datasets[args.dataset])
        else:
            fetch(datasets[args.dataset])
        return 0
    except (DatasetError, OSError, KeyError, tomllib.TOMLDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
