#!/usr/bin/env python3
"""Verify pinned external baseline revisions and executable distributions."""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "baselines" / "manifest.toml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, text=True, capture_output=True
    ).stdout.strip()


def verify(allow_missing_wheels: bool) -> list[str]:
    with MANIFEST.open("rb") as handle:
        records = tomllib.load(handle)["baseline"]
    errors: list[str] = []
    for record in records:
        ident = record["id"]
        path = ROOT / record["local_path"]
        if record["kind"] == "git":
            if not (path / ".git").is_dir():
                errors.append(f"{ident}: missing checkout {path.relative_to(ROOT)}")
                continue
            try:
                actual = git_output(path, "rev-parse", "HEAD")
                dirty = git_output(path, "status", "--porcelain")
            except subprocess.CalledProcessError as exc:
                errors.append(f"{ident}: git verification failed ({exc.returncode})")
                continue
            if actual != record["revision"]:
                errors.append(f"{ident}: expected {record['revision']}, found {actual}")
            if dirty:
                errors.append(f"{ident}: checkout has uncommitted changes")
            license_path = path / "LICENSE"
            if not license_path.is_file() or sha256(license_path) != record["license_file_sha256"]:
                errors.append(f"{ident}: upstream LICENSE hash mismatch")
            patch_name = record.get("reproducibility_patch")
            patch_expected = record.get("reproducibility_patch_sha256")
            if bool(patch_name) != bool(patch_expected):
                errors.append(f"{ident}: reproducibility patch path/hash must be specified together")
            elif patch_name:
                patch_path = ROOT / patch_name
                if not patch_path.is_file():
                    errors.append(f"{ident}: missing reproducibility patch {patch_name}")
                elif sha256(patch_path) != patch_expected:
                    errors.append(f"{ident}: reproducibility patch hash mismatch")
                elif subprocess.run(
                    ["git", "-C", str(path), "apply", "--check", str(patch_path)],
                    text=True,
                    capture_output=True,
                ).returncode:
                    errors.append(f"{ident}: reproducibility patch does not apply to pinned revision")
        elif record["kind"] == "python-wheel":
            if not path.is_file():
                if not allow_missing_wheels:
                    errors.append(f"{ident}: missing wheel {path.relative_to(ROOT)}")
            elif sha256(path) != record["wheel_sha256"]:
                errors.append(f"{ident}: wheel hash mismatch")
            try:
                actual_version = metadata.version(ident)
            except metadata.PackageNotFoundError:
                errors.append(f"{ident}: distribution is not installed in this Python environment")
            else:
                if actual_version != record["version"]:
                    errors.append(f"{ident}: expected version {record['version']}, found {actual_version}")
        else:
            errors.append(f"{ident}: unsupported baseline kind {record['kind']!r}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-missing-wheels",
        action="store_true",
        help="permit an installed, correctly versioned distribution when its cached wheel is absent",
    )
    args = parser.parse_args()
    try:
        errors = verify(args.allow_missing_wheels)
    except (OSError, KeyError, tomllib.TOMLDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("verified 3 pinned official baselines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
