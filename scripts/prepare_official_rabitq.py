#!/usr/bin/env python3
"""Build and run the pinned upstream RaBitQ/E-RaBitQ IVF estimator on SIFT."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "baselines" / "RaBitQ-Library"
REPRO_SOURCE = ROOT / "data" / "baselines" / "RaBitQ-Library-reproducible"
BUILD = ROOT / "data" / "baselines" / "RaBitQ-Library-reproducible-build"
ERAB = ROOT / "data" / "erab"
MANIFEST = ROOT / "baselines" / "manifest.toml"


# Retained logs are shipped in the artifact, so every machine-specific path is
# normalized at the point of write. `record_bundle` already rewrites the artifact
# root to "."; the second-run recheck happens in a randomly named temporary
# directory that it cannot know about, so this script registers that one itself.
# Numerical output and provenance manifests are unaffected.
REDACTIONS: list[tuple[str, str]] = [(str(ROOT), ".")]


def redact(text: str) -> str:
    for literal, placeholder in REDACTIONS:
        text = text.replace(literal, placeholder)
    return text


def run(argv: list[str]) -> None:
    print("+ " + redact(" ".join(argv)), flush=True)
    subprocess.run(argv, cwd=ROOT, check=True)


def output(argv: list[str], cwd: Path) -> str:
    return subprocess.run(
        argv, cwd=cwd, check=True, text=True, capture_output=True
    ).stdout.rstrip("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rabitq_record() -> dict:
    with MANIFEST.open("rb") as handle:
        records = tomllib.load(handle)["baseline"]
    return next(record for record in records if record["id"] == "rabitq-library")


def prepare_reproducible_source(record: dict) -> Path:
    revision = record["revision"]
    patch = ROOT / record["reproducibility_patch"]
    if REPRO_SOURCE.exists():
        actual = output(["git", "rev-parse", "HEAD"], REPRO_SOURCE)
        status = output(["git", "status", "--porcelain"], REPRO_SOURCE).splitlines()
        expected_status = [" M include/rabitqlib/utils/rotator.hpp"]
        reverse_check = subprocess.run(
            ["git", "apply", "--reverse", "--check", str(patch)],
            cwd=REPRO_SOURCE,
            text=True,
            capture_output=True,
        )
        if actual != revision or status != expected_status or reverse_check.returncode:
            raise RuntimeError(
                f"unexpected state in {REPRO_SOURCE.relative_to(ROOT)}; "
                "remove that ignored worktree deliberately and rerun"
            )
        return REPRO_SOURCE

    run([
        "git", "-C", str(SOURCE), "worktree", "add", "--detach",
        str(REPRO_SOURCE), revision,
    ])
    run(["git", "-C", str(REPRO_SOURCE), "apply", str(patch)])
    print(
        "upstream RaBitQ/E-RaBitQ source: pinned revision "
        f"{revision}, artifact rotation seed 42",
        flush=True,
    )
    return REPRO_SOURCE


def produce_outputs(indexer: Path, adapter: Path, destination: Path) -> dict[int, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    outputs: dict[int, Path] = {}
    for bits in range(1, 5):
        index = destination / f"erabIVF_n1_b{bits}.index"
        ids = destination / f"erabIVF_n1_b{bits}.ivecs"
        run([
            str(indexer), str(ERAB / "base.fvecs"),
            str(ERAB / "ivfA" / "centroids_1.fvecs"),
            str(ERAB / "ivfA" / "cids_1.ivecs"), str(bits), str(index), "ip", "false",
        ])
        run([str(adapter), str(index), str(ERAB / "query.fvecs"), str(ids), "1", "10", "false"])
        outputs[bits] = ids
    return outputs


def main() -> int:
    run([sys.executable, "scripts/verify_baselines.py"])
    run([sys.executable, "scripts/erab_prep.py"])
    source = prepare_reproducible_source(rabitq_record())
    run(["cmake", "-S", str(source), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release"])
    run(["cmake", "--build", str(BUILD), "--target", "ivf_rabitq_indexing", "-j"])

    adapter = BUILD / "erab_ivf_query10"
    run([
        os.environ.get("CXX", "c++"), "-std=c++17", "-O3", "-march=native", "-fopenmp",
        "-I", str(source / "include"), str(ROOT / "official_harness" / "erab_ivf_query10.cpp"),
        str(BUILD / "librabitq_core.a"), "-lrt", "-o", str(adapter),
    ])

    indexer = source / "bin" / "ivf_rabitq_indexing"
    first = produce_outputs(indexer, adapter, ERAB / "ivfA")
    with tempfile.TemporaryDirectory(prefix="ultravec-rabitq-recheck-") as directory:
        REDACTIONS.append((directory, "<temporary-directory>"))
        second = produce_outputs(indexer, adapter, Path(directory))
        for bits in range(1, 5):
            first_hash = sha256(first[bits])
            second_hash = sha256(second[bits])
            if first_hash != second_hash:
                raise RuntimeError(
                    f"upstream RaBitQ/E-RaBitQ output is nondeterministic at {bits} bits: "
                    f"{first_hash} != {second_hash}"
                )
            method = "RaBitQ" if bits == 1 else "E-RaBitQ"
            print(f"upstream {method} {bits}-bit output verified twice: {first_hash}")
    print("prepared deterministic upstream RaBitQ/E-RaBitQ IP top-10 outputs for 1--4 bits/dimension")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
