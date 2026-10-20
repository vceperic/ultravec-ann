#!/usr/bin/env python3
"""Validate artifact configuration and retained claim evidence.

Portable mode is designed for a fresh standalone repository or a source archive:
it validates every retained bundle without requiring the original authoring Git
history or the large public datasets. ``--strict`` additionally requires those
datasets and resolvable authoring commits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / "baselines" / "manifest.toml"
COMPUTATION_PREFIXES = (
    "src/",
    "scripts/",
    "datasets/",
    "baselines/",
    "official_harness/",
)
COMPUTATION_FILES = {
    "Cargo.lock",
    "Cargo.toml",
    "rust-toolchain.toml",
    "requirements.in",
    "requirements.txt",
    "experiments.toml",
}
VALIDATED_STATUS = "validated-clean-evidence"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(argv: list[str]) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ["git", *argv], cwd=ROOT, check=False, capture_output=True
        )
    except OSError:
        return subprocess.CompletedProcess(["git", *argv], 127, b"", b"")


def computation_paths() -> list[Path]:
    """List computation files from Git when possible, otherwise from the archive."""
    listing = git(["ls-files", "-z"])
    paths: set[Path] = set()
    if listing.returncode == 0:
        paths.update(
            Path(raw.decode("utf-8"))
            for raw in listing.stdout.split(b"\0")
            if raw
            if raw.decode("utf-8") in COMPUTATION_FILES
            or raw.decode("utf-8").startswith(COMPUTATION_PREFIXES)
        )

    paths.update(
        Path(name) for name in COMPUTATION_FILES if (ROOT / name).is_file()
    )
    for prefix in COMPUTATION_PREFIXES:
        directory = ROOT / prefix
        if not directory.is_dir():
            continue
        paths.update(
            path.relative_to(ROOT)
            for path in directory.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        )
    return sorted(path for path in paths if (ROOT / path).is_file())


def computation_sha256() -> str:
    paths = computation_paths()
    if not paths:
        raise ValueError("no computation files found")
    digest = hashlib.sha256()
    for relative in paths:
        digest.update(str(relative).encode("utf-8") + b"\0")
        digest.update(bytes.fromhex(sha256(ROOT / relative)))
    return digest.hexdigest()


def computation_sha256_at(commit: str) -> str | None:
    if not commit or git(["cat-file", "-e", f"{commit}^{{commit}}"]).returncode:
        return None
    prefix_result = git(["rev-parse", "--show-prefix"])
    prefix = (
        prefix_result.stdout.decode("utf-8", errors="replace").strip()
        if prefix_result.returncode == 0
        else ""
    )
    attempts = (
        (["ls-tree", "-r", "-z", "--format=%(objectname) %(path)", commit, "--", "."], prefix),
        (["ls-tree", "-r", "-z", "--full-tree", "--format=%(objectname) %(path)", commit], ""),
    )
    entries: list[tuple[str, str]] = []
    for argv, strip in attempts:
        listing = git(argv)
        if listing.returncode:
            continue
        entries = []
        for raw in listing.stdout.decode("utf-8", errors="replace").split("\0"):
            if not raw:
                continue
            blob, _, path = raw.partition(" ")
            relative = path[len(strip):] if strip and path.startswith(strip) else path
            if relative in COMPUTATION_FILES or relative.startswith(COMPUTATION_PREFIXES):
                entries.append((relative, blob))
        if entries:
            break
    if not entries:
        return None

    digest = hashlib.sha256()
    for relative, blob in sorted(entries):
        content = git(["cat-file", "blob", blob])
        if content.returncode:
            return None
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(content.stdout).digest())
    return digest.hexdigest()


def baseline_revisions() -> dict[str, dict[str, str]]:
    with BASELINES.open("rb") as handle:
        records = tomllib.load(handle)["baseline"]
    output: dict[str, dict[str, str]] = {}
    for record in records:
        if record["kind"] == "git":
            output[record["id"]] = {
                "kind": "git",
                "revision": record["revision"],
                "license": record["license"],
            }
            if "reproducibility_patch" in record:
                output[record["id"]]["reproducibility_patch"] = record[
                    "reproducibility_patch"
                ]
                output[record["id"]]["reproducibility_patch_sha256"] = record[
                    "reproducibility_patch_sha256"
                ]
        else:
            output[record["id"]] = {
                "kind": "python-wheel",
                "version": record["version"],
                "wheel_sha256": record["wheel_sha256"],
                "license": record["license"],
            }
    return output


def experiment_steps(config: dict) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for tier in config.get("tiers", {}).values():
        for step in tier.get("steps", []):
            experiment = step.get("experiment")
            if experiment:
                output.setdefault(experiment, step)
    return output


def declared_seed(step: dict) -> int | None:
    if "seed" in step:
        return step["seed"]
    argv = [str(value) for value in step.get("argv", [])]
    if "--seed" in argv and argv.index("--seed") + 1 < len(argv):
        try:
            return int(argv[argv.index("--seed") + 1])
        except ValueError:
            return None
    return None


def validate_schema(config: dict, claims: dict) -> list[str]:
    errors: list[str] = []
    if config.get("schema_version") != 1:
        errors.append("experiments.toml: schema_version must be 1")
    tiers = config.get("tiers")
    if not isinstance(tiers, dict) or not tiers:
        errors.append("experiments.toml: tiers must be a non-empty table")
        tiers = {}

    seen: dict[str, dict] = {}
    for name, tier in tiers.items():
        if not isinstance(tier.get("required_files", []), list):
            errors.append(f"tier {name}: required_files must be a list")
        for step in tier.get("steps", []):
            if not (
                (isinstance(step.get("argv"), list) and step["argv"])
                or step.get("tier")
            ):
                errors.append(f"tier {name}: invalid step {step!r}")
            if "datasets" in step and not isinstance(step["datasets"], list):
                errors.append(f"tier {name}: step datasets must be a list")
            if "depends_on" in step and not isinstance(step["depends_on"], list):
                errors.append(f"tier {name}: step depends_on must be a list")
            if "seed" in step and not isinstance(step["seed"], int):
                errors.append(f"tier {name}: step seed must be an integer")
            experiment = step.get("experiment")
            if experiment:
                previous = seen.setdefault(experiment, step)
                if (
                    declared_seed(previous) != declared_seed(step)
                    or previous.get("argv") != step.get("argv")
                ):
                    errors.append(
                        f"experiment {experiment}: inconsistent duplicate definitions"
                    )

    if claims.get("schema_version") != 1 or not isinstance(claims.get("claim"), list):
        errors.append(
            "results/claims.toml must contain schema_version=1 and [[claim]] entries"
        )
    if claims.get("status") != VALIDATED_STATUS:
        errors.append(
            "results/claims.toml status must be "
            f"{VALIDATED_STATUS!r}, found {claims.get('status')!r}"
        )
    for claim in claims.get("claim", []):
        ident = claim.get("id", "<unnamed>")
        required = claim.get("required_experiments")
        if not claim.get("id") or not isinstance(claim.get("enabled"), bool):
            errors.append(f"invalid claim entry: {claim!r}")
        if not isinstance(required, list):
            errors.append(f"claim {ident}: required_experiments must be a list")
            continue
        for experiment in required:
            step = seen.get(experiment)
            if step is None:
                errors.append(f"claim {ident}: unknown experiment {experiment}")
            elif declared_seed(step) is None:
                errors.append(
                    f"claim {ident}: experiment {experiment} has no controlled seed"
                )

    scann = seen.get("systems-scann-sift1m")
    if scann and "--verify-determinism" not in scann.get("argv", []):
        errors.append("systems-scann-sift1m must enable --verify-determinism")
    return errors


def validate_evidence(
    config: dict,
    claims: dict,
    include_disabled: bool,
    strict: bool,
) -> list[str]:
    errors: list[str] = []
    notes: set[str] = set()
    steps = experiment_steps(config)
    current_computation = computation_sha256()
    current_baselines = baseline_revisions()
    current_lock = sha256(ROOT / "requirements.txt")
    current_dataset_manifest = sha256(ROOT / "datasets" / "manifest.toml")
    current_derived_manifest = sha256(ROOT / "datasets" / "derived.toml")
    current_baseline_manifest = sha256(BASELINES)

    for claim in claims["claim"]:
        ident = claim["id"]
        if not claim["enabled"] and not include_disabled:
            errors.append(f"claim {ident}: disabled pending independent reproduction")
            continue
        for experiment in claim["required_experiments"]:
            manifest = ROOT / "results" / "generated" / experiment / "manifest.json"
            if not manifest.is_file():
                errors.append(f"claim {ident}: missing experiment bundle {experiment}")
                continue
            try:
                record = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(
                    f"claim {ident}: invalid {manifest.relative_to(ROOT)}: {exc}"
                )
                continue

            required = {
                "campaign_start_commit",
                "campaign_start_computation_sha256",
                "campaign_start_dirty",
                "artifact_commit",
                "artifact_tree",
                "artifact_computation_sha256",
                "command",
                "dataset_sha256",
                "dataset_manifest_sha256",
                "derived_manifest_sha256",
                "baseline_manifest_sha256",
                "baseline_revisions",
                "python_lock_sha256",
                "seed",
                "toolchain",
                "cpu",
                "threads",
            }
            absent = sorted(required - record.keys())
            if absent:
                errors.append(
                    f"claim {ident}: {experiment} lacks provenance fields "
                    + ", ".join(absent)
                )
            if record.get("status") != "passed":
                errors.append(f"claim {ident}: experiment {experiment} did not pass")
            if record.get("campaign_start_dirty"):
                errors.append(f"claim {ident}: campaign for {experiment} did not start clean")
            if record.get("artifact_dirty"):
                errors.append(f"claim {ident}: experiment {experiment} used a dirty tree")
            if record.get("campaign_start_commit") != record.get("artifact_commit"):
                errors.append(f"claim {ident}: artifact commit changed during {experiment}")
            if record.get("campaign_start_computation_sha256") != record.get(
                "artifact_computation_sha256"
            ):
                errors.append(
                    f"claim {ident}: computation inputs changed during {experiment}"
                )
            if not isinstance(record.get("dataset_sha256"), dict) or not record.get(
                "dataset_sha256"
            ):
                errors.append(
                    f"claim {ident}: experiment {experiment} recorded no dataset hashes"
                )
            if record.get("experiment") != experiment:
                errors.append(
                    f"claim {ident}: bundle identity mismatch for {experiment}"
                )
            expected_seed = declared_seed(steps[experiment])
            if record.get("seed") != expected_seed:
                errors.append(
                    f"claim {ident}: {experiment} seed is {record.get('seed')!r}, "
                    f"expected {expected_seed!r}"
                )
            if record.get("baseline_revisions") != current_baselines:
                errors.append(f"claim {ident}: baseline pins changed after {experiment}")
            if record.get("python_lock_sha256") != current_lock:
                errors.append(f"claim {ident}: Python lock changed after {experiment}")
            for key, actual, expected in (
                ("dataset manifest", record.get("dataset_manifest_sha256"), current_dataset_manifest),
                ("derived manifest", record.get("derived_manifest_sha256"), current_derived_manifest),
                ("baseline manifest", record.get("baseline_manifest_sha256"), current_baseline_manifest),
            ):
                if actual != expected:
                    errors.append(f"claim {ident}: {key} changed after {experiment}")

            for dependency in steps[experiment].get("depends_on", []):
                hashes = record.get("dependency_bundle_sha256", {})
                if dependency not in hashes:
                    notes.add(
                        f"{experiment} predates dependency-bundle hashing for {dependency}"
                    )

            commit = record.get("artifact_commit", "")
            recorded_computation = record.get("artifact_computation_sha256")
            commit_computation = computation_sha256_at(commit)
            if commit_computation is None:
                message = (
                    f"{experiment} authoring commit {str(commit)[:12]} is unavailable; "
                    "the recorded computation hash is retained structurally"
                )
                if strict:
                    errors.append(f"claim {ident}: {message}")
                else:
                    notes.add(message)
            elif recorded_computation != commit_computation:
                errors.append(
                    f"claim {ident}: {experiment} computation hash does not match "
                    f"its recorded commit {str(commit)[:12]}"
                )
            else:
                if recorded_computation != current_computation:
                    notes.add(
                        f"{experiment} matches its recorded commit and predates "
                        "the current computation tree"
                    )
                if git(["merge-base", "--is-ancestor", commit, "HEAD"]).returncode:
                    message = (
                        f"{experiment} commit {str(commit)[:12]} is not an ancestor "
                        "of this checkout"
                    )
                    if strict:
                        errors.append(f"claim {ident}: {message}")
                    else:
                        notes.add(message)

            for relative, expected in record.get("dataset_sha256", {}).items():
                # Bundles retained before the interpreter was excluded from dataset
                # collection still carry a `data/venv/**` entry. It describes the host
                # -- the venv `python` resolves to the system interpreter -- so a patch
                # upgrade would otherwise invalidate evidence it cannot have changed.
                if relative.startswith("data/venv/"):
                    continue
                path = ROOT / relative
                if not path.is_file():
                    if strict:
                        errors.append(
                            f"claim {ident}: missing strict-mode dataset {relative}"
                        )
                    else:
                        notes.add(
                            "datasets are absent; recorded hashes were checked "
                            "structurally only"
                        )
                elif sha256(path) != expected:
                    errors.append(
                        f"claim {ident}: dataset hash no longer matches for {relative}"
                    )

    for note in sorted(notes):
        print(f"NOTE: {note}.", file=sys.stderr)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema-only", action="store_true")
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="validate evidence for disabled claims before promoting them",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="require datasets and resolvable authoring history",
    )
    args = parser.parse_args()
    try:
        with (ROOT / "experiments.toml").open("rb") as handle:
            config = tomllib.load(handle)
        with (ROOT / "results" / "claims.toml").open("rb") as handle:
            claims = tomllib.load(handle)
        errors = validate_schema(config, claims)
        if not args.schema_only and not errors:
            errors.extend(
                validate_evidence(
                    config, claims, args.include_disabled, strict=args.strict
                )
            )
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    if args.schema_only:
        print("experiment schema is valid")
    else:
        mode = "strict" if args.strict else "portable"
        print(f"all retained claim evidence is valid ({mode} mode)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
