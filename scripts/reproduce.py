#!/usr/bin/env python3
"""Strict tier runner for the ANN artifact (Python 3.11+)."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import tomllib

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments.toml"
BASELINES = ROOT / "baselines" / "manifest.toml"
COMPUTATION_PREFIXES = ("src/", "scripts/", "datasets/", "baselines/", "official_harness/")
COMPUTATION_FILES = {
    "Cargo.lock", "Cargo.toml", "rust-toolchain.toml", "requirements.in",
    "requirements.txt", "experiments.toml",
}

SHA_CACHE: dict[tuple[Path, int, int], str] = {}


class GateError(RuntimeError):
    pass


def load_config() -> dict:
    with CONFIG.open("rb") as handle:
        config = tomllib.load(handle)
    if config.get("schema_version") != 1 or not isinstance(config.get("tiers"), dict):
        raise GateError(f"unsupported or malformed configuration: {CONFIG}")
    return config


def distribution_errors(tier: dict) -> list[str]:
    errors: list[str] = []
    for requirement in tier.get("required_distributions", []):
        name, separator, expected = requirement.partition("==")
        if not name or not separator or not expected:
            errors.append(f"malformed distribution requirement {requirement!r}")
            continue
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            errors.append(f"{name}=={expected} is not installed in the active Python environment")
        else:
            if actual != expected:
                errors.append(f"{name}: expected {expected}, found {actual}")
    return errors


def required_files(name: str, tiers: dict, stack: tuple[str, ...] = ()) -> list[str]:
    if name in stack:
        raise GateError(f"recursive tier reference: {' -> '.join(stack + (name,))}")
    tier = tiers.get(name)
    if tier is None:
        raise GateError(f"unknown tier {name!r}")
    required = list(tier.get("required_files", []))
    for step in tier.get("steps", []):
        if "tier" in step:
            required.extend(required_files(step["tier"], tiers, stack + (name,)))
    return list(dict.fromkeys(required))


def tier_env(name: str, tiers: dict, stack: tuple[str, ...] = ()) -> dict[str, str]:
    """The ULTRAVEC_* operating point a tier declares, merged over referenced tiers.

    An evidence tier's codec knobs belong in `[tiers.<name>.env]`, not in whoever's
    shell launched it. Two tiers that disagree on one key would silently mix protocols
    within a campaign, so a conflict is an error rather than a last-writer-wins merge.
    """
    if name in stack:
        raise GateError(f"recursive tier reference: {' -> '.join(stack + (name,))}")
    tier = tiers.get(name)
    if tier is None:
        raise GateError(f"unknown tier {name!r}")
    declared: dict[str, str] = {}
    for key, value in dict(tier.get("env", {})).items():
        if not key.startswith("ULTRAVEC_"):
            raise GateError(
                f"tier {name!r} declares {key!r}; only ULTRAVEC_* knobs belong in a tier env"
            )
        declared[key] = str(value)
    for step in tier.get("steps", []):
        if "tier" in step:
            for key, value in tier_env(step["tier"], tiers, stack + (name,)).items():
                if declared.get(key, value) != value:
                    raise GateError(
                        f"tier {name!r} and its referenced tier {step['tier']!r} disagree on "
                        f"{key}: {declared[key]!r} vs {value!r}"
                    )
                declared.setdefault(key, value)
    return declared


def expand_steps(name: str, tiers: dict, stack: tuple[str, ...] = ()) -> list[dict]:
    if name in stack:
        raise GateError(f"recursive tier reference: {' -> '.join(stack + (name,))}")
    if name not in tiers:
        raise GateError(f"unknown tier: {name}")
    output: list[dict] = []
    for step in tiers[name].get("steps", []):
        if "tier" in step:
            output.extend(expand_steps(step["tier"], tiers, stack + (name,)))
        elif isinstance(step.get("argv"), list) and step["argv"]:
            output.append(step)
        else:
            raise GateError(f"tier {name} has malformed step {step!r}")
    return output


def doctor(config: dict) -> int:
    failures = 0
    print(f"artifact root: {ROOT}")
    print(f"python: {sys.version.split()[0]} ({sys.executable})")
    for tool in ("cargo", "rustc", "cmake", "c++"):
        found = shutil.which(tool)
        print(f"{tool}: {found or 'MISSING'}")
        failures += found is None
    for name, tier in config["tiers"].items():
        missing = [
            path
            for path in required_files(name, config["tiers"])
            if not (ROOT / path).is_file()
        ]
        distributions = distribution_errors(tier)
        print(f"tier {name}: {'READY' if not missing and not distributions else 'BLOCKED'}")
        for item in missing:
            print(f"  missing: {item}")
        for error in distributions:
            print(f"  Python environment: {error}")
    lock_missing = not (ROOT / "Cargo.lock").is_file()
    if lock_missing:
        print("release blocker: Cargo.lock is absent; locked/offline Rust builds are not reproducible")
        failures += 1
    return 1 if failures else 0


def sha256(path: Path) -> str:
    stat = path.stat()
    key = (path.resolve(), stat.st_size, stat.st_mtime_ns)
    if key in SHA_CACHE:
        return SHA_CACHE[key]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    SHA_CACHE[key] = value
    return value


def computation_paths() -> list[Path]:
    """List computation inputs in a Git checkout or a plain source archive."""
    completed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=False, capture_output=True
    )
    paths: set[Path] = set()
    if completed.returncode == 0:
        paths.update(
            Path(raw.decode("utf-8"))
            for raw in completed.stdout.split(b"\0")
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
    paths = {path for path in paths if (ROOT / path).is_file()}
    if not paths:
        raise GateError("no computation files found in the artifact")
    return sorted(paths)


def computation_sha256() -> str:
    paths = computation_paths()
    digest = hashlib.sha256()
    for relative in paths:
        digest.update(str(relative).encode("utf-8") + b"\0")
        digest.update(bytes.fromhex(sha256(ROOT / relative)))
    return digest.hexdigest()


def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if not path.is_dir():
        raise GateError(f"missing dependency bundle: {path.relative_to(ROOT)}")
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8") + b"\0")
        digest.update(bytes.fromhex(sha256(child)))
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


def command_output(argv: list[str]) -> str:
    result = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else f"unavailable ({result.returncode})"


def public_command(argv: list[str]) -> list[str]:
    """Normalize the active interpreter path for portable retained manifests."""
    output = list(argv)
    if output:
        try:
            if Path(output[0]).resolve() == Path(sys.executable).resolve():
                output[0] = "python3"
        except OSError:
            pass
    return output


def git_revision() -> tuple[str, bool]:
    revision = command_output(["git", "rev-parse", "HEAD"])
    # Scope the cleanliness check to the COMPUTATION tree. Two reasons: `git status`
    # reports the whole repository regardless of working directory, and this artifact
    # may sit in a subdirectory of a larger checkout, where an unscoped check fires
    # on edits that have nothing to do with the experiment; and a
    # run necessarily writes its own bundle under results/, so including outputs would
    # report dirty for every run by construction. What must be clean for evidence to
    # mean anything is the code and pins that produced it.
    scope = sorted(COMPUTATION_FILES) + [p.rstrip("/") for p in COMPUTATION_PREFIXES]
    dirty = bool(command_output(["git", "status", "--porcelain", "--", *scope]))
    return revision, dirty


def run_live(argv: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Run a step while teeing and retaining both output streams."""
    process = subprocess.Popen(
        argv,
        cwd=ROOT,
        env=env,
        text=True,
        bufsize=1,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []

    def pump(stream, sink, parts: list[str]) -> None:
        try:
            for chunk in iter(stream.readline, ""):
                parts.append(chunk)
                sink.write(chunk)
                sink.flush()
        finally:
            stream.close()

    assert process.stdout is not None and process.stderr is not None
    threads = [
        threading.Thread(target=pump, args=(process.stdout, sys.stdout, stdout_parts)),
        threading.Thread(target=pump, args=(process.stderr, sys.stderr, stderr_parts)),
    ]
    for thread in threads:
        thread.start()
    returncode = process.wait()
    for thread in threads:
        thread.join()
    return subprocess.CompletedProcess(
        argv, returncode, "".join(stdout_parts), "".join(stderr_parts)
    )


def record_bundle(
    step: dict,
    argv: list[str],
    completed: subprocess.CompletedProcess[str],
    env: dict,
    campaign_revision: str,
    campaign_computation: str,
    campaign_dirty: bool,
) -> None:
    experiment = step.get("experiment")
    if not experiment:
        return
    bundle = ROOT / "results" / "generated" / str(experiment)
    # Preserve an existing passing bundle when a re-run exits unsuccessfully;
    # diagnostics from the new attempt are written beside it for inspection.
    if completed.returncode and bundle.is_dir():
        bundle = bundle.with_name(f"{experiment}.failed")
        print(
            f"  step failed; retained evidence left intact, diagnostics -> "
            f"{bundle.relative_to(ROOT)}",
            flush=True,
        )
    bundle.mkdir(parents=True, exist_ok=True)
    public_stdout = completed.stdout.replace(str(ROOT), ".")
    public_stderr = completed.stderr.replace(str(ROOT), ".")
    (bundle / "stdout.txt").write_text(public_stdout, encoding="utf-8")
    (bundle / "stderr.txt").write_text(public_stderr, encoding="utf-8")
    datasets: dict[str, str] = {}
    # Corpora are fetch-only and are routinely symlinked to a shared copy, at either
    # granularity: `data` itself, or an individual corpus directory under it. Deriving
    # the bundle key from the *declared* path and hashing the *resolved* file handles
    # both, where resolving both sides only handled the first -- with `data/sift` a
    # symlink, every candidate resolved outside `data/` and was skipped, producing
    # bundles with no dataset hashes at all.
    data_root = (ROOT / "data").resolve()
    for value in [*argv, *[str(path) for path in step.get("datasets", [])]]:
        text = str(value)
        # `data/venv/**` is a build artifact, not an input. A Python step's argv[0] is
        # the venv interpreter, whose `python` is a symlink chain ending at the system
        # `/usr/bin/python3`, so hashing it records a fact about the host rather than
        # about the experiment: no reader reproduces that byte string, and a routine
        # interpreter upgrade retroactively invalidates every bundle that recorded it.
        # The interpreter's identity is already pinned by `python_lock_sha256` over
        # requirements.txt and by the recorded `toolchain`.
        if text.startswith("data/venv/") or "/data/venv/" in text:
            continue
        declared = Path(text)
        if declared.is_absolute():
            candidate = declared.resolve()
            try:
                key = Path("data") / candidate.relative_to(data_root)
            except ValueError:
                continue
        else:
            # The declared path is what the experiment names and what a reader will
            # look for; keep it as the key regardless of where it points.
            if declared.parts[:1] != ("data",):
                continue
            key = declared
            candidate = (ROOT / declared).resolve()
        if candidate.is_file():
            datasets[str(key)] = sha256(candidate)
    # A dataset-bearing step must record at least one dataset hash before its
    # provenance bundle can be accepted.
    if step.get("datasets") and not datasets:
        raise GateError(
            f"step {step['name']!r} declares datasets but none could be hashed; "
            "check that the paths under data/ resolve to readable files"
        )
    revision, dirty = git_revision()
    seed = step.get("seed")
    if "--seed" in argv and argv.index("--seed") + 1 < len(argv):
        seed = int(argv[argv.index("--seed") + 1])
    dependency_bundles = {
        dependency: directory_sha256(
            ROOT / "results" / "generated" / str(dependency)
        )
        for dependency in step.get("depends_on", [])
    }
    # Prefer the concrete model name. On Linux platform.processor() returns the bare
    # architecture ("x86_64"), which is truthy, so a `if not cpu` fallback never fires
    # and every manifest records an unusable CPU field -- while README.md asks readers
    # to report the CPU model with each timing. Try /proc/cpuinfo first.
    cpu = ""
    if Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                cpu = line.split(":", 1)[-1].strip()
                break
    cpu = cpu or platform.processor() or platform.machine()
    manifest = {
        "schema_version": 1,
        "experiment": experiment,
        "status": "passed" if completed.returncode == 0 else "failed",
        "campaign_start_commit": campaign_revision,
        "campaign_start_computation_sha256": campaign_computation,
        "campaign_start_dirty": campaign_dirty,
        "artifact_commit": revision,
        "artifact_tree": command_output(["git", "rev-parse", "HEAD^{tree}"]),
        "artifact_dirty": dirty,
        "artifact_computation_sha256": computation_sha256(),
        "command": public_command(argv),
        "dataset_sha256": datasets,
        "dataset_manifest_sha256": sha256(ROOT / "datasets" / "manifest.toml"),
        "derived_manifest_sha256": sha256(ROOT / "datasets" / "derived.toml"),
        "baseline_manifest_sha256": sha256(BASELINES),
        "baseline_revisions": baseline_revisions(),
        "python_lock_sha256": sha256(ROOT / "requirements.txt"),
        "seed": seed,
        "dependency_bundle_sha256": dependency_bundles,
        "toolchain": {
            "rustc": command_output(["rustc", "--version"]),
            "cargo": command_output(["cargo", "--version"]),
            "python": sys.version.split()[0],
        },
        "cpu": cpu or "unknown",
        "threads": {
            "rayon": env.get("RAYON_NUM_THREADS", "default"),
            "omp": env.get("OMP_NUM_THREADS", "default"),
        },
        # Record every active ULTRAVEC_* variable because these settings can alter
        # codec behavior. Evidence tiers explicitly select M=12 and thread counts.
        "ultravec_env": {
            key: value for key, value in sorted(env.items())
            if key.startswith("ULTRAVEC_")
        },
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_tier(name: str, config: dict, dry_run: bool) -> int:
    tier = config["tiers"].get(name)
    if tier is None:
        raise GateError(f"unknown tier {name!r}")
    # --dry-run exists to show what a tier would run. Enforcing inputs first made it
    # useless to exactly the reader who has not staged the corpora yet.
    missing = [] if dry_run else [
        path for path in required_files(name, config["tiers"]) if not (ROOT / path).is_file()
    ]
    if missing:
        detail = "\n".join(f"  - {item}" for item in missing)
        raise GateError(f"tier {name!r} is blocked by missing required inputs:\n{detail}")
    # Skipped under --dry-run for the same reason as required_files: showing the
    # commands must not require the environment that runs them.
    distributions = [] if dry_run else distribution_errors(tier)
    if distributions:
        detail = "\n".join(f"  - {item}" for item in distributions)
        raise GateError(
            f"tier {name!r} is blocked by the active Python environment ({sys.executable}):\n{detail}"
        )
    steps = expand_steps(name, config["tiers"])
    if not steps:
        raise GateError(f"tier {name!r} has no commands")
    campaign_revision, campaign_dirty = git_revision()
    campaign_computation = computation_sha256()
    produces_evidence = any(step.get("experiment") for step in steps)
    if (
        produces_evidence
        and name != "smoke"
        and not dry_run
        and campaign_revision.startswith("unavailable")
    ):
        raise GateError(
            "evidence tiers require a committed Git snapshot; portable check and "
            "doctor tiers remain available from source archives"
        )
    if name == "full" and not dry_run:
        if campaign_revision.startswith("unavailable") or campaign_dirty:
            raise GateError("the full campaign must start from a clean committed artifact tree")
    env = os.environ.copy()
    if name == "check":
        # Correctness tests must not inherit research-only process-global knobs.
        # Several backends retain environment compatibility for the CLI, so a
        # developer shell must not silently change unit-test construction.
        for key in [key for key in env if key.startswith("ULTRAVEC_")]:
            del env[key]
    else:
        # Evidence tiers define their operating point explicitly. A conflicting
        # inherited value is rejected so one campaign cannot mix protocols.
        declared = {
            "ULTRAVEC_TRELLIS_MEM": "12",
            "RAYON_NUM_THREADS": "20",
            "OMP_NUM_THREADS": "20",
        }
        declared.update(tier_env(name, config["tiers"]))
        for key, value in declared.items():
            inherited = env.get(key)
            if inherited is not None and inherited != value:
                raise GateError(
                    f"{key}={inherited!r} is set in the environment but this tier "
                    f"requires {value!r}; unset it rather than let the two disagree"
                )
        # Anything else the launching shell exported is refused rather than inherited.
        # An undeclared knob still lands in the manifest, so it reads as protocol long
        # after the shell that set it is gone, while a clean rerun of the same tier
        # silently measures something else. `ULTRAVEC_IVF_CENTER=1` reached four
        # systems bundles exactly this way.
        # Harness pointers name WHERE things live, not WHAT is measured, and the
        # container must set `ULTRAVEC_BIN` because its runtime stage ships the
        # binary at /usr/local/bin rather than target/release. They are inheritable
        # for that reason; every one still lands in the manifest's `ultravec_env`,
        # so a bundle records the paths it actually used. Research knobs -- anything
        # that changes a measurement -- stay refused.
        harness_pointers = {
            "ULTRAVEC_BIN",
            "ULTRAVEC_RESULTS_DIR",
            "ULTRAVEC_ERAB_DIR",
        }
        stray = sorted(
            key for key in env
            if key.startswith("ULTRAVEC_")
            and key not in declared
            and key not in harness_pointers
        )
        if stray:
            raise GateError(
                f"tier {name!r} inherits undeclared knobs from the environment: "
                + ", ".join(f"{key}={env[key]!r}" for key in stray)
                + "; unset them, or declare them under [tiers."
                + f"{name}.env] so the bundle records a protocol rather than a shell"
            )
        env.update(declared)
    for index, step in enumerate(steps, 1):
        argv = [str(part) for part in step["argv"]]
        if argv[0] == "python3":
            executable = Path(sys.executable)
            try:
                argv[0] = str(executable.relative_to(ROOT))
            except ValueError:
                argv[0] = str(executable)
        print(f"[{index}/{len(steps)}] {step['name']}: {' '.join(argv)}", flush=True)
        if not dry_run:
            completed = run_live(argv, env)
            if name == "full":
                current_revision, current_dirty = git_revision()
                current_computation = computation_sha256()
                if (
                    current_revision != campaign_revision
                    or current_computation != campaign_computation
                    or current_dirty
                ):
                    record_bundle(
                        step,
                        argv,
                        completed,
                        env,
                        campaign_revision,
                        campaign_computation,
                        campaign_dirty,
                    )
                    raise GateError(
                        "the committed artifact or computational inputs changed during the full campaign"
                    )
            record_bundle(
                step,
                argv,
                completed,
                env,
                campaign_revision,
                campaign_computation,
                campaign_dirty,
            )
            if completed.returncode:
                raise GateError(f"step {step['name']!r} failed with exit code {completed.returncode}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tier", nargs="?", default="doctor")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true", help="list configured tiers and exit")
    args = parser.parse_args()
    try:
        config = load_config()
        if args.list:
            for name, value in config["tiers"].items():
                print(f"{name:8} {value.get('description', '')}")
            return 0
        if args.tier == "doctor":
            return doctor(config)
        return run_tier(args.tier, config, args.dry_run)
    except (GateError, OSError, tomllib.TOMLDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
