#!/usr/bin/env python3
"""Repair and police the commit a retained bundle records.

Every bundle names the commit it was produced at. That name is how
``validate_claims.py --strict`` ties retained numbers to the code that computed
them. It breaks in one specific, silent way: a campaign runs in a worktree, the
branch is rebased before promotion, and the recorded SHA survives only as a
dangling object. Portable validation still passes -- the recorded computation
hash still matches the orphaned commit -- so nothing complains until the objects
are pruned, at which point the binding cannot be checked at all.

``--check`` is the gate for that: it fails when any bundle records a commit that
is not an ancestor of ``HEAD``. Run it from the monorepo, where the authoring
history exists; in the extracted standalone repository no bundle's commit
resolves and the check is not meaningful.

``--map OLD=NEW`` performs the repair. It refuses unless the two commits carry
byte-identical computation trees, because that equality is the whole reason a
re-stamp is a metadata correction rather than a claim about a run that never
happened. The original SHA is retained in ``rebased_from_commit``.

    python3 scripts/restamp_bundles.py --check
    python3 scripts/restamp_bundles.py --map 591750dce662=dcfb727478e3 --apply
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import validate_claims as vc

ROOT = Path(__file__).resolve().parents[1]
BUNDLES = ROOT / "results" / "generated"


def manifests() -> list[Path]:
    return sorted(BUNDLES.glob("*/manifest.json"))


def resolve(commit: str) -> str | None:
    """Full SHA for a commit-ish, or None when the object is absent."""
    result = vc.git(["rev-parse", "--verify", f"{commit}^{{commit}}"])
    if result.returncode:
        return None
    return result.stdout.decode("utf-8", errors="replace").strip()


def tree_of(commit: str) -> str | None:
    result = vc.git(["rev-parse", "--verify", f"{commit}^{{tree}}"])
    if result.returncode:
        return None
    return result.stdout.decode("utf-8", errors="replace").strip()


def check() -> int:
    """Fail when a bundle records a commit unreachable from HEAD."""
    problems: list[str] = []
    checked = 0
    for manifest in manifests():
        record = json.loads(manifest.read_text(encoding="utf-8"))
        commit = record.get("artifact_commit", "")
        name = manifest.parent.name
        if not commit:
            problems.append(f"{name}: no artifact_commit recorded")
            continue
        checked += 1
        if resolve(commit) is None:
            problems.append(f"{name}: commit {commit[:12]} does not resolve here")
        elif vc.git(["merge-base", "--is-ancestor", commit, "HEAD"]).returncode:
            problems.append(f"{name}: commit {commit[:12]} is not an ancestor of HEAD")
    if problems:
        print("bundle provenance check FAILED:")
        for problem in problems:
            print(f"  {problem}")
        print(
            "\nA rebase before promotion orphans the recorded SHA. Re-stamp with\n"
            "  python3 scripts/restamp_bundles.py --map <old>=<new> --apply"
        )
        return 1
    print(f"bundle provenance gate: PASS ({checked} bundles, all commits ancestors of HEAD)")
    return 0


def restamp(pairs: list[tuple[str, str]], apply: bool) -> int:
    """Rewrite the recorded commit for every bundle naming one of `pairs`."""
    resolved: list[tuple[str, str, str]] = []
    for old, new in pairs:
        old_full, new_full = resolve(old), resolve(new)
        if old_full is None:
            print(f"ERROR: source commit {old} does not resolve")
            return 2
        if new_full is None:
            print(f"ERROR: target commit {new} does not resolve")
            return 2
        # The safety property. Equal computation trees mean the re-stamp changes
        # which commit is named and nothing about what was computed; unequal ones
        # mean the target is a different experiment and the mapping is wrong.
        old_hash, new_hash = vc.computation_sha256_at(old_full), vc.computation_sha256_at(new_full)
        if old_hash is None or new_hash is None:
            print(f"ERROR: cannot compute a computation hash for {old[:12]} or {new[:12]}")
            return 2
        if old_hash != new_hash:
            print(
                f"ERROR: refusing {old[:12]} -> {new[:12]}: computation trees differ\n"
                f"  {old[:12]}: {old_hash}\n  {new[:12]}: {new_hash}"
            )
            return 2
        new_tree = tree_of(new_full)
        if new_tree is None:
            print(f"ERROR: cannot read the tree of {new[:12]}")
            return 2
        resolved.append((old_full, new_full, new_tree))

    touched = 0
    for manifest in manifests():
        record = json.loads(manifest.read_text(encoding="utf-8"))
        commit = record.get("artifact_commit", "")
        match = next((entry for entry in resolved if entry[0] == commit), None)
        if match is None:
            continue
        _, new_full, new_tree = match
        record["rebased_from_commit"] = commit
        record["artifact_commit"] = new_full
        # The validator requires these two to agree; a campaign whose start and end
        # commits differ is reported as "artifact commit changed during <experiment>".
        record["campaign_start_commit"] = new_full
        record["artifact_tree"] = new_tree
        touched += 1
        print(f"{manifest.parent.name}: {commit[:12]} -> {new_full[:12]}")
        if apply:
            # Byte-for-byte the format `reproduce.py` writes, so a later campaign
            # that regenerates this bundle produces a clean diff rather than a
            # whole-file rewrite.
            manifest.write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    if not touched:
        print("no bundle records any of the given source commits")
        return 1
    print(f"{'re-stamped' if apply else 'would re-stamp'} {touched} bundle(s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="fail if any recorded commit is unreachable from HEAD"
    )
    parser.add_argument(
        "--map", action="append", default=[], metavar="OLD=NEW",
        help="re-stamp bundles recording OLD to name NEW instead (repeatable)",
    )
    parser.add_argument(
        "--apply", action="store_true", help="write the changes; without it the run is a preview"
    )
    args = parser.parse_args(argv)
    if args.check and args.map:
        print("ERROR: --check inspects, --map rewrites; run them separately")
        return 2
    if args.check:
        return check()
    if not args.map:
        parser.print_usage()
        print("ERROR: pass --check or at least one --map OLD=NEW")
        return 2
    pairs: list[tuple[str, str]] = []
    for entry in args.map:
        old, separator, new = entry.partition("=")
        if not separator or not old or not new:
            print(f"ERROR: malformed --map {entry!r}; expected OLD=NEW")
            return 2
        pairs.append((old, new))
    return restamp(pairs, args.apply)


if __name__ == "__main__":
    raise SystemExit(main())
