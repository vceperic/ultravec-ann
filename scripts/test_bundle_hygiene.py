#!/usr/bin/env python3
"""Ensure committed evidence does not carry authoring-machine paths."""
from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The recheck directory is created by tempfile with this prefix. Its literal path
# identifies nothing sensitive, but shipping it contradicts the documented
# normalization, so the placeholder is the only accepted form.
EPHEMERAL = re.compile(r"/tmp/ultravec-[A-Za-z0-9_.-]+")

# Absolute paths that identify the machine the evidence was produced on. Kept
# generic on purpose: naming a specific checkout root here would publish it.
FORBIDDEN = re.compile(
    r"""(
          /(home|root|Users)/          # a user's home directory
        | /[A-Za-z0-9_.-]+/\.(claude|codex)/   # an agent worktree layout
        | -----BEGIN\ (RSA\ |OPENSSH\ )?PRIVATE\ KEY
    )""",
    re.VERBOSE,
)


def tracked_files() -> list[Path]:
    """Only shipped files; fall back to source-archive contents without Git."""
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=False
    )
    if out.returncode == 0 and out.stdout:
        return [ROOT / raw.decode() for raw in out.stdout.split(b"\0") if raw]
    excluded = {".git", "target", "data", ".venv", "__pycache__", ".pytest_cache"}
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file() and not excluded.intersection(path.relative_to(ROOT).parts)
    ]


class BundleHygiene(unittest.TestCase):
    def test_no_machine_paths_in_tracked_files(self) -> None:
        offenders: list[str] = []
        for path in tracked_files():
            # This test states the pattern it forbids, so it necessarily contains it.
            if path.name == "test_bundle_hygiene.py":
                continue
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue  # binary (figures, fixtures) -- nothing to leak in text form
            for number, line in enumerate(text.splitlines(), start=1):
                if FORBIDDEN.search(line):
                    rel = path.relative_to(ROOT)
                    offenders.append(f"{rel}:{number}")
        self.assertEqual(
            offenders,
            [],
            "tracked files carry authoring-machine paths; the tier that wrote them "
            "must normalize the artifact root to '.' at the point of write:\n  "
            + "\n  ".join(offenders[:40]),
        )

    def test_evidence_logs_use_relative_paths(self) -> None:
        """Bundles record where they wrote things; that must be repo-relative."""
        generated = ROOT / "results" / "generated"
        if not generated.is_dir():
            self.skipTest("no generated bundles present")
        bad: list[str] = []
        for path in sorted(generated.rglob("*")):
            if not path.is_file() or path.suffix not in {".txt", ".json", ".csv", ".md"}:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if "→ wrote /" in line or "wrote /" in line and line.strip().startswith("→"):
                    bad.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()[:90]}")
        self.assertEqual(
            bad, [], "evidence logs record absolute output paths:\n  " + "\n  ".join(bad[:20])
        )

    def test_no_ephemeral_temp_paths_in_evidence(self) -> None:
        """The recheck temporary directory must be normalized at the point of write."""
        generated = ROOT / "results" / "generated"
        if not generated.is_dir():
            self.skipTest("no generated bundles present")
        offenders: list[str] = []
        for path in sorted(generated.rglob("*")):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if EPHEMERAL.search(line):
                    offenders.append(f"{path.relative_to(ROOT)}:{number}")
        self.assertEqual(
            offenders,
            [],
            "evidence carries a literal recheck temporary directory; "
            "prepare_official_rabitq.py must redact it to '<temporary-directory>' "
            "when it echoes the command:\n  " + "\n  ".join(offenders[:20]),
        )

if __name__ == "__main__":
    unittest.main()
