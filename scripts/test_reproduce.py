"""Regression tests for the strict campaign runner."""
from __future__ import annotations

from io import StringIO
import os
import sys
import unittest
from unittest import mock

from scripts import reproduce


class RunLiveTests(unittest.TestCase):
    def test_public_command_hides_active_interpreter_path(self) -> None:
        command = [sys.executable, "scripts/example.py", "--seed", "42"]
        self.assertEqual(
            reproduce.public_command(command),
            ["python3", "scripts/example.py", "--seed", "42"],
        )

    def test_tees_and_retains_both_streams(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        command = [
            sys.executable,
            "-c",
            "import sys; print('captured-out'); print('captured-err', file=sys.stderr)",
        ]
        with mock.patch.object(reproduce.sys, "stdout", stdout), mock.patch.object(
            reproduce.sys, "stderr", stderr
        ):
            result = reproduce.run_live(command, os.environ.copy())

        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "captured-out\n")
        self.assertEqual(result.stderr, "captured-err\n")
        self.assertEqual(stdout.getvalue(), result.stdout)
        self.assertEqual(stderr.getvalue(), result.stderr)

    def test_returns_nonzero_status_without_losing_stderr(self) -> None:
        stderr = StringIO()
        command = [
            sys.executable,
            "-c",
            "import sys; print('expected failure', file=sys.stderr); raise SystemExit(7)",
        ]
        with mock.patch.object(reproduce.sys, "stdout", StringIO()), mock.patch.object(
            reproduce.sys, "stderr", stderr
        ):
            result = reproduce.run_live(command, os.environ.copy())

        self.assertEqual(result.returncode, 7)
        self.assertEqual(result.stderr, "expected failure\n")
        self.assertEqual(stderr.getvalue(), result.stderr)


if __name__ == "__main__":
    unittest.main()


class TierEnvTests(unittest.TestCase):
    """An evidence tier's codec knobs must come from the tier, not from a shell."""

    @staticmethod
    def _config(tiers: dict) -> dict:
        return {"schema_version": 1, "tiers": tiers}

    def test_systems_tier_declares_the_ivf_centering_it_reports(self) -> None:
        # The regression this whole mechanism exists for: centering reached four
        # systems bundles from an interactive shell, so a clean rerun of the tier
        # would have measured the uncentered protocol the manuscript does not report.
        config = reproduce.load_config()
        declared = reproduce.tier_env("systems", config["tiers"])
        self.assertEqual(declared.get("ULTRAVEC_IVF_CENTER"), "1")

    def test_undeclared_knob_is_refused_rather_than_silently_inherited(self) -> None:
        config = reproduce.load_config()
        with mock.patch.dict(os.environ, {"ULTRAVEC_ROTATION": "pca"}, clear=False):
            with self.assertRaises(reproduce.GateError) as caught:
                reproduce.run_tier("systems", config, dry_run=True)
        self.assertIn("ULTRAVEC_ROTATION", str(caught.exception))

    def test_harness_pointer_is_inherited_rather_than_refused(self) -> None:
        """The container ships its binary at /usr/local/bin and exports
        ULTRAVEC_BIN to say so, which is a location and not a protocol. Refusing it
        made `reproduce.py smoke` fail inside the published image while the README
        advertised that the image can run exactly doctor, check and smoke."""
        config = reproduce.load_config()
        with mock.patch.dict(
            os.environ, {"ULTRAVEC_BIN": "/usr/local/bin/ultravec"}, clear=False
        ):
            with mock.patch("sys.stdout", new=StringIO()):
                reproduce.run_tier("smoke", config, dry_run=True)

    def test_a_research_knob_is_still_refused_beside_a_harness_pointer(self) -> None:
        """The exemption is a named allowlist, not a relaxation: a knob that can
        change a measurement stays refused even when a harness pointer is set."""
        config = reproduce.load_config()
        with mock.patch.dict(
            os.environ,
            {"ULTRAVEC_BIN": "/usr/local/bin/ultravec", "ULTRAVEC_ROTATION": "pca"},
            clear=False,
        ):
            with self.assertRaises(reproduce.GateError) as caught:
                reproduce.run_tier("smoke", config, dry_run=True)
        self.assertIn("ULTRAVEC_ROTATION", str(caught.exception))
        self.assertNotIn("ULTRAVEC_BIN", str(caught.exception))

    def test_declared_knob_may_be_inherited_only_at_its_declared_value(self) -> None:
        config = reproduce.load_config()
        with mock.patch.dict(os.environ, {"ULTRAVEC_IVF_CENTER": "1"}, clear=False):
            with mock.patch("sys.stdout", new=StringIO()):
                reproduce.run_tier("systems", config, dry_run=True)
        with mock.patch.dict(os.environ, {"ULTRAVEC_IVF_CENTER": "0"}, clear=False):
            with self.assertRaises(reproduce.GateError):
                reproduce.run_tier("systems", config, dry_run=True)

    def test_referenced_tiers_merge_and_conflicts_are_an_error(self) -> None:
        merged = self._config({
            "parent": {"env": {"ULTRAVEC_A": "1"}, "steps": [{"tier": "child"}]},
            "child": {"env": {"ULTRAVEC_B": "2"}, "steps": []},
        })
        self.assertEqual(
            reproduce.tier_env("parent", merged["tiers"]),
            {"ULTRAVEC_A": "1", "ULTRAVEC_B": "2"},
        )
        clashing = self._config({
            "parent": {"env": {"ULTRAVEC_A": "1"}, "steps": [{"tier": "child"}]},
            "child": {"env": {"ULTRAVEC_A": "9"}, "steps": []},
        })
        with self.assertRaises(reproduce.GateError):
            reproduce.tier_env("parent", clashing["tiers"])

    def test_a_tier_may_not_declare_a_non_codec_variable(self) -> None:
        config = self._config({"t": {"env": {"PATH": "/tmp"}, "steps": []}})
        with self.assertRaises(reproduce.GateError):
            reproduce.tier_env("t", config["tiers"])
