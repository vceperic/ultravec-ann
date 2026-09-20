"""Regression tests for portable claim-validation policy."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tomllib
import unittest

from scripts import validate_claims

ROOT = Path(__file__).resolve().parents[1]


class ClaimValidationSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        with (ROOT / "experiments.toml").open("rb") as handle:
            self.config = tomllib.load(handle)
        with (ROOT / "results" / "claims.toml").open("rb") as handle:
            self.claims = tomllib.load(handle)

    def test_validated_status_is_enforced(self) -> None:
        claims = copy.deepcopy(self.claims)
        claims["status"] = "requires-clean-rerun"
        errors = validate_claims.validate_schema(self.config, claims)
        self.assertTrue(any("status must be" in error for error in errors))

    def test_claim_bound_scann_requires_a_seed(self) -> None:
        config = copy.deepcopy(self.config)
        for tier in config["tiers"].values():
            for step in tier.get("steps", []):
                if step.get("experiment") == "systems-scann-sift1m":
                    step.pop("seed", None)
                    argv = step.get("argv", [])
                    if "--seed" in argv:
                        index = argv.index("--seed")
                        del argv[index:index + 2]
        errors = validate_claims.validate_schema(config, self.claims)
        self.assertTrue(any("has no controlled seed" in error for error in errors))

    def test_scann_determinism_gate_is_enforced(self) -> None:
        config = copy.deepcopy(self.config)
        for tier in config["tiers"].values():
            for step in tier.get("steps", []):
                if step.get("experiment") == "systems-scann-sift1m":
                    step["argv"] = [
                        value
                        for value in step["argv"]
                        if value != "--verify-determinism"
                    ]
        errors = validate_claims.validate_schema(config, self.claims)
        self.assertTrue(any("--verify-determinism" in error for error in errors))

    def test_resolvable_evidence_commits_match_recorded_computation(self) -> None:
        seen: set[str] = set()
        for manifest in sorted((ROOT / "results" / "generated").glob("*/manifest.json")):
            record = json.loads(manifest.read_text(encoding="utf-8"))
            commit = record.get("artifact_commit")
            if not commit or commit in seen:
                continue
            seen.add(commit)
            computation = validate_claims.computation_sha256_at(commit)
            recorded = record.get("artifact_computation_sha256")
            if computation is not None and recorded is not None:
                self.assertEqual(computation, recorded, commit)


if __name__ == "__main__":
    unittest.main()
