"""Regression tests for the paper's declared evaluation protocol."""
from __future__ import annotations

import ast
from pathlib import Path
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ExperimentProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        with (ROOT / "experiments.toml").open("rb") as handle:
            self.config = tomllib.load(handle)

    def experiment_step(self, experiment: str) -> dict:
        for tier in self.config["tiers"].values():
            for step in tier.get("steps", []):
                if step.get("experiment") == experiment:
                    return step
        self.fail(f"experiment not found: {experiment}")

    def test_main_flat_comparisons_use_1000_queries_and_retain_query_rows(self) -> None:
        # The superseded M=12 campaign is what retains per-query rows.
        for experiment in (
            "common-reconstruction-sift",
            "common-reconstruction-gist",
            "common-reconstruction-embedding",
        ):
            argv = self.experiment_step(experiment)["argv"]
            query_max = argv[argv.index("--query-max") + 1]
            self.assertEqual(query_max, "1000", experiment)
            output = argv[argv.index("--per-query-out") + 1]
            self.assertEqual(output, f"results/generated/{experiment}/per-query.csv")

    def test_reported_flat_comparison_uses_1000_queries(self) -> None:
        """The campaign the manuscript reports, which the test above did not cover.

        Asserting the query count only of the superseded runs left the reported ones
        unchecked, and quietly implied they retain per-query rows as well. They do not:
        their bootstrap is computed in-run and the intervals are what the bundle keeps,
        which is what Section 5.4 now says.
        """
        for experiment in ("flat16-sift", "flat16-gist", "flat16-dbpedia", "flat1m-sift"):
            argv = self.experiment_step(experiment)["argv"]
            query_max = argv[argv.index("--query-max") + 1]
            self.assertEqual(query_max, "1000", experiment)
            self.assertNotIn(
                "--per-query-out",
                argv,
                f"{experiment} now retains per-query rows; Section 5.4 and PAPER-MAP "
                "say the reported campaign does not, so update both",
            )

    def test_warm_comparisons_use_1000_queries(self) -> None:
        source = (ROOT / "scripts" / "dehub_vs_strong_matched.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        domains = next(
            node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "DOMAINS"
                for target in node.targets
            )
        )
        self.assertIsInstance(domains, ast.List)
        query_counts = [
            item.elts[-1].value
            for item in domains.elts
            if isinstance(item, ast.Tuple)
            and isinstance(item.elts[-1], ast.Constant)
        ]
        self.assertEqual(query_counts, [1000, 1000, 1000])


if __name__ == "__main__":
    unittest.main()
