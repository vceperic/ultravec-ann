"""Release-shape and license-scope regression tests."""
from __future__ import annotations

from pathlib import Path
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReleaseMetadataTests(unittest.TestCase):
    def test_software_is_mit_only(self) -> None:
        with (ROOT / "Cargo.toml").open("rb") as handle:
            cargo = tomllib.load(handle)
        self.assertEqual(cargo["package"]["license"], "MIT")
        self.assertFalse((ROOT / "LICENSE-APACHE").exists())
        self.assertFalse((ROOT / "LICENSE-MIT").exists())
        self.assertIn("MIT License", (ROOT / "LICENSE").read_text(encoding="utf-8"))

    def test_repository_carries_no_paper(self) -> None:
        # The paper is published separately; the repository is code and evidence.
        self.assertFalse((ROOT / "paper").exists())
        self.assertEqual(list(ROOT.rglob("*.pdf")) and [
            p for p in ROOT.rglob("*.pdf")
            if not any(part in {"target", "data", ".venv"} for part in p.parts)
        ], [])

    def test_metadata_names_the_public_repository_without_inventing_a_release(self) -> None:
        citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        self.assertIn("version: 0.1.0", citation)
        self.assertIn("license: MIT", citation)
        self.assertIn(
            'repository-code: "https://github.com/vceperic/ultravec-ann"',
            citation,
        )
        self.assertNotIn("date-released:", citation)
        self.assertNotIn("doi:", citation.lower())
        self.assertNotIn("status: submitted", citation)

        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("https://github.com/vceperic/ultravec-ann", readme)
        self.assertNotIn("future standalone repository", readme)
        self.assertNotIn("no public remote", readme)

    def test_artifact_does_not_duplicate_manuscript_source(self) -> None:
        self.assertFalse((ROOT / "manuscript").exists())
        self.assertFalse(any(ROOT.glob("*.tex")))


if __name__ == "__main__":
    unittest.main()
