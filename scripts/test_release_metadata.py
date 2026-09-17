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

    def test_paper_pdf_is_explicitly_excluded_from_mit(self) -> None:
        relative = "paper/ultravec-ann-vldbj.pdf"
        license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        paper_license = (ROOT / "paper/PAPER-LICENSE.md").read_text(encoding="utf-8")
        self.assertIn(relative, license_text)
        self.assertIn("explicitly excluded", license_text)
        self.assertIn("All rights reserved", paper_license)
        self.assertIn("excluded from the repository's MIT license", paper_license)
        self.assertTrue((ROOT / relative).is_file())

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
        self.assertFalse(any((ROOT / "paper").glob("*.tex")))


if __name__ == "__main__":
    unittest.main()
