"""Regression tests for the distribution-shift comparison protocol."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts import ann_distshift


class ShiftProtocolTests(unittest.TestCase):
    def test_trellis_reconstructs_database_but_not_query(self) -> None:
        reconstruction = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        queries = np.array([[0.75, 0.25]], dtype=np.float32)
        ranked = np.array([[0]], dtype=np.int64)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            ann_distshift, "recon", return_value=Path(directory) / "db.fvecs"
        ) as reconstruct, mock.patch.object(
            ann_distshift, "rd", return_value=reconstruction
        ), mock.patch.object(
            ann_distshift, "cosine_topk", return_value=ranked
        ) as topk:
            recall, byte_count = ann_distshift.trellis_recall(
                Path(directory) / "source.fvecs",
                queries,
                2,
                10,
                [{0}],
                "test",
                Path(directory),
            )

        reconstruct.assert_called_once()
        np.testing.assert_array_equal(topk.call_args.args[0], reconstruction)
        np.testing.assert_array_equal(topk.call_args.args[1], queries)
        self.assertEqual(recall, 0.1)
        self.assertEqual(byte_count, 7)


if __name__ == "__main__":
    unittest.main()
