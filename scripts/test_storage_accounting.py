"""Regression tests for experiment-facing storage labels."""
from __future__ import annotations

import unittest

try:
    from scripts.storage_accounting import trellis_serialized_bytes
except ModuleNotFoundError:  # direct `python scripts/...py` invocation
    from storage_accounting import trellis_serialized_bytes


class TrellisStorageTests(unittest.TestCase):
    def test_includes_state_and_rescale_at_power_of_two_dimension(self) -> None:
        self.assertEqual(trellis_serialized_bytes(128, 2, 10), 38)
        self.assertEqual(trellis_serialized_bytes(128, 4, 10), 70)

    def test_includes_transform_padding(self) -> None:
        self.assertEqual(trellis_serialized_bytes(960, 4, 10), 518)
        self.assertEqual(trellis_serialized_bytes(1536, 4, 10), 1030)

    def test_rejects_invalid_parameters(self) -> None:
        with self.assertRaises(ValueError):
            trellis_serialized_bytes(0, 2, 10)
        with self.assertRaises(ValueError):
            trellis_serialized_bytes(128, 0, 10)
        with self.assertRaises(ValueError):
            trellis_serialized_bytes(128, 2, -1)


if __name__ == "__main__":
    unittest.main()
