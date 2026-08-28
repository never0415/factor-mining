import tempfile
import unittest
from pathlib import Path

import torch

from min_gp.factor_export import factor_frame, write_factor_parquet
from min_gp.spectral_data import load_daily_factor_leaves


class FactorExportTest(unittest.TestCase):
    def test_canonical_frame_order_and_status(self):
        factor = torch.tensor([[1.0, float("nan")], [3.0, 4.0]])
        frame = factor_frame(factor, ["a", "b"], ["d1", "d2"])
        self.assertEqual(frame["instrument"].tolist(), ["a", "a", "b", "b"])
        self.assertEqual(frame["trade_date"].tolist(), ["d1", "d2", "d1", "d2"])
        self.assertEqual(frame["status"].tolist(), ["ok", "invalid", "ok", "ok"])

    def test_export_is_a_daily_gp_leaf_without_conversion(self):
        factor = torch.tensor([[1.0, float("nan")], [3.0, 4.0]])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "candidate.parquet"
            write_factor_parquet(factor, ["a", "b"], ["d1", "d2"], path)
            loaded = load_daily_factor_leaves(
                path, ["d1", "d2"], ["a", "b"], ["factor"], device="cpu"
            )["factor"]
        self.assertTrue(torch.allclose(loaded, factor, equal_nan=True))

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            factor_frame(torch.ones(2, 3), ["a"], ["d1", "d2", "d3"])


if __name__ == "__main__":
    unittest.main()
