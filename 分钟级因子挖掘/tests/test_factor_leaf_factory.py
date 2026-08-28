import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch

from min_gp.factor_leaf_factory import (
    LeafFactoryConfig, _pair_similarity, build_external_factor_leaves,
)


class FactorLeafFactoryTest(unittest.TestCase):
    def test_all_external_leaves_are_produced_from_local_data(self):
        torch.manual_seed(31)
        instruments, days, minutes = 4, 25, 12
        names = [f"sh60000{i}" for i in range(instruments)]
        dates = pd.bdate_range("2024-01-02", periods=days).strftime("%Y-%m-%d").tolist()
        close = 10 + torch.rand(instruments, days, minutes)
        context = {
            "open": close * .999,
            "high": close * 1.002,
            "low": close * .998,
            "close": close,
            "volume": torch.rand(instruments, days, minutes) * 100 + 1,
        }
        daily_close = close[..., -1]
        pool = torch.ones(instruments, days, dtype=torch.bool)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            market = root / "market.parquet"
            pd.DataFrame({
                "trade_date": dates,
                "close": np.linspace(100, 110, days),
            }).to_parquet(market, index=False)
            exposures = root / "exposures.parquet"
            pd.DataFrame([
                {
                    "instrument": instrument, "trade_date": date,
                    "ln_float_market_cap": np.log(1e10 + index),
                    "sw_level1": "test",
                }
                for index, (instrument, date) in enumerate(
                    (item for instrument in names for item in
                     [(instrument, date) for date in dates])
                )
            ]).to_parquet(exposures, index=False)
            report = build_external_factor_leaves(
                context, daily_close, pool, dates, names,
                ("raw_panic", "rushing_forward", "water_boat", "cooperation_effect"),
                LeafFactoryConfig(
                    market_parquet=str(market),
                    exposures_parquet=str(exposures),
                    api_fallback=False,
                    cache_directory=str(root / "cache"),
                ),
            )
        expected = {
            "amount_share", "volume_share", "up_volume_down_price_mask",
            "daily_close", "market_close", "daily_return", "float_market_cap",
            "high_amount", "low_amount", "price_state", "pair_similarity",
        }
        self.assertFalse(report["missing"])
        self.assertIn("local:CSI500:", report["built"]["market_close"])
        self.assertLessEqual(expected, set(context))
        self.assertEqual(context["pair_similarity"].shape, (instruments, instruments, days))
        self.assertEqual(context["pair_similarity"].dtype, torch.float16)
        self.assertEqual(context["price_state"].dtype, torch.int8)
        volume_sum = context["volume_share"].float().sum(0)
        amount_sum = context["amount_share"].float().sum(0)
        self.assertTrue(torch.allclose(volume_sum, torch.ones_like(volume_sum), atol=.01))
        self.assertTrue(torch.allclose(amount_sum, torch.ones_like(amount_sum), atol=.01))
        self.assertNotIn("rushing_forward", context)

    def test_pair_similarity_matches_brute_equality_count(self):
        close = torch.tensor([
            [[10., 11., 10., 12., 11., 13.]],
            [[10., 9., 10., 8., 9., 7.]],
        ])
        volume = torch.tensor([
            [[1., 2., 3., 2., 4., 5.]],
            [[1., 1., 2., 3., 2., 4.]],
        ])
        score = _pair_similarity(close, volume)
        self.assertEqual(score.shape, (2, 2, 1))
        self.assertTrue(torch.equal(score, score.transpose(0, 1)))
        self.assertTrue(torch.all(score.diagonal(dim1=0, dim2=1) >= score[0, 1]))


if __name__ == "__main__":
    unittest.main()
