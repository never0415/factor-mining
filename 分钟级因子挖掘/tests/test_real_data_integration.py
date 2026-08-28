"""Opt-in tests against the configured local parquet lake.

Run with ``MIN_GP_RUN_INTEGRATION=1 uv run python -m unittest ...``.
"""

import os
import unittest

import torch

from min_gp.config import (
    ADJUSTED_CLOSE_PARQUET, INDUSTRY_VALUE_EXPOSURES_PARQUET,
    MINUTE_PARQUET, ZZ500_PIT_PARQUET,
)
from min_gp.data import load_pit_codes, load_pit_dates
from min_gp.spectral_data import (
    build_minute_slice, load_daily_close_tensor, load_daily_exposures,
)


@unittest.skipUnless(os.environ.get("MIN_GP_RUN_INTEGRATION") == "1", "opt-in real data test")
class RealParquetIntegrationTest(unittest.TestCase):
    def test_minute_price_exposure_alignment(self):
        start, end = "2024-01-02", "2024-01-10"
        instruments = load_pit_codes(ZZ500_PIT_PARQUET, start, end)[:5]
        dates = load_pit_dates(ZZ500_PIT_PARQUET, start, end)
        minute, meta = build_minute_slice(
            MINUTE_PARQUET, start, end, ("close", "volume"),
            instruments=instruments, dates=dates, device="cpu",
        )
        self.assertEqual(meta["NM"], 240)
        close = load_daily_close_tensor(
            ADJUSTED_CLOSE_PARQUET, meta["dates"], meta["instruments"], "cpu"
        )
        styles, industry, levels = load_daily_exposures(
            INDUSTRY_VALUE_EXPOSURES_PARQUET, meta["dates"],
            meta["instruments"], device="cpu",
        )
        self.assertEqual(close.shape, minute["volume"].shape[:2])
        self.assertEqual(industry.shape, close.shape)
        self.assertIn("ln_float_market_cap", styles)
        self.assertTrue(levels)
        self.assertTrue(torch.isfinite(close).any())


if __name__ == "__main__":
    unittest.main()

