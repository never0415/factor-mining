import unittest

import numpy as np
import torch

from min_gp.factor_diagnostics import (
    daily_pearson_ic, daily_rank_ic, diagnose, parse_factor, prepare_signal,
    weekly_portfolios,
)
from min_gp.evaluation import BatchedNeutralizer


class FactorDiagnosticsTest(unittest.TestCase):
    def test_ic_and_rank_ic_are_cross_sectional(self):
        factor = torch.arange(10, dtype=torch.float32).unsqueeze(1).repeat(1, 3)
        returns = factor * 2 + 1
        self.assertTrue(torch.allclose(
            daily_pearson_ic(factor, returns, 5), torch.ones(3)
        ))
        self.assertTrue(torch.allclose(
            daily_rank_ic(factor, returns, 5), torch.ones(3)
        ))

    def test_weekly_cost_is_charged_only_on_weight_changes(self):
        signal = torch.arange(10, dtype=torch.float32).unsqueeze(1).repeat(1, 2)
        returns = signal / 100
        result = weekly_portfolios(
            signal, returns, ["d1", "d2"], groups=5,
            cost_bps=30, min_cross_section=5,
        )
        self.assertAlmostEqual(result["turnover"][0], 1.0)
        self.assertAlmostEqual(result["turnover"][1], 0.0)
        self.assertAlmostEqual(
            result["net_ls"][0], result["gross_ls"][0] - .003,
        )
        self.assertAlmostEqual(result["net_ls"][1], result["gross_ls"][1])

    def test_diagnose_returns_compounded_nav(self):
        signal = torch.arange(10, dtype=torch.float32).unsqueeze(1).repeat(1, 3)
        returns = signal / 100
        metrics, series = diagnose(
            signal, returns, ["d1", "d2", "d3"],
            min_cross_section=5, groups=5,
        )
        self.assertAlmostEqual(metrics["ic"], 1.0, places=6)
        self.assertAlmostEqual(metrics["rank_ic"], 1.0, places=6)
        self.assertEqual(metrics["weeks"], 3)
        expected = np.cumprod(1 + series["net_ls"])
        self.assertTrue(np.allclose(series["net_nav"], expected))

    def test_windows_path_parser(self):
        self.assertEqual(
            parse_factor(r"best=C:\data\factor.parquet:-1"),
            ("best", r"C:\data\factor.parquet", -1),
        )

    def test_prepare_signal_neutralizes_after_averaging(self):
        torch.manual_seed(8)
        size = torch.randn(50, 6)
        industry = torch.arange(50).remainder(5).view(-1, 1).expand(-1, 6)
        factor = 3 * size + (industry == 2).float() * 4 + torch.randn(50, 6) * .01
        neutralizer = BatchedNeutralizer(
            factor.shape, (size,), industry, min_cross_section=20,
        )
        signal = prepare_signal(factor, 1, -1, neutralizer)
        valid = torch.isfinite(signal)
        correlation = torch.corrcoef(
            torch.stack((signal[valid], size[valid]))
        )[0, 1]
        self.assertLess(abs(float(correlation)), 1e-4)


if __name__ == "__main__":
    unittest.main()
