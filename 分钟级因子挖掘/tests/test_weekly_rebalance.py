"""Calendar-week labels and trailing signal sampling semantics."""

import unittest

import torch

from min_gp.evaluation import trailing_signal_mean
from min_gp.label import tensor_weekly_fwd_ret, week_end_mask


class WeeklyRebalanceTest(unittest.TestCase):
    def setUp(self):
        # The second week is holiday-shortened and ends on Wednesday.
        self.dates = (
            "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
            "2024-01-08", "2024-01-09", "2024-01-10",
        )

    def test_last_actual_trading_day_of_each_week_is_selected(self):
        self.assertEqual(
            week_end_mask(self.dates).tolist(),
            [False, False, False, True, False, False, True],
        )

    def test_return_connects_one_rebalance_close_to_the_next(self):
        close = torch.tensor([[10., 11., 12., 20., 21., 22., 30.]])
        result = tensor_weekly_fwd_ret(close, self.dates)
        self.assertAlmostEqual(float(result[0, 3]), 0.5)
        self.assertTrue(torch.isnan(result[0, :3]).all())
        self.assertTrue(torch.isnan(result[0, 4:]).all())

    def test_five_day_mean_uses_current_and_four_prior_days_only(self):
        factor = torch.tensor([[1., 2., 3., 4., 5., 6.]])
        sampled = trailing_signal_mean(factor, 5)
        self.assertTrue(torch.isnan(sampled[0, :4]).all())
        self.assertEqual(sampled[0, 4:].tolist(), [3.0, 4.0])

    def test_five_day_mean_requires_five_valid_observations(self):
        factor = torch.tensor([[1., 2., float("nan"), 4., 5.]])
        self.assertTrue(torch.isnan(trailing_signal_mean(factor, 5)).all())


if __name__ == "__main__":
    unittest.main()
