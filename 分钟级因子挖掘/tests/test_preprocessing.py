import unittest

import torch

from min_gp.numeric.preprocessing import (
    align_signal, industry_one_hot, neutralize, remove_outliers,
)


class RemoveOutliersTest(unittest.TestCase):
    def test_clips_each_date_to_median_plus_or_minus_five_mad(self):
        factor = torch.tensor([
            [0.0, -100.0],
            [1.0, 0.0],
            [2.0, 1.0],
            [3.0, 2.0],
            [4.0, 3.0],
            [100.0, 4.0],
        ])
        result = remove_outliers(factor)
        self.assertEqual(float(result[-1, 0]), 10.0)
        self.assertEqual(float(result[0, 1]), -6.0)
        self.assertEqual(float(result[2, 0]), 2.0)

    def test_is_cross_section_local_and_preserves_missing_values(self):
        factor = torch.tensor([
            [1.0, float("nan")],
            [1.0, 2.0],
            [1.0, 3.0],
            [20.0, 4.0],
        ])
        result = remove_outliers(factor)
        self.assertEqual(float(result[-1, 0]), 1.0)
        self.assertTrue(torch.isnan(result[0, 1]))
        self.assertEqual(float(result[-1, 1]), 4.0)

    def test_rejects_negative_multiplier(self):
        with self.assertRaises(ValueError):
            remove_outliers(torch.ones(3, 2), n_mad=-1)


class IndustryOneHotTest(unittest.TestCase):
    def test_expands_codes_into_fixed_width_dummies(self):
        industry = torch.tensor([[0, 1], [2, 2], [1, 0]])
        result = industry_one_hot(industry, levels=4)
        self.assertEqual(tuple(result.shape), (3, 2, 4))
        self.assertEqual(result[0, 0].tolist(), [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(result[1, 1].tolist(), [0.0, 0.0, 1.0, 0.0])

    def test_missing_industry_gets_an_all_zero_row(self):
        result = industry_one_hot(torch.tensor([[-1], [0]]), levels=2)
        self.assertEqual(result[0, 0].tolist(), [0.0, 0.0])
        self.assertEqual(result[1, 0].tolist(), [1.0, 0.0])


class NeutralizeTest(unittest.TestCase):
    def test_removes_industry_means(self):
        # Two industries with different levels; the residual must be the
        # deviation from each industry's own mean, so both groups centre on 0.
        factor = torch.tensor([[10.0], [12.0], [30.0], [34.0]])
        industry = torch.tensor([[0], [0], [1], [1]])
        result = neutralize(factor, industry, levels=2, min_cross_section=1)
        self.assertTrue(torch.allclose(
            result.squeeze(1), torch.tensor([-1.0, 1.0, -2.0, 2.0]), atol=1e-5
        ))

    def test_removes_a_linear_market_cap_exposure(self):
        cap = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        factor = 3.0 * cap + 5.0
        result = neutralize(
            factor, industry=None, continuous=(cap,), min_cross_section=1
        )
        self.assertTrue(torch.allclose(
            result, torch.zeros_like(result), atol=1e-4
        ))

    def test_survives_an_industry_absent_from_the_cross_section(self):
        # Level 1 has no members: its design column is exactly zero and the
        # normal matrix is singular. pinv must still return the group means.
        factor = torch.tensor([[10.0], [12.0], [30.0], [34.0]])
        industry = torch.tensor([[0], [0], [2], [2]])
        result = neutralize(factor, industry, levels=3, min_cross_section=1)
        self.assertTrue(torch.allclose(
            result.squeeze(1), torch.tensor([-1.0, 1.0, -2.0, 2.0]), atol=1e-5
        ))

    def test_drops_names_whose_exposure_is_missing(self):
        factor = torch.tensor([[10.0], [12.0], [14.0]])
        industry = torch.tensor([[0], [0], [-1]])
        result = neutralize(factor, industry, levels=1, min_cross_section=1)
        self.assertTrue(torch.isnan(result[2, 0]))
        self.assertFalse(torch.isnan(result[0, 0]))

    def test_drops_dates_below_the_minimum_cross_section(self):
        factor = torch.tensor([[10.0], [12.0]])
        industry = torch.tensor([[0], [0]])
        result = neutralize(factor, industry, levels=1, min_cross_section=30)
        self.assertTrue(torch.isnan(result).all())

    def test_rejects_a_design_with_no_exposures(self):
        with self.assertRaises(ValueError):
            neutralize(torch.ones(3, 2))


class AlignSignalTest(unittest.TestCase):
    def test_close_offset_delays_by_one_day(self):
        factor = torch.tensor([[1.0, 2.0, 3.0]])
        result = align_signal(factor, "close")
        self.assertTrue(torch.isnan(result[0, 0]))
        self.assertEqual(result[0, 1].item(), 1.0)
        self.assertEqual(result[0, 2].item(), 2.0)

    def test_none_offset_is_a_passthrough(self):
        factor = torch.tensor([[1.0, 2.0]])
        self.assertTrue(torch.equal(align_signal(factor, "none"), factor))

    def test_rejects_an_unknown_offset(self):
        with self.assertRaises(ValueError):
            align_signal(torch.ones(2, 2), "open")


if __name__ == "__main__":
    unittest.main()
