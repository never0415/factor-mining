import unittest

import torch

from min_gp.factors.handbook import (
    ClimbMountainTemplate, CompleteTideTemplate, CooperationEffectTemplate,
    DarkFlowTemplate, EqualTreatmentTemplate, HiddenFlowerTemplate,
    IncompleteDefinitionError, LongShortBattleTemplate,
    RushingForwardTemplate, WaterBoatTemplate,
)
from min_gp.operators.distribution import boxcox_grid_mle


def _data(I=35, D=30, M=240):
    torch.manual_seed(12)
    close = 10 * torch.cumprod(1 + torch.randn(I, D, M) * .001, 2)
    open_ = close * (1 + torch.randn_like(close) * .0002)
    high = torch.maximum(open_, close) * 1.001
    low = torch.minimum(open_, close) * .999
    volume = torch.rand(I, D, M) * 1000 + 1
    return open_, high, low, close, volume


class CoreHandbookFactorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.open, cls.high, cls.low, cls.close, cls.volume = _data()

    def _shape_and_finite(self, factor):
        self.assertEqual(tuple(factor.shape), self.close.shape[:2])
        self.assertTrue(torch.isfinite(factor).any())

    def test_complete_tide(self):
        self._shape_and_finite(CompleteTideTemplate().evaluate(self.close, self.volume))

    def test_climb_mountain(self):
        self._shape_and_finite(ClimbMountainTemplate().evaluate(
            self.open, self.high, self.low, self.close
        ))

    def test_hidden_flower(self):
        self._shape_and_finite(HiddenFlowerTemplate().evaluate(self.close, self.volume))

    def test_long_short_battle(self):
        self._shape_and_finite(LongShortBattleTemplate().evaluate(
            self.high, self.low, self.close, self.volume
        ))

    def test_equal_treatment(self):
        self._shape_and_finite(EqualTreatmentTemplate().evaluate(
            self.open, self.close, self.volume
        ))

    def test_dark_flow(self):
        self._shape_and_finite(DarkFlowTemplate().evaluate(
            self.open, self.high, self.low, self.volume
        ))

    def test_boxcox_mle_preserves_shape_and_missingness(self):
        x = self.volume[:2, :2, :20].clone()
        x[0, 0, 0] = 0
        result = boxcox_grid_mle(x)
        self.assertEqual(result.shape, x.shape)
        self.assertTrue(torch.isnan(result[0, 0, 0]))


class ExplicitContractTest(unittest.TestCase):
    def test_incomplete_source_definitions_refuse_to_guess(self):
        with self.assertRaises(IncompleteDefinitionError):
            RushingForwardTemplate().evaluate(None, None, None)
        with self.assertRaises(IncompleteDefinitionError):
            WaterBoatTemplate().evaluate(None, None, None)
        with self.assertRaises(IncompleteDefinitionError):
            CooperationEffectTemplate().evaluate(None, None, None, None)


if __name__ == "__main__":
    unittest.main()

