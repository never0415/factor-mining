"""Regression tests for the incremental-fitness hard gates and rank residual."""

from dataclasses import replace
import unittest

import torch

from min_gp.evaluation import WalkForwardConfig, evaluate_incremental_fitness
from min_gp.evaluation.incremental import _mean_ic, net_long_short_return


def _cfg(**overrides) -> WalkForwardConfig:
    base = dict(
        min_train_days=20, valid_days=20, n_splits=4, embargo_days=0,
        min_cross_section=20, min_valid_ic_days=5, cost_bps=0.0,
        min_fold_consistency=0.75, min_folds=3,
        # Synthetic factors reshuffle their ranks every period, so the turnover
        # gate would reject them for a reason unrelated to what is under test.
        max_turnover=None,
    )
    base.update(overrides)
    return WalkForwardConfig(**base)


class RankResidualTest(unittest.TestCase):
    """A monotone transform of the anchor carries no incremental information."""

    def setUp(self):
        torch.manual_seed(11)
        I, D = 60, 120
        self.anchor = torch.randn(I, D)
        # Strictly increasing map: identical cross-sectional ranks, very
        # different levels. Any incremental IC here is an artefact.
        self.candidate = self.anchor ** 3
        self.returns = self.anchor + 0.3 * torch.randn(I, D)

    def test_rank_residual_reports_no_incremental_signal(self):
        score = evaluate_incremental_fitness(
            self.candidate, self.anchor, self.returns, complexity=1,
            cfg=_cfg(rank_residual=True),
        )
        self.assertTrue(score.valid)
        self.assertGreater(score.robust_ic, 0.5)
        self.assertLess(abs(score.incremental_ic), 0.05)

    def test_level_residual_fabricates_incremental_signal(self):
        """The true incremental information is exactly zero; levels report ~0.48.

        The sign of the artefact depends on the transform, so the magnitude is
        what matters: a linear fit cannot absorb a monotone non-linear anchor,
        and whatever it leaves behind is graded as new information.
        """
        rank_score = evaluate_incremental_fitness(
            self.candidate, self.anchor, self.returns, complexity=1,
            cfg=_cfg(rank_residual=True),
        )
        level_score = evaluate_incremental_fitness(
            self.candidate, self.anchor, self.returns, complexity=1,
            cfg=_cfg(rank_residual=False),
        )
        self.assertGreater(abs(level_score.incremental_ic), 0.3)
        self.assertGreater(
            abs(level_score.incremental_ic), 10 * abs(rank_score.incremental_ic)
        )

    def test_incremental_ic_never_exceeds_own_ic(self):
        score = evaluate_incremental_fitness(
            self.candidate, self.anchor, self.returns, complexity=1,
            cfg=_cfg(rank_residual=True),
        )
        self.assertLessEqual(abs(score.incremental_ic), abs(score.robust_ic))


class FoldConsistencyGateTest(unittest.TestCase):
    def _sign_flipping_case(self):
        torch.manual_seed(3)
        I, D = 60, 120
        factor = torch.randn(I, D)
        noise = 0.2 * torch.randn(I, D)
        returns = factor + noise
        # Flip the relationship halfway through: early folds agree with the
        # paper direction, later folds contradict it.
        returns[:, D // 2:] = -factor[:, D // 2:] + noise[:, D // 2:]
        return factor, returns

    def test_sign_flipping_factor_is_rejected(self):
        factor, returns = self._sign_flipping_case()
        score = evaluate_incremental_fitness(
            factor, torch.randn_like(factor), returns, complexity=1,
            cfg=_cfg(),
        )
        self.assertFalse(score.valid)
        # Worst on every axis, novelty included: a rejected genome whose pool
        # correlation defaulted to zero would be unbeatable on that objective
        # and would land on the Pareto front.
        self.assertEqual(score.objectives, (-1e9, -1e9, -1e9, -1e9, -1.0))


    def test_loose_gate_would_have_accepted_it(self):
        """Guards the gate itself: without it the genome survives."""
        factor, returns = self._sign_flipping_case()
        score = evaluate_incremental_fitness(
            factor, torch.randn_like(factor), returns, complexity=1,
            cfg=_cfg(min_fold_consistency=0.0),
        )
        self.assertTrue(score.valid)
        self.assertLess(score.fold_consistency, 0.75)

    def test_too_few_usable_folds_is_rejected(self):
        torch.manual_seed(5)
        factor = torch.randn(60, 120)
        returns = factor + 0.2 * torch.randn(60, 120)
        score = evaluate_incremental_fitness(
            factor, torch.randn_like(factor), returns, complexity=1,
            cfg=_cfg(min_folds=99),
        )
        self.assertFalse(score.valid)


class HoldingPeriodTest(unittest.TestCase):
    def test_sparse_weekly_columns_equal_their_compressed_series(self):
        torch.manual_seed(21)
        factor = torch.randn(30, 8)
        returns = torch.full_like(factor, float("nan"))
        eligible = torch.tensor([0, 2, 5, 7])
        returns[:, eligible] = torch.randn(30, len(eligible))
        sparse_ic = _mean_ic(factor, returns, 20)
        dense_ic = _mean_ic(factor[:, eligible], returns[:, eligible], 20)
        self.assertAlmostEqual(sparse_ic[0], dense_ic[0], places=7)
        self.assertEqual(sparse_ic[1], dense_ic[1])
        sparse_net, _ = net_long_short_return(
            factor, returns, 1, .2, 30, 20, 1
        )
        dense_net, _ = net_long_short_return(
            factor[:, eligible], returns[:, eligible], 1, .2, 30, 20, 1
        )
        self.assertAlmostEqual(sparse_net, dense_net, places=7)

    def test_multi_day_labels_are_only_scored_on_rebalance_dates(self):
        factor = torch.tensor([[2., -2., 2., -2.], [-2., 2., -2., 2.]])
        returns = torch.tensor([[.1, .1, .1, .1], [-.1, -.1, -.1, -.1]])
        daily, _ = net_long_short_return(factor, returns, 1, 0.5, 0, 2, 1)
        two_day, _ = net_long_short_return(factor, returns, 1, 0.5, 0, 2, 2)
        self.assertGreater(two_day, daily)
        self.assertAlmostEqual(two_day, 0.2, places=6)

    def test_thirty_basis_points_matches_fifteen_per_side(self):
        factor = torch.tensor([[1.0], [-1.0]])
        returns = torch.zeros_like(factor)
        net, _ = net_long_short_return(
            factor, returns, 1, 0.5, 30.0, 2, 1
        )
        # Long +1 and short -1 trade two units from cash. At 15 bp per
        # one-way unit the total opening cost is 30 bp of portfolio capital.
        self.assertAlmostEqual(net, -0.003, places=7)


class DiscoveryDirectionTest(unittest.TestCase):
    def test_negative_factor_is_retained_and_direction_is_settled(self):
        torch.manual_seed(13)
        factor = torch.randn(60, 120)
        returns = -factor + .1 * torch.randn_like(factor)
        score = evaluate_incremental_fitness(
            factor, torch.randn_like(factor), returns, 1,
            _cfg(direction_mode="discovery"),
        )
        self.assertTrue(score.valid)
        self.assertEqual(score.direction, -1)
        self.assertGreater(score.robust_ic, 0)


if __name__ == "__main__":
    unittest.main()
