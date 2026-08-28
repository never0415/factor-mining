"""The turnover gate, and the rank-smoothing operator that lets a genome pass it.

Across 76 hold-out candidates corr(turnover, net return) was -0.62 and every
candidate above 0.7 lost money, so rebalance cost — not rank quality — was what
separated winners from losers. Turnover is already charged inside
``net_long_short``, but that objective trades against IC, so a genome could buy
rank quality with cost and still reach the front. These tests pin the gate that
stops it and the operator that gives the search a way to comply.
"""

import unittest

import torch

from min_gp.evaluation import WalkForwardConfig, evaluate_incremental_fitness
from min_gp.evaluation.incremental import net_long_short_return
from min_gp.operators import build_operator_registry
from min_gp.operators.temporal import rank_smooth_daily


def _cfg(**overrides):
    base = dict(
        min_train_days=20, valid_days=20, n_splits=4, embargo_days=0,
        min_cross_section=20, min_valid_ic_days=5, cost_bps=0.0,
        min_fold_consistency=0.0, min_folds=1,
    )
    base.update(overrides)
    return WalkForwardConfig(**base)


def _churning(I=60, D=120, seed=0):
    """A factor whose ranks are redrawn every date: near-full replacement."""
    torch.manual_seed(seed)
    factor = torch.randn(I, D)
    returns = factor * 0.3 + torch.randn(I, D) * 0.02
    return factor, returns


def _persistent(I=60, D=120, seed=0):
    """A factor that barely moves: almost no rebalancing."""
    torch.manual_seed(seed)
    base = torch.randn(I, 1)
    factor = base + torch.randn(I, D) * 0.01
    returns = factor * 0.3 + torch.randn(I, D) * 0.02
    return factor, returns


class TurnoverMeasurementTest(unittest.TestCase):
    def test_full_replacement_measures_two(self):
        factor, returns = _churning()
        _, turnover = net_long_short_return(factor, returns, 1, 0.2, 0.0, 20, 1)
        # Gross exposure is 2 (one long leg, one short leg), so a complete
        # replacement measures 2.0, not 1.0.
        self.assertGreater(turnover, 1.4)
        self.assertLessEqual(turnover, 2.0)

    def test_a_static_factor_measures_near_zero(self):
        factor, returns = _persistent()
        _, turnover = net_long_short_return(factor, returns, 1, 0.2, 0.0, 20, 1)
        self.assertLess(turnover, 0.1)

    def test_the_first_rebalance_is_charged(self):
        """Building the book from cash is a real trade, not a free position."""
        factor, returns = _persistent()
        _, turnover = net_long_short_return(factor, returns, 1, 0.2, 0.0, 20, 1)
        self.assertGreater(turnover, 0.0)


class TurnoverGateTest(unittest.TestCase):
    def test_a_churning_factor_is_rejected(self):
        factor, returns = _churning()
        score = evaluate_incremental_fitness(
            factor, torch.randn_like(factor), returns, complexity=1,
            cfg=_cfg(max_turnover=0.5),
        )
        self.assertFalse(score.valid)
        self.assertEqual(score.objectives, (-1e9, -1e9, -1e9, -1e9, -1.0))

    def test_the_same_factor_survives_without_the_gate(self):
        """Guards the gate itself: the rejection must come from turnover."""
        factor, returns = _churning()
        score = evaluate_incremental_fitness(
            factor, torch.randn_like(factor), returns, complexity=1,
            cfg=_cfg(max_turnover=None),
        )
        self.assertTrue(score.valid)
        self.assertGreater(score.turnover, 0.5)

    def test_a_persistent_factor_passes(self):
        factor, returns = _persistent()
        score = evaluate_incremental_fitness(
            factor, torch.randn_like(factor), returns, complexity=1,
            cfg=_cfg(max_turnover=0.5),
        )
        self.assertTrue(score.valid)
        self.assertLess(score.turnover, 0.5)

    def test_turnover_is_reported_on_the_fitness(self):
        factor, returns = _persistent()
        score = evaluate_incremental_fitness(
            factor, torch.randn_like(factor), returns, complexity=1,
            cfg=_cfg(max_turnover=None),
        )
        self.assertTrue(0.0 < score.turnover <= 2.0)


class RankSmoothTest(unittest.TestCase):
    """Portfolio turnover follows rank changes, so damp ranks, not levels."""

    def test_rank_smoothing_lowers_turnover(self):
        factor, returns = _churning()
        _, raw = net_long_short_return(factor, returns, 1, 0.2, 0.0, 20, 1)
        smoothed = rank_smooth_daily(factor, window=20)
        _, damped = net_long_short_return(smoothed, returns, 1, 0.2, 0.0, 20, 1)
        # Measured: 1.60 unsmoothed against 0.37 at a twenty-day window.
        self.assertLess(damped, raw * 0.35)

    def test_a_longer_window_damps_more(self):
        factor, _ = _churning()
        turnovers = []
        for window in (5, 20, 40):
            smoothed = rank_smooth_daily(factor, window=window)
            _, value = net_long_short_return(
                smoothed, torch.randn_like(factor), 1, 0.2, 0.0, 20, 1
            )
            turnovers.append(value)
        self.assertEqual(turnovers, sorted(turnovers, reverse=True))

    def test_window_one_is_the_plain_cross_sectional_rank(self):
        from min_gp.numeric.ranking import cross_section_rank
        factor, _ = _churning()
        self.assertTrue(torch.allclose(
            rank_smooth_daily(factor, window=1),
            cross_section_rank(factor), equal_nan=True,
        ))

    def test_it_is_registered_with_history_and_cross_section_scope(self):
        registry = build_operator_registry()
        spec = registry.get("rank_smooth_daily")
        self.assertTrue(spec.needs_full_cross_section)
        self.assertTrue(spec.needs_history)
        self.assertEqual(spec.resolved_history_days({"window": 40}), 39)

    def test_it_can_fill_the_low_frequency_slot(self):
        from min_gp.factors.event_skeleton import slot_candidates
        registry = build_operator_registry()
        names = {s.name for s in slot_candidates(registry, "low_frequency")}
        self.assertIn("rank_smooth_daily", names)


if __name__ == "__main__":
    unittest.main()
