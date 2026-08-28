"""Preflight must mirror the fitness layer's fold-skipping semantics."""

from datetime import date, timedelta
from pathlib import Path
import unittest

from min_gp.evaluation import DEFAULT_COST_BPS, WalkForwardConfig
from min_gp.preflight import assess_folds, label_day_flags


def _dates(count):
    start = date(2024, 1, 1)
    return tuple(str(start + timedelta(days=index)) for index in range(count))


class FoldFeasibilityTest(unittest.TestCase):
    def test_one_thin_extra_fold_does_not_reject_three_scorable_folds(self):
        cfg = WalkForwardConfig(
            min_train_days=5, valid_days=4, n_splits=4, embargo_days=0,
            min_valid_ic_days=4, min_folds=3, holding_period=1,
            direction_mode="paper",
        )
        report = assess_folds(_dates(21), "daily", cfg)
        self.assertEqual(report.splits, 4)
        self.assertEqual(report.min_valid_label_days, 3)
        self.assertEqual(report.scorable_folds, 3)
        self.assertTrue(report.ok)

    def test_discovery_requires_enough_training_labels(self):
        discovery = WalkForwardConfig(
            min_train_days=3, valid_days=5, n_splits=3, embargo_days=0,
            min_valid_ic_days=5, min_folds=3, holding_period=1,
            direction_mode="discovery",
        )
        paper = WalkForwardConfig(
            min_train_days=3, valid_days=5, n_splits=3, embargo_days=0,
            min_valid_ic_days=5, min_folds=3, holding_period=1,
            direction_mode="paper",
        )
        self.assertFalse(assess_folds(_dates(19), "daily", discovery).ok)
        self.assertTrue(assess_folds(_dates(19), "daily", paper).ok)

    def test_daily_horizon_and_signal_warmup_remove_unusable_dates(self):
        cfg = WalkForwardConfig(
            holding_period=5, signal_average_days=3,
        )
        flags = label_day_flags(_dates(10), "daily", cfg)
        self.assertEqual(flags, [False, False, True, True, True] + [False] * 5)

    def test_core_transaction_cost_default_is_thirty_basis_points(self):
        self.assertEqual(DEFAULT_COST_BPS, 30.0)
        self.assertEqual(WalkForwardConfig().cost_bps, 30.0)


class CliOrderingTest(unittest.TestCase):
    def test_preflight_runs_before_expensive_factor_inputs_are_loaded(self):
        root = Path(__file__).resolve().parents[1]
        cases = (
            ("event_gp.py", "check_or_exit(dates", "build_minute_slice("),
            ("dripping_gp.py", "check_or_exit(dates", "build_volume_slice("),
            ("daily_gp.py", "check_or_exit(dates", "load_daily_factor_leaves("),
        )
        for filename, check, expensive in cases:
            with self.subTest(filename=filename):
                source = (root / filename).read_text(encoding="utf-8")
                self.assertLess(source.index(check), source.index(expensive))


if __name__ == "__main__":
    unittest.main()
