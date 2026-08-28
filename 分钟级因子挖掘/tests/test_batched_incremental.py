import math
import unittest

import torch

from min_gp.evaluation.batched_incremental import BatchedIncrementalEvaluator
from min_gp.evaluation.incremental import (
    WalkForwardConfig, evaluate_incremental_fitness,
)


def _assert_score_close(test, expected, actual, places=6):
    test.assertEqual(expected.valid, actual.valid)
    test.assertEqual(expected.direction, actual.direction)
    for name in (
        "robust_ic", "incremental_ic", "net_long_short", "complexity",
        "fold_consistency", "coverage", "turnover",
    ):
        left, right = getattr(expected, name), getattr(actual, name)
        if math.isnan(left):
            test.assertTrue(math.isnan(right), name)
        else:
            test.assertAlmostEqual(left, right, places=places, msg=name)


class BatchedIncrementalEquivalenceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260824)
        instruments, days = 47, 160
        self.baseline = torch.randn(instruments, days)
        self.candidates = [
            self.baseline * 0.25 + torch.randn(instruments, days) * 0.8,
            torch.randn(instruments, days),
            torch.randn(instruments, days),
        ]
        self.returns = torch.full((instruments, days), float("nan"))
        for day in range(4, days, 5):
            self.returns[:, day] = (
                0.04 * self.candidates[0][:, day]
                + torch.randn(instruments) * 0.03
            )
        self.candidates[1][::7, ::5] = float("nan")
        self.candidates[2][:, ::10] = 1.0
        self.baseline[::11, ::5] = float("nan")

    def _config(self, **overrides):
        values = dict(
            min_train_days=50, valid_days=30, n_splits=3,
            embargo_days=5, min_cross_section=20,
            min_valid_ic_days=5, cost_bps=30, quantile=0.2,
            holding_period=1, signal_average_days=5,
            direction_mode="discovery", min_fold_consistency=0.0,
            min_folds=2, max_turnover=None, rank_residual=True,
        )
        values.update(overrides)
        return WalkForwardConfig(**values)

    def _compare(self, cfg, batch_size):
        expected = [
            evaluate_incremental_fitness(
                candidate, self.baseline, self.returns, index + 1, cfg
            )
            for index, candidate in enumerate(self.candidates)
        ]
        evaluator = BatchedIncrementalEvaluator(
            self.baseline, self.returns, cfg
        )
        prepared = [
            evaluator.prepare_candidate(candidate).cpu()
            for candidate in self.candidates
        ]
        actual = evaluator.evaluate_batch(
            prepared, (1, 2, 3), batch_size=batch_size
        )
        for left, right in zip(expected, actual):
            _assert_score_close(self, left, right)
        return actual

    def test_discovery_rank_residual_matches_the_scalar_reference(self):
        self._compare(self._config(), batch_size=2)

    def test_paper_level_residual_matches_the_scalar_reference(self):
        self._compare(self._config(
            direction_mode="paper", paper_direction=-1,
            rank_residual=False,
        ), batch_size=3)

    def test_sparse_labels_keep_fold_relative_holding_period(self):
        self._compare(self._config(holding_period=2), batch_size=2)

    def test_batch_size_and_candidate_order_do_not_change_scores(self):
        cfg = self._config()
        one = self._compare(cfg, batch_size=1)
        evaluator = BatchedIncrementalEvaluator(
            self.baseline, self.returns, cfg
        )
        order = (2, 0, 1)
        prepared = [
            evaluator.prepare_candidate(self.candidates[index]).cpu()
            for index in order
        ]
        shuffled = evaluator.evaluate_batch(
            prepared, tuple(index + 1 for index in order), batch_size=3
        )
        restored = [None] * 3
        for source, score in zip(order, shuffled):
            restored[source] = score
        for left, right in zip(one, restored):
            _assert_score_close(self, left, right)

    def test_turnover_gate_has_the_same_decision(self):
        cfg = self._config(max_turnover=0.5)
        scores = self._compare(cfg, batch_size=2)
        self.assertTrue(all(not score.valid for score in scores))

    def test_progress_reports_each_fitness_chunk(self):
        cfg = self._config()
        evaluator = BatchedIncrementalEvaluator(
            self.baseline, self.returns, cfg
        )
        prepared = [
            evaluator.prepare_candidate(candidate).cpu()
            for candidate in self.candidates
        ]
        updates = []
        evaluator.evaluate_batch(
            prepared, (1, 2, 3), batch_size=2,
            progress=lambda done, total: updates.append((done, total)),
        )
        self.assertEqual(updates, [(2, 3), (3, 3)])


if __name__ == "__main__":
    unittest.main()
