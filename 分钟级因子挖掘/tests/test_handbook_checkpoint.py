import json
from pathlib import Path
import tempfile
import unittest

import torch

from min_gp.evaluation import WalkForwardConfig
from min_gp.factors.handbook_skeleton import (
    handbook_anchor, handbook_seed_population,
)
from min_gp.gp.handbook import HandbookFactorGP, HandbookGPConfig


def _inputs(instruments=24, days=45):
    torch.manual_seed(91)
    returns = torch.randn(instruments, days) * 0.02
    return {
        "daily_close": 100 * torch.exp(torch.randn(instruments, days) * 0.01),
        "market_close": 100 * torch.exp(torch.randn(days) * 0.01),
    }, returns


def _fitness():
    return WalkForwardConfig(
        min_train_days=10, valid_days=8, n_splits=3,
        embargo_days=1, min_cross_section=10,
        min_valid_ic_days=3, min_fold_consistency=0.0,
        min_folds=1, signal_average_days=2, max_turnover=None,
    )


class HandbookCheckpointTest(unittest.TestCase):
    def _engine(self, checkpoint, **overrides):
        context, returns = _inputs()
        values = dict(
            population_size=3, generations=2, elite=1, seed=7,
            verbose=False, baseline_name="raw_panic",
            checkpoint_path=str(checkpoint), data_fingerprint="fixture-v1",
            fitness_batch_size=2,
        )
        values.update(overrides)
        return HandbookFactorGP(
            context, returns,
            gp_config=HandbookGPConfig(**values),
            fitness_config=_fitness(),
        )

    def test_all_handbook_genome_kinds_round_trip(self):
        engine = self._engine(Path("unused.json"))
        for genome in handbook_seed_population():
            restored = engine._restore_genome(genome.to_dict())
            self.assertEqual(genome, restored)

    def test_checkpoint_restores_population_scores_and_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "handbook.json"
            first = self._engine(checkpoint)
            population = first._initial_population()
            first._fitness_cache[population[0]] = first.evaluate(population[0])
            first._checkpoint(1, population)
            expected_random = [first.rng.random() for _ in range(5)]

            resumed = self._engine(checkpoint, resume=True)
            generation, restored = resumed._resume_population()
            self.assertEqual(generation, 1)
            self.assertEqual(population, restored)
            self.assertIn(restored[0], resumed._fitness_cache)
            self.assertEqual(
                expected_random, [resumed.rng.random() for _ in range(5)]
            )

    def test_changed_data_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "handbook.json"
            first = self._engine(checkpoint)
            first._checkpoint(0, first._initial_population())
            changed = self._engine(
                checkpoint, resume=True, data_fingerprint="fixture-v2"
            )
            with self.assertRaisesRegex(ValueError, "mismatch"):
                changed._resume_population()

    def test_run_writes_a_completed_generation_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "handbook.json"
            engine = self._engine(checkpoint, generations=0)
            ranked = engine.run()
            self.assertEqual(len(ranked), 3)
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertTrue(payload["completed"])
            self.assertEqual(payload["generation"], 0)
            self.assertEqual(len(payload["population"]), 3)
            self.assertEqual(len(payload["scores"]), 3)


if __name__ == "__main__":
    unittest.main()
