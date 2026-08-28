import random
import unittest
import tempfile
from pathlib import Path

import torch

from min_gp.dsl import LeafNode, SemanticType
from min_gp.evaluation import FactorArchive, WalkForwardConfig, neutralize_factor
from min_gp.gp.daily import DAILY_TREE_OPERATORS, DailyFactorGP, DailyGPConfig
from min_gp.gp.typed_tree import (
    TypedTreeGenome, crossover_trees, mutate_tree, random_tree,
)
from min_gp.operators import build_operator_registry


class TypedTreeTest(unittest.TestCase):
    def test_generation_mutation_crossover_and_round_trip(self):
        registry = build_operator_registry()
        leaves = tuple(
            LeafNode(name, SemanticType.DAILY_RAW_FACTOR) for name in ("a", "b")
        )
        rng = random.Random(2)
        roots = [
            TypedTreeGenome(random_tree(
                registry, SemanticType.DAILY_FACTOR, leaves, 4, rng,
                DAILY_TREE_OPERATORS,
            )) for _ in range(2)
        ]
        child = crossover_trees(roots[0], roots[1], registry, rng)
        child = mutate_tree(child, registry, leaves, 2, rng, DAILY_TREE_OPERATORS)
        rebuilt = TypedTreeGenome.from_dict(child.to_dict(), registry)
        self.assertEqual(rebuilt, child)
        values = {"a": torch.randn(10, 20), "b": torch.randn(10, 20)}
        self.assertEqual(tuple(child.evaluate(values, registry).shape), (10, 20))


class NeutralizationTest(unittest.TestCase):
    @staticmethod
    def _daily_reference(factor, continuous, industry, min_cross_section):
        out = torch.full_like(factor, float("nan"))
        for day in range(factor.shape[1]):
            valid = torch.isfinite(factor[:, day])
            for exposure in continuous:
                valid &= torch.isfinite(exposure[:, day])
            valid &= industry[:, day] >= 0
            if int(valid.sum()) < min_cross_section:
                continue
            y = factor[valid, day]
            columns = [torch.ones_like(y)]
            for exposure in continuous:
                x = exposure[valid, day]
                columns.append(
                    (x - x.mean()) / x.std(unbiased=False).clamp(min=1e-8)
                )
            code = industry[valid, day]
            levels = torch.unique(code)
            columns.extend((code == level).float() for level in levels[1:])
            design = torch.stack(columns, dim=1)
            beta = torch.linalg.lstsq(design, y.unsqueeze(1)).solution[:, 0]
            out[valid, day] = y - design @ beta
        return out

    def test_batched_neutralization_matches_daily_ols_with_missing_values(self):
        torch.manual_seed(13)
        factor = torch.randn(70, 11)
        size = torch.randn(70, 11)
        industry = torch.arange(70).remainder(5).view(-1, 1).expand(-1, 11).clone()
        factor[::13, 2] = float("nan")
        size[::17, 4] = float("nan")
        industry[::19, 7] = -1
        actual = neutralize_factor(factor, (size,), industry, 20)
        expected = self._daily_reference(factor, (size,), industry, 20)
        self.assertTrue(torch.equal(torch.isnan(actual), torch.isnan(expected)))
        valid = torch.isfinite(expected)
        self.assertTrue(torch.allclose(actual[valid], expected[valid], atol=2e-5))

    def test_removes_linear_size_and_industry_exposure(self):
        torch.manual_seed(3)
        size = torch.randn(60, 8)
        industry = torch.arange(60).remainder(3).view(-1, 1).expand(-1, 8)
        factor = 2 * size + (industry == 1).float() * 3 + torch.randn(60, 8) * .01
        residual = neutralize_factor(factor, (size,), industry, 20)
        valid = torch.isfinite(residual)
        corr = torch.corrcoef(torch.stack((residual[valid], size[valid])))[0, 1]
        self.assertLess(abs(float(corr)), 1e-4)

    def test_archive_rejects_duplicate_signal(self):
        x = torch.randn(30, 20)
        archive = FactorArchive(max_abs_correlation=.8)
        self.assertTrue(archive.add("x", x)[0])
        self.assertFalse(archive.add("copy", 2 * x)[0])


class DailyGPTest(unittest.TestCase):
    def test_small_run_completes(self):
        torch.manual_seed(4)
        leaves = {"a": torch.randn(35, 90), "b": torch.randn(35, 90)}
        returns = leaves["a"] + .2 * torch.randn(35, 90)
        gp = DailyFactorGP(
            leaves, returns,
            gp_config=DailyGPConfig(
                population_size=6, generations=1, elite=2,
                max_depth=3, verbose=False,
            ),
            fitness_config=WalkForwardConfig(
            max_turnover=None,
                min_train_days=20, valid_days=20, n_splits=3,
                embargo_days=1, min_cross_section=20,
                min_valid_ic_days=5, min_fold_consistency=0,
            ),
        )
        self.assertEqual(len(gp.run()), 6)

    def test_checkpoint_can_resume(self):
        torch.manual_seed(6)
        leaves = {"a": torch.randn(30, 70), "b": torch.randn(30, 70)}
        returns = leaves["a"] + .1 * torch.randn(30, 70)
        fitness = WalkForwardConfig(
            max_turnover=None,
            min_train_days=15, valid_days=15, n_splits=3, embargo_days=1,
            min_cross_section=20, min_valid_ic_days=5,
            min_fold_consistency=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = str(Path(directory) / "checkpoint.json")
            config = DailyGPConfig(
                population_size=5, generations=1, elite=2, max_depth=3,
                verbose=False, checkpoint_path=checkpoint,
            )
            first = DailyFactorGP(leaves, returns, gp_config=config, fitness_config=fitness)
            first.run()
            self.assertTrue(Path(checkpoint).exists())
            resumed = DailyFactorGP(
                leaves, returns,
                gp_config=DailyGPConfig(**{**config.__dict__, "resume": True}),
                fitness_config=fitness,
            )
            self.assertEqual(len(resumed.run()), 5)


if __name__ == "__main__":
    unittest.main()
