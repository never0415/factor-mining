"""Numerical and structural locks for the 59 seed-tree migrations."""

import random
import unittest

import torch

from min_gp.dsl import ConstantNode, LeafNode, OperatorNode, SemanticType
from min_gp.expr import Ctx, parse
from min_gp.factors.seed_tree import SeedTreeGenome, typed_seed_population
from min_gp.operators import build_operator_registry
from min_gp.seeds import SEEDS
from min_gp.gp.seed_tree import SeedTreeFactorGP, SeedTreeGPConfig
from min_gp.evaluation import WalkForwardConfig


def _walk(node):
    yield node
    if isinstance(node, OperatorNode):
        for child in node.children:
            yield from _walk(child)


def _context(I=4, D=25, M=48):
    torch.manual_seed(123)
    close = 10 + torch.rand(I, D, M)
    open_ = close * (1 + torch.randn(I, D, M) * .001)
    high = torch.maximum(open_, close) * (1 + torch.rand(I, D, M) * .002)
    low = torch.minimum(open_, close) * (1 - torch.rand(I, D, M) * .002)
    volume = torch.rand(I, D, M) * 100 + 1
    ret = torch.full_like(close, float("nan"))
    ret[..., 1:] = close[..., 1:] / close[..., :-1] - 1
    tensors = {
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "tp": (high + low + close) / 3,
        "ret": ret, "amp": high / low - 1,
    }
    for name, probability in (
        ("is_peak", .12), ("is_ridge", .15), ("is_valley", .10),
        ("is_jump_peak", .10), ("is_jump_ridge", .13),
        ("is_amp_valley", .10),
    ):
        tensors[name] = torch.rand(I, D, M) < probability
    masks = {
        "mask_am": torch.arange(M) < M // 2,
        "mask_pm": torch.arange(M) >= M // 2,
    }
    return tensors, masks


class SeedTreeMigrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = build_operator_registry()
        cls.population = typed_seed_population(cls.registry)
        cls.by_name = {genome.name: genome for genome in cls.population}
        cls.tensors, cls.masks = _context()
        cls.context = {**cls.tensors, **cls.masks}

    def test_every_seed_is_a_registered_typed_tree(self):
        self.assertEqual(len(self.population), 59)
        self.assertEqual(set(self.by_name), set(SEEDS))
        registered = set(self.registry.names())
        for genome in self.population:
            self.assertTrue(set(genome.operator_slots) <= registered)
            self.assertTrue(all(name.startswith("seed_") for name in genome.operator_slots))

    def test_configuration_domains_and_literal_constants_are_recovered(self):
        nodes = [node for genome in self.population for node in _walk(genome.root)]
        constants = [node for node in nodes if isinstance(node, ConstantNode)]
        parameter_count = sum(len(node.params) for node in nodes if isinstance(node, OperatorNode))
        self.assertEqual(len(constants), 73)
        self.assertEqual(parameter_count, 149)
        self.assertLessEqual({node.value for node in constants}, {0, 1, 2})

    def test_all_59_anchors_match_the_old_runtime(self):
        old_context = Ctx(
            self.tensors, self.masks,
            {"I": next(iter(self.tensors.values())).shape[0],
             "NM": next(iter(self.tensors.values())).shape[-1]},
            device="cpu",
        )
        for name, expression in SEEDS.items():
            with self.subTest(factor=name):
                expected = old_context.eval(parse(expression))
                actual = self.by_name[name].evaluate(self.context, self.registry)
                self.assertEqual(actual.shape, expected.shape)
                self.assertTrue(torch.allclose(
                    actual.to(expected.dtype), expected,
                    atol=1e-5, rtol=1e-5, equal_nan=True,
                ))

    def test_all_59_trees_are_chunk_invariant(self):
        for genome in self.population:
            with self.subTest(factor=genome.name):
                whole = genome.evaluate(self.context, self.registry, chunk_rows=10_000)
                chunked = genome.evaluate(self.context, self.registry, chunk_rows=12)
                self.assertTrue(torch.allclose(
                    whole.float(), chunked.float(),
                    atol=1e-5, rtol=1e-5, equal_nan=True,
                ))

    def test_serialization_round_trip(self):
        for genome in self.population:
            restored = SeedTreeGenome.from_dict(genome.to_dict(), self.registry)
            self.assertEqual(str(restored), str(genome))

    def test_tree_mutation_and_crossover_preserve_root_type(self):
        leaves = tuple(
            LeafNode(name, SemanticType.SESSION_MASK if name.startswith("mask_")
                     else SemanticType.LEGACY_MINUTE)
            for name in sorted({field for genome in self.population for field in genome.required_fields})
        )
        left, right = self.population[0], self.population[-1]
        mutated = left.mutate(self.registry, leaves, random.Random(7), max_depth=4)
        crossed = left.crossover(right, self.registry, random.Random(11))
        self.assertEqual(mutated.root.output_type, SemanticType.LEGACY_DAILY)
        self.assertEqual(crossed.root.output_type, SemanticType.LEGACY_DAILY)

    def test_migrated_trees_are_connected_to_a_production_gp(self):
        tensors, masks = _context(I=20, D=25, M=48)
        engine = SeedTreeFactorGP(
            {**tensors, **masks}, torch.randn(20, 25),
            gp_config=SeedTreeGPConfig(
                population_size=4, generations=1, elite=1,
                verbose=False, seed=2,
            ),
            fitness_config=WalkForwardConfig(
            max_turnover=None,
                min_train_days=8, valid_days=5, n_splits=2,
                min_cross_section=10, min_valid_ic_days=2,
                min_fold_consistency=0, min_folds=1,
            ),
        )
        self.assertEqual(len(engine.run()), 4)

    def test_precise_types_flow_into_generic_seed_ops_but_not_backwards(self):
        precise = OperatorNode(
            "seed_abs_minute",
            (LeafNode("close", SemanticType.MINUTE_CLOSE),), {},
        ).bind(self.registry)
        self.assertEqual(precise.output_type, SemanticType.LEGACY_MINUTE)
        with self.assertRaises(TypeError):
            OperatorNode(
                "minute_return",
                (LeafNode("x", SemanticType.LEGACY_MINUTE),),
                {"horizon": 1},
            ).bind(self.registry)

    def test_production_pool_can_build_a_cross_family_hybrid(self):
        tensors, masks = _context(I=6, D=10, M=48)
        context = {**tensors, **masks}
        engine = SeedTreeFactorGP(
            context, torch.randn(6, 10),
            gp_config=SeedTreeGPConfig(
                population_size=4, generations=0, verbose=False, seed=9,
            ),
        )
        local = next(g for g in engine.seeds if not g.name.startswith("shared:"))
        shared = next(
            g for g in engine.seeds
            if g.name.startswith("shared:EventSkeletonGenome")
        )
        local_ops = {n.name for n in _walk(local.root) if isinstance(n, OperatorNode)}
        shared_ops = {n.name for n in _walk(shared.root) if isinstance(n, OperatorNode)}
        hybrid = None
        rng = random.Random(19)
        for _ in range(200):
            candidate = local.crossover(shared, engine.registry, rng)
            names = {n.name for n in _walk(candidate.root) if isinstance(n, OperatorNode)}
            if names & local_ops and names & shared_ops:
                hybrid = candidate
                break
        self.assertIsNotNone(hybrid)
        self.assertEqual(hybrid.evaluate(
            context, engine.registry, chunk_rows=24
        ).shape, (6, 10))
        initial = engine._initial()
        self.assertTrue(any(g.name.startswith("shared:") for g in initial))

    def test_seed_work_uses_the_persisted_runtime_calibration(self):
        tensors, masks = _context(I=6, D=10, M=48)
        engine = SeedTreeFactorGP(
            {**tensors, **masks}, torch.randn(6, 10),
            gp_config=SeedTreeGPConfig(
                population_size=2, generations=0, verbose=False,
                include_catalog_anchors=False,
            ),
        )
        estimate = engine.cost_estimate(engine.seeds[0])
        self.assertEqual(estimate.uncalibrated_cost_units, 0)
        self.assertFalse(estimate.uncalibrated_operators)
        self.assertGreater(estimate.calibrated_seconds, 0)


if __name__ == "__main__":
    unittest.main()
