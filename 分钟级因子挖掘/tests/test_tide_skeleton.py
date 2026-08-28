"""Fine-grained Complete-Tide reproduction, variation and cost controls."""

from dataclasses import replace
import random
import unittest

import torch

from min_gp.dsl import (
    ExecutionBudget, LeafNode, OperatorNode, SemanticType,
    estimate_expression_cost,
)
from min_gp.factors.event_skeleton import Slot
from min_gp.factors.handbook import CompleteTideTemplate
from min_gp.factors.handbook_skeleton import (
    HandbookSkeletonGenome, handbook_anchor, handbook_seed_population,
)
from min_gp.factors.tide_skeleton import (
    CompleteTideSkeletonGenome, complete_tide_anchor,
)
from min_gp.gp.tide import (
    crossover_tide_genomes, mutate_tide_genome, random_tide_genome,
)
from min_gp.gp.handbook import HandbookFactorGP, HandbookGPConfig
from min_gp.operators import build_operator_registry


def _minute(I=7, D=24, M=48, seed=41):
    torch.manual_seed(seed)
    close = 10 * torch.cumprod(
        1 + torch.randn(I, D, M) * 0.001, dim=2
    )
    volume = torch.rand(I, D, M) * 1000 + 1
    return {"close": close, "volume": volume}


class AnchorTest(unittest.TestCase):
    def test_anchor_is_exact_for_multiple_parameter_sets_and_chunks(self):
        context = _minute()
        for neighborhood, edges, smooth in ((5, 0, 5), (9, 15, 10)):
            expected = CompleteTideTemplate(
                neighborhood, edges, smooth
            ).evaluate(context["close"], context["volume"])
            genome = complete_tide_anchor(neighborhood, edges, smooth)
            actual = genome.evaluate(context, chunk_rows=14)
            self.assertTrue(torch.allclose(
                expected, actual, atol=1e-6, rtol=1e-6, equal_nan=True
            ))

    def test_handbook_anchor_and_catalog_seed_use_the_decomposed_genome(self):
        anchor = handbook_anchor("complete_tide")
        self.assertIsInstance(anchor, CompleteTideSkeletonGenome)
        self.assertIsInstance(handbook_seed_population()[0], CompleteTideSkeletonGenome)
        self.assertNotIn("handbook_complete_tide", anchor.operator_slots)
        self.assertEqual(len(anchor.operator_slots), 15)

    def test_round_trip_remains_compatible_with_handbook_loader(self):
        anchor = complete_tide_anchor()
        restored = HandbookSkeletonGenome.from_dict(anchor.to_dict())
        self.assertEqual(restored, anchor)

    def test_index_roles_reject_a_right_locator_in_the_left_slot(self):
        anchor = complete_tide_anchor()
        with self.assertRaises(TypeError):
            replace(
                anchor,
                left_locator=Slot.of(
                    "locate_right_valley", exclude_edges=15
                ),
            )


class VariationTest(unittest.TestCase):
    def test_mutation_changes_internal_operators_not_a_whole_core(self):
        registry = build_operator_registry()
        rng = random.Random(42)
        anchor = complete_tide_anchor()
        variants = {mutate_tide_genome(anchor, rng, registry) for _ in range(50)}
        self.assertGreater(len(variants), 5)
        self.assertTrue(any(
            genome.operator_slots != anchor.operator_slots for genome in variants
        ))
        for genome in variants:
            genome.expression(registry)
            self.assertNotIn("handbook_complete_tide", genome.operator_slots)

    def test_random_and_crossover_genomes_are_typed_and_evaluable(self):
        registry = build_operator_registry()
        rng = random.Random(43)
        left = random_tide_genome(rng, registry)
        right = random_tide_genome(rng, registry)
        child = crossover_tide_genomes(left, right, rng)
        child.expression(registry)
        factor = child.evaluate(_minute(I=4, D=12), registry, chunk_rows=8)
        self.assertEqual(factor.shape, (4, 12))


class CostControlTest(unittest.TestCase):
    def test_user_benchmark_extrapolates_hidden_flower_to_full_grid(self):
        registry = build_operator_registry()
        root, _ = handbook_anchor("hidden_flower").expression(registry)
        estimate = estimate_expression_cost(
            root, registry, {"I": 1100, "D": 1700, "M": 240, "P": 7}
        )
        # The full-domain benchmark takes the slowest OLS lag profile rather
        # than the former single-profile measurement, then adds every
        # decomposed downstream part. Minute leaves at this shape do not fit a
        # single chunk, so the reported figure also carries the halo the date
        # chunks recompute -- assert the two factors separately so a change to
        # either one names itself.
        # The halo factor is structural — chunk size and history window are
        # both determined by the shape — so it is pinned exactly.
        self.assertAlmostEqual(estimate.halo_amplification, 1.25, delta=0.01)
        # The seconds are not: they are re-derived from whatever the machine
        # measured the last time the archive was rebuilt, and a re-benchmark
        # moved this figure 13% without anything about the factor changing.
        # What the test is here to defend is the order of magnitude — this is
        # an hours-long factor, not a minutes-long one — and the rejection that
        # follows from it.
        base = estimate.calibrated_seconds / estimate.halo_amplification
        self.assertGreater(base, 2000.0)
        self.assertLess(base, 6000.0)
        self.assertTrue(estimate.fully_calibrated)
        self.assertFalse(ExecutionBudget(
            max_estimated_seconds=60, allow_uncalibrated=True
        ).accepts(estimate))

    def test_cost_estimator_counts_a_shared_dag_node_once(self):
        registry = build_operator_registry()
        raw = LeafNode("raw", SemanticType.DAILY_RAW_FACTOR)
        shared = OperatorNode("daily_abs", (raw,)).bind(registry)
        root = OperatorNode("daily_add", (shared, shared)).bind(registry)
        estimate = estimate_expression_cost(root, registry, {"I": 10, "D": 20})
        self.assertEqual(
            estimate.cost_units,
            registry.get("daily_abs").cost + registry.get("daily_add").cost,
        )
        self.assertEqual(root.complexity_with(registry), estimate.cost_units)

    def test_static_cost_unit_budget_handles_unmeasured_new_parts(self):
        anchor_estimate = complete_tide_anchor().cost_estimate(
            {"I": 150, "D": 120, "M": 240}
        )
        self.assertTrue(anchor_estimate.fully_calibrated)
        self.assertIsNotNone(anchor_estimate.calibrated_peak_bytes)
        variant = replace(
            complete_tide_anchor(),
            activity=Slot.of("rolling_volume_std", neighborhood=9),
        )
        root, registry = variant.expression()
        registry.replace(replace(
            registry.get("rolling_volume_std"), calibration=None
        ))
        estimate = estimate_expression_cost(
            root, registry, {"I": 1100, "D": 1700, "M": 240}
        )
        self.assertIn("rolling_volume_std", estimate.uncalibrated_operators)
        self.assertTrue(ExecutionBudget(
            max_cost_units=estimate.cost_units,
            allow_uncalibrated=True,
        ).accepts(estimate))
        self.assertFalse(ExecutionBudget(
            max_cost_units=estimate.cost_units - 1,
            allow_uncalibrated=True,
        ).accepts(estimate))

    def test_handbook_gp_rejects_expensive_core_before_evaluation(self):
        context = _minute(I=10, D=20)
        engine = HandbookFactorGP(
            context, torch.randn(10, 20),
            gp_config=HandbookGPConfig(
                population_size=2, generations=0, elite=1,
                verbose=False, max_estimated_seconds=0.01,
                allow_uncalibrated_cost=True,
            ),
        )
        hidden = handbook_anchor("hidden_flower")
        self.assertGreater(
            engine.cost_estimate(hidden).calibrated_seconds, 0.01
        )
        self.assertFalse(engine._admissible(hidden))
        self.assertFalse(engine.evaluate(hidden).valid)


if __name__ == "__main__":
    unittest.main()
