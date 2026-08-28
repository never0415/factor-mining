"""Fine-grained Climb-Mountain reproduction and variation tests."""

from dataclasses import replace
import random
import unittest

import torch

from min_gp.factors.climb_skeleton import (
    ClimbMountainSkeletonGenome, climb_mountain_anchor,
)
from min_gp.factors.event_skeleton import Slot
from min_gp.factors.handbook import ClimbMountainTemplate
from min_gp.factors.handbook_skeleton import (
    HandbookSkeletonGenome, handbook_anchor, handbook_seed_population,
)
from min_gp.gp.climb import (
    crossover_climb_genomes, mutate_climb_genome, random_climb_genome,
)
from min_gp.gp.handbook import HandbookFactorGP, HandbookGPConfig
from min_gp.operators import build_operator_registry


def _minute(I=7, D=24, M=48, seed=51):
    torch.manual_seed(seed)
    close = 10 * torch.cumprod(
        1 + torch.randn(I, D, M) * 0.001, dim=2
    )
    open_ = close * (1 + torch.randn_like(close) * 0.0002)
    return {
        "open": open_,
        "high": torch.maximum(open_, close) * 1.001,
        "low": torch.minimum(open_, close) * 0.999,
        "close": close,
    }


class AnchorTest(unittest.TestCase):
    def test_anchor_matches_template_for_parameters_and_date_chunks(self):
        context = _minute()
        for window, smooth in ((3, 5), (5, 10)):
            expected = ClimbMountainTemplate(window, smooth).evaluate(
                context["open"], context["high"],
                context["low"], context["close"],
            )
            actual = climb_mountain_anchor(window, smooth).evaluate(
                context, chunk_rows=14
            )
            self.assertTrue(torch.allclose(
                expected, actual, atol=1e-6, rtol=1e-6, equal_nan=True
            ))

    def test_handbook_anchor_and_seed_use_decomposed_genome(self):
        anchor = handbook_anchor("climb_mountain")
        self.assertIsInstance(anchor, ClimbMountainSkeletonGenome)
        self.assertTrue(any(
            isinstance(genome, ClimbMountainSkeletonGenome)
            for genome in handbook_seed_population()
        ))
        self.assertEqual(len(anchor.operator_slots), 7)
        self.assertNotIn("handbook_climb_mountain", anchor.operator_slots)
        self.assertIn("conditional_covariance", anchor.operator_slots)

    def test_round_trip_through_compatibility_loader(self):
        anchor = climb_mountain_anchor()
        self.assertEqual(
            HandbookSkeletonGenome.from_dict(anchor.to_dict()), anchor
        )

    def test_incompatible_ratio_operator_is_rejected(self):
        with self.assertRaises(TypeError):
            replace(
                climb_mountain_anchor(),
                ratio=Slot.of("close_minute_return", horizon=1),
            )


class VariationTest(unittest.TestCase):
    def test_mutation_changes_internal_slots(self):
        registry = build_operator_registry()
        rng = random.Random(52)
        anchor = climb_mountain_anchor()
        variants = {
            mutate_climb_genome(anchor, rng, registry) for _ in range(40)
        }
        self.assertGreater(len(variants), 5)
        self.assertTrue(any(
            value.operator_slots != anchor.operator_slots for value in variants
        ))
        for value in variants:
            value.expression(registry)
            self.assertNotIn("handbook_climb_mountain", value.operator_slots)

    def test_random_crossover_is_typed_and_evaluable(self):
        registry = build_operator_registry()
        rng = random.Random(53)
        left = random_climb_genome(rng, registry)
        right = random_climb_genome(rng, registry)
        child = crossover_climb_genomes(left, right, rng)
        child.expression(registry)
        factor = child.evaluate(_minute(I=4, D=12), registry, chunk_rows=8)
        self.assertEqual(factor.shape, (4, 12))

    def test_handbook_gp_supports_an_ohlc_only_climb_island(self):
        context = _minute(I=10, D=20)
        engine = HandbookFactorGP(
            context, torch.randn(10, 20),
            gp_config=HandbookGPConfig(
                population_size=3, generations=0, elite=1, verbose=False,
            ),
        )
        population = engine._initial_population()
        self.assertTrue(population)
        self.assertTrue(all(
            isinstance(genome, ClimbMountainSkeletonGenome)
            for genome in population
        ))


class CostTest(unittest.TestCase):
    def test_anchor_is_calibrated_and_chunking_reduces_peak_not_runtime_work(self):
        genome = climb_mountain_anchor()
        target = {"I": 1100, "D": 1700, "M": 240}
        whole = genome.cost_estimate(target)
        chunked = genome.cost_estimate(target, chunk_rows=4096)
        self.assertTrue(whole.fully_calibrated)
        self.assertTrue(chunked.peak_fully_calibrated)
        self.assertAlmostEqual(
            whole.calibrated_seconds, chunked.calibrated_seconds, places=9
        )
        self.assertLess(
            chunked.calibrated_peak_bytes, whole.calibrated_peak_bytes
        )
        self.assertLess(chunked.calibrated_peak_bytes, 2 * 1024**3)

    def test_non_anchor_parameter_uses_the_full_domain_calibration(self):
        variant = climb_mountain_anchor(window=3)
        estimate = variant.cost_estimate({"I": 150, "D": 120, "M": 240})
        self.assertTrue(estimate.fully_calibrated)
        self.assertNotIn(
            "rolling_ohlc_dispersion", estimate.uncalibrated_operators
        )


if __name__ == "__main__":
    unittest.main()
