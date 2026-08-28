"""Tests for the slot-based event skeleton.

The skeleton only earns its keep if it reproduces the hand-written templates
exactly and rejects illegal fills before they reach a population, so those two
properties are tested first.
"""

import unittest
import tempfile
from pathlib import Path

import torch

from min_gp.dsl import OperatorSpec, SemanticType
from min_gp.factors import ModerateRiskTemplate, WaitRescueTemplate
from min_gp.factors.event_skeleton import (
    Branch,
    EventSkeletonGenome,
    Slot,
    moderate_risk_anchor,
    slot_candidates,
    wait_rescue_anchor,
)
from min_gp.operators import build_operator_registry
from min_gp.gp.event import EventFactorGP, EventGPConfig
from min_gp.gp.event_skeleton import (
    crossover_skeleton, mutate_skeleton, random_skeleton,
)
from min_gp.evaluation import WalkForwardConfig
import random


def _minute(I=35, D=30, M=240, seed=5):
    torch.manual_seed(seed)
    volume = torch.rand(I, D, M) * 1000 + 1
    close = 10 * torch.cumprod(1 + torch.randn(I, D, M) * 0.001, dim=2)
    return {"close": close, "volume": volume}


class AnchorEquivalenceTest(unittest.TestCase):
    """Decomposing into slots must not change the factor."""

    def test_moderate_risk_anchor_matches_the_template(self):
        tensors = _minute()
        for smooth in (5, 20):
            template = ModerateRiskTemplate(smooth_window=smooth).evaluate(
                tensors["close"], tensors["volume"], 512
            )
            skeleton = moderate_risk_anchor(smooth_window=smooth).evaluate(
                tensors, 512
            )
            self.assertTrue(
                torch.allclose(
                    template, skeleton, atol=1e-6, rtol=1e-6, equal_nan=True
                ),
                f"smooth_window={smooth}",
            )

    def test_wait_rescue_anchor_matches_the_template(self):
        tensors = _minute()
        for follow in (3, 5):
            template = WaitRescueTemplate(follow_window=follow).evaluate(
                tensors["volume"], 512
            )
            skeleton = wait_rescue_anchor(follow_window=follow).evaluate(
                tensors, 512
            )
            self.assertTrue(
                torch.allclose(
                    template, skeleton, atol=1e-6, rtol=1e-6, equal_nan=True
                ),
                f"follow_window={follow}",
            )

    def test_anchors_carry_the_paper_parameters_through(self):
        tensors = _minute()
        a = moderate_risk_anchor(sigma=1.5, exclude_edges=15).evaluate(tensors, 512)
        b = ModerateRiskTemplate(sigma=1.5, exclude_edges=15).evaluate(
            tensors["close"], tensors["volume"], 512
        )
        self.assertTrue(torch.allclose(a, b, atol=1e-6, rtol=1e-6, equal_nan=True))


class SlotDerivationTest(unittest.TestCase):
    """Slot candidates come from type signatures, not a hand-kept table."""

    def test_each_slot_has_the_expected_members(self):
        registry = build_operator_registry()
        names = {
            kind: {spec.name for spec in slot_candidates(registry, kind)}
            for kind in (
                "detector", "statistic", "aggregator", "cross_section",
                "low_frequency", "combiner",
            )
        }
        self.assertIn("delta_sigma_event", names["detector"])
        self.assertIn("topk_separated_events", names["detector"])
        self.assertIn("forward_window_std", names["statistic"])
        self.assertIn("follow_ratio_series", names["statistic"])
        self.assertIn("masked_daily_mean_signal", names["aggregator"])
        self.assertIn("cross_section_identity", names["cross_section"])
        self.assertIn("mean_std_blend", names["low_frequency"])
        self.assertIn("equal_blend", names["combiner"])

    def test_spectral_operators_stay_out_of_event_slots(self):
        registry = build_operator_registry()
        detectors = {spec.name for spec in slot_candidates(registry, "detector")}
        self.assertNotIn("fft_power", detectors)
        low = {spec.name for spec in slot_candidates(registry, "low_frequency")}
        self.assertNotIn("band_power_ratio", low)

    def test_volume_only_leaf_space_excludes_close_dependent_statistics(self):
        registry = build_operator_registry()
        specs = slot_candidates(registry, "statistic", {"volume"})
        self.assertTrue(specs)
        self.assertTrue(all(
            spec.input_types[0] == SemanticType.MINUTE_VOLUME for spec in specs
        ))

    def test_registering_an_operator_widens_the_slot(self):
        registry = build_operator_registry()
        before = len(slot_candidates(registry, "cross_section"))
        registry.register(OperatorSpec(
            "cross_section_demo", (SemanticType.DAILY_RAW_FACTOR,),
            SemanticType.DAILY_RAW_FACTOR, lambda x: x,
        ))
        after = slot_candidates(registry, "cross_section")
        self.assertEqual(len(after), before + 1)
        self.assertIn("cross_section_demo", {spec.name for spec in after})

    def test_unknown_slot_name_raises(self):
        with self.assertRaises(ValueError):
            slot_candidates(build_operator_registry(), "not_a_slot")


class GenomeValidationTest(unittest.TestCase):
    def test_type_incompatible_fill_is_rejected_at_construction(self):
        anchor = wait_rescue_anchor()
        with self.assertRaises(TypeError):
            EventSkeletonGenome(
                # An FFT power spectrum cannot act as an event mask.
                detector=Slot.of("fft_power"),
                primary=anchor.primary,
            )

    def test_out_of_domain_parameter_is_rejected(self):
        anchor = wait_rescue_anchor()
        with self.assertRaises(ValueError):
            EventSkeletonGenome(
                detector=Slot.of(
                    "topk_separated_events", k=999, exclude_before=15, min_gap=5
                ),
                primary=anchor.primary,
            )

    def test_second_branch_requires_a_combiner(self):
        anchor = moderate_risk_anchor()
        with self.assertRaises(ValueError):
            EventSkeletonGenome(
                detector=anchor.detector,
                primary=anchor.primary,
                secondary=anchor.secondary,
                combiner=None,
            )

    def test_genome_is_hashable_and_round_trips(self):
        for anchor in (moderate_risk_anchor(), wait_rescue_anchor()):
            self.assertEqual(
                EventSkeletonGenome.from_dict(anchor.to_dict()), anchor
            )
            self.assertEqual(len({anchor, anchor}), 1)

    def test_required_fields_follow_the_statistic(self):
        self.assertEqual(
            moderate_risk_anchor().required_fields, ("close", "volume")
        )
        self.assertEqual(wait_rescue_anchor().required_fields, ("volume",))


class RecombinationTest(unittest.TestCase):
    """The point of the decomposition: parts cross factor boundaries."""

    def test_hybrid_of_two_reports_evaluates(self):
        tensors = _minute()
        wait, moderate = wait_rescue_anchor(), moderate_risk_anchor()
        # 待著而救's top-k detector feeding 适度冒险's forward-window volatility:
        # a factor neither report wrote.
        hybrid = EventSkeletonGenome(
            detector=wait.detector,
            primary=moderate.primary,
        )
        factor = hybrid.evaluate(tensors, 512)
        self.assertEqual(tuple(factor.shape), tensors["volume"].shape[:2])
        self.assertGreater(int(torch.isfinite(factor).sum()), 0)
        self.assertNotEqual(str(hybrid), str(wait))
        self.assertNotEqual(str(hybrid), str(moderate))

    def test_swapping_one_slot_changes_the_factor(self):
        tensors = _minute()
        anchor = wait_rescue_anchor()
        swapped = EventSkeletonGenome(
            detector=anchor.detector,
            primary=Branch(
                statistic=anchor.primary.statistic,
                aggregator=Slot.of("masked_daily_median"),
                cross_section=anchor.primary.cross_section,
                low_frequency=anchor.primary.low_frequency,
            ),
        )
        a = anchor.evaluate(tensors, 512)
        b = swapped.evaluate(tensors, 512)
        both = torch.isfinite(a) & torch.isfinite(b)
        self.assertTrue(both.any())
        self.assertFalse(torch.allclose(a[both], b[both]))

    def test_expression_string_shows_the_whole_tree(self):
        text = str(wait_rescue_anchor())
        for part in (
            "mean_std_blend", "cross_section_identity",
            "masked_daily_mean_signal", "follow_ratio_series",
            "topk_separated_events",
        ):
            self.assertIn(part, text)


class OperatorGenomeGPTest(unittest.TestCase):
    def test_event_checkpoint_rejects_changed_search_configuration(self):
        tensors = _minute(I=25, D=20, M=20)
        returns = torch.randn(25, 20)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = str(Path(directory) / "event.json")
            config = EventGPConfig(
                family="event_skeleton", population_size=4, generations=1,
                elite=1, seed=0, verbose=False, checkpoint_path=checkpoint,
            )
            first = EventFactorGP(tensors, returns, gp_config=config)
            first._checkpoint(0, first._initial_population())
            changed = EventFactorGP(
                tensors, returns,
                gp_config=EventGPConfig(**{
                    **config.__dict__, "seed": 1, "resume": True,
                }),
            )
            with self.assertRaisesRegex(ValueError, "configuration"):
                changed._resume_population()

    def test_volume_only_operator_gp_never_creates_a_close_genome(self):
        tensors = _minute(I=35, D=70, M=30)
        tensors.pop("close")
        returns = torch.randn(35, 70)
        engine = EventFactorGP(
            tensors, returns,
            gp_config=EventGPConfig(
                family="event_skeleton", population_size=5, generations=1,
                elite=2, chunk_rows=256, verbose=False,
            ),
            fitness_config=WalkForwardConfig(
            max_turnover=None,
                min_train_days=15, valid_days=15, n_splits=3,
                embargo_days=1, min_cross_section=20,
                min_valid_ic_days=5, min_fold_consistency=0,
            ),
        )
        ranked = engine.run()
        self.assertEqual(engine.baseline_genome, wait_rescue_anchor())
        self.assertTrue(all(row[1].required_fields == ("volume",) for row in ranked))

    def test_variation_changes_operators_and_remains_valid(self):
        rng = random.Random(9)
        a, b = moderate_risk_anchor(), wait_rescue_anchor()
        children = {mutate_skeleton(a, rng) for _ in range(20)}
        self.assertGreater(len(children), 1)
        for child in children:
            child.expression()
        crossover_skeleton(a, b, rng).expression()
        random_skeleton(rng).expression()

    def test_small_operator_gp_run_completes(self):
        tensors = _minute(I=35, D=90, M=40)
        torch.manual_seed(10)
        returns = torch.randn(35, 90)
        engine = EventFactorGP(
            tensors, returns,
            gp_config=EventGPConfig(
                family="event_skeleton", population_size=6, generations=1,
                elite=2, chunk_rows=256, verbose=False,
            ),
            fitness_config=WalkForwardConfig(
            max_turnover=None,
                min_train_days=20, valid_days=20, n_splits=3,
                embargo_days=1, min_cross_section=20,
                min_valid_ic_days=5, min_fold_consistency=0,
            ),
        )
        ranked = engine.run()
        self.assertEqual(len(ranked), 6)
        self.assertTrue(all(isinstance(row[1], EventSkeletonGenome) for row in ranked))


if __name__ == "__main__":
    unittest.main()
