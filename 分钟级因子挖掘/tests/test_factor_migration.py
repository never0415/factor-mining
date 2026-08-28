"""Completeness and anchor-equivalence tests for the unified factor catalog."""

from collections import Counter
import random
import unittest

import torch

from min_gp.evaluation import WalkForwardConfig
from min_gp.factors.catalog import (
    FactorCatalogEntry, build_factor_catalog, directed_recombination_audit,
    is_fine_grained,
    migration_audit, migration_progress, recombination_components,
)
from min_gp.dsl import LeafNode, OperatorNode, SemanticType
from min_gp.factors.dripping_skeleton import (
    DrippingSkeletonGenome, dripping_stone_anchor,
)
from min_gp.factors.dripping_stone import DrippingStoneTemplate
from min_gp.factors.handbook import (
    ClimbMountainTemplate, CompleteTideTemplate, CooperationEffectTemplate,
    DarkFlowTemplate, EqualTreatmentTemplate, HiddenFlowerTemplate,
    LongShortBattleTemplate, RawPanicTemplate, RushingForwardTemplate,
    WaterBoatTemplate,
)
from min_gp.factors.handbook_skeleton import (
    HandbookSkeletonGenome, handbook_anchor,
)
from min_gp.factors.tide_skeleton import CompleteTideSkeletonGenome
from min_gp.factors.climb_skeleton import ClimbMountainSkeletonGenome
from min_gp.factors.handbook_composed import ComposedHandbookGenome
from min_gp.gp.handbook import (
    HandbookFactorGP, HandbookGPConfig, mutate_handbook_genome,
)
from min_gp.operators import build_operator_registry
from min_gp.seeds import SEEDS


def _minute(I=8, D=25, M=48):
    torch.manual_seed(21)
    close = 10 * torch.cumprod(1 + torch.randn(I, D, M) * .001, 2)
    open_ = close * (1 + torch.randn_like(close) * .0002)
    high = torch.maximum(open_, close) * 1.001
    low = torch.minimum(open_, close) * .999
    volume = torch.rand(I, D, M) * 1000 + 1
    return dict(open=open_, high=high, low=low, close=close, volume=volume)


class MigrationAuditTest(unittest.TestCase):
    def test_every_known_factor_has_leaves_registered_ops_and_slots(self):
        entries = build_factor_catalog()
        self.assertEqual(len(entries), 72)
        self.assertEqual(Counter(entry.source for entry in entries), {
            "dripping_stone": 1, "event_handbook": 2,
            "reproduction_handbook": 10, "seed_catalog": 59,
        })
        self.assertEqual(migration_audit(entries), ())
        self.assertTrue(all(entry.evolvable for entry in entries))

    def test_all_seed_operator_positions_are_in_the_typed_registry(self):
        seeds = [
            entry for entry in build_factor_catalog()
            if entry.source == "seed_catalog"
        ]
        self.assertEqual(len(seeds), len(SEEDS))
        for entry in seeds:
            self.assertTrue(entry.operator_slots)
            self.assertEqual(entry.backend, "typed_tree")
            self.assertTrue(all(name.startswith("seed_") for name in entry.operator_slots))

    def test_progress_is_derived_and_reports_connectivity_and_cost_coverage(self):
        progress = migration_progress()
        self.assertEqual({key: progress[key] for key in (
            "catalog_complete", "typed_dsl_migrated",
            "fine_grained_recombinable", "production_gp_connected",
            "weak_recombination_components", "weak_component_sizes",
            "strong_recombination_components", "strong_component_sizes",
            "directed_source_factor_count", "directed_sink_factor_count",
            "cross_component_directed_edges",
            "total",
        )}, {
            "catalog_complete": 72,
            "typed_dsl_migrated": 72,
            "fine_grained_recombinable": 72,
            "production_gp_connected": 72,
            "weak_recombination_components": 1,
            "weak_component_sizes": (72,),
            "strong_recombination_components": 2,
            "strong_component_sizes": (59, 13),
            "directed_source_factor_count": 13,
            "directed_sink_factor_count": 59,
            "cross_component_directed_edges": 767,
            "total": 72,
        })
        self.assertGreater(progress["cost_calibrated_operators"], 0)
        self.assertLess(
            progress["cost_calibrated_operators"],
            progress["cost_used_operators"],
        )
        self.assertEqual(tuple(map(len, recombination_components())), (72,))
        directed = directed_recombination_audit()
        self.assertEqual(
            tuple(map(len, directed["strong_components"])), (59, 13)
        )
        self.assertEqual(tuple(map(len, directed["source_components"])), (13,))
        self.assertEqual(tuple(map(len, directed["sink_components"])), (59,))
        self.assertTrue(all(
            not hasattr(entry, "fine_grained") for entry in build_factor_catalog()
        ))

    def test_whole_factor_wrapper_cannot_fake_fine_grained_completion(self):
        registry = build_operator_registry()
        root = OperatorNode(
            "handbook_complete_tide",
            (
                LeafNode("close", SemanticType.MINUTE_CLOSE),
                LeafNode("volume", SemanticType.MINUTE_VOLUME),
            ),
            {"neighborhood": 9, "exclude_edges": 30, "smooth_window": 20},
        ).bind(registry)

        class WholeFactorGenome:
            def expression(self, supplied_registry):
                return root, supplied_registry

        entry = FactorCatalogEntry(
            "fake", "test", "typed_tree", "seed_tree",
            WholeFactorGenome(), ("close", "volume"),
            ("handbook_complete_tide",),
        )
        self.assertFalse(is_fine_grained(entry, registry))


class TypedAnchorEquivalenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = _minute()

    def assert_same(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.allclose(
            actual, expected, atol=1e-6, rtol=1e-6, equal_nan=True
        ))

    def test_dripping_anchor_matches_original_template(self):
        volume = _minute(I=4, D=6, M=240)["volume"]
        expected = DrippingStoneTemplate.paper_anchor().evaluate(volume, 64)
        genome = dripping_stone_anchor()
        actual = genome.evaluate({"volume": volume}, 64)
        self.assert_same(actual, expected)
        self.assertEqual(DrippingSkeletonGenome.from_dict(genome.to_dict()), genome)

    def test_all_six_local_handbook_anchors_match_templates(self):
        c = self.context
        expected = {
            "complete_tide": CompleteTideTemplate().evaluate(c["close"], c["volume"]),
            "climb_mountain": ClimbMountainTemplate().evaluate(
                c["open"], c["high"], c["low"], c["close"]
            ),
            "hidden_flower": HiddenFlowerTemplate().evaluate(c["close"], c["volume"]),
            "long_short_battle": LongShortBattleTemplate().evaluate(
                c["high"], c["low"], c["close"], c["volume"]
            ),
            "equal_treatment": EqualTreatmentTemplate().evaluate(
                c["open"], c["close"], c["volume"]
            ),
            "dark_flow": DarkFlowTemplate().evaluate(
                c["open"], c["high"], c["low"], c["volume"]
            ),
        }
        for name, factor in expected.items():
            genome = handbook_anchor(name)
            self.assert_same(genome.evaluate(c), factor)
            self.assert_same(genome.evaluate(c, chunk_rows=24), factor)
            self.assertEqual(
                HandbookSkeletonGenome.from_dict(genome.to_dict()), genome
            )

    def test_external_input_handbook_anchors_match_templates(self):
        I, D, M = 6, 25, 20
        torch.manual_seed(22)
        daily_close = 10 * torch.cumprod(1 + torch.randn(I, D) * .01, 1)
        market_close = daily_close.mean(0)
        amount_share = torch.rand(I, D, M)
        volume_share = torch.rand(I, D, M)
        trend = torch.rand(I, D, M) > .5
        high_amount, low_amount = torch.rand(I, D), torch.rand(I, D)
        cap = torch.rand(I, D) + 1
        state = torch.randint(0, 3, (I, D, M))
        daily_return = torch.randn(I, D) * .01
        similarity = torch.rand(I, I, D)
        contexts = {
            "raw_panic": dict(
                daily_close=daily_close, market_close=market_close,
            ),
            "rushing_forward": dict(
                amount_share=amount_share, volume_share=volume_share,
                up_volume_down_price_mask=trend,
            ),
            "water_boat": dict(
                high_amount=high_amount, low_amount=low_amount,
                float_market_cap=cap,
            ),
            "cooperation_effect": dict(
                volume_share=volume_share, price_state=state,
                daily_return=daily_return, pair_similarity=similarity,
            ),
        }
        expected = {
            "raw_panic": RawPanicTemplate().evaluate(daily_close, market_close),
            "rushing_forward": RushingForwardTemplate().evaluate(
                amount_share, volume_share, trend
            ),
            "water_boat": WaterBoatTemplate().evaluate(
                high_amount, low_amount, cap
            ),
            "cooperation_effect": CooperationEffectTemplate().evaluate(
                volume_share, state, daily_return, similarity
            ),
        }
        for name, factor in expected.items():
            genome = handbook_anchor(name)
            self.assert_same(genome.evaluate(contexts[name]), factor)
            self.assert_same(genome.evaluate(contexts[name], chunk_rows=18), factor)
            self.assertFalse(any(
                operator.startswith("handbook_")
                for operator in genome.operator_slots
            ))

    def test_handbook_mutation_changes_operator_slots_and_stays_registered(self):
        registry = build_operator_registry()
        fields = set(self.context)
        rng = random.Random(7)
        anchor = handbook_anchor("complete_tide")
        variants = {
            mutate_handbook_genome(anchor, rng, fields, registry)
            for _ in range(30)
        }
        self.assertTrue(any(g.operator_slots != anchor.operator_slots for g in variants))
        for genome in variants:
            self.assertLessEqual(set(genome.operator_slots), set(registry.names()))


class HandbookGPSmokeTest(unittest.TestCase):
    def test_small_registered_slot_gp_run(self):
        context = _minute(I=30, D=45, M=48)
        returns = torch.randn(30, 45)
        engine = HandbookFactorGP(
            context, returns,
            gp_config=HandbookGPConfig(
                population_size=5, generations=1, elite=2,
                verbose=False, seed=4,
            ),
            fitness_config=WalkForwardConfig(
            max_turnover=None,
                min_train_days=10, valid_days=8, n_splits=3,
                min_cross_section=15, min_valid_ic_days=3,
                min_fold_consistency=0, min_folds=1,
            ),
        )
        ranked = engine.run()
        self.assertEqual(len(ranked), 5)
        self.assertTrue(all(
            isinstance(
                row[1], (HandbookSkeletonGenome, CompleteTideSkeletonGenome)
                + (ClimbMountainSkeletonGenome, ComposedHandbookGenome)
            ) for row in ranked
        ))


if __name__ == "__main__":
    unittest.main()
