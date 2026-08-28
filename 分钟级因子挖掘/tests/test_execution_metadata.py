"""Execution metadata and semantic chunking invariants."""

import unittest
from dataclasses import replace

import torch

from min_gp.dsl import (
    ExecutionScope,
    LeafNode,
    OperatorNode,
    SemanticType,
    evaluate_daily_expression,
)
from min_gp.factors.event_skeleton import Slot
from min_gp.factors.dripping_skeleton import dripping_stone_anchor
from min_gp.factors.handbook_skeleton import HandbookSkeletonGenome, handbook_anchor
from min_gp.operators import build_operator_registry


class MetadataTest(unittest.TestCase):
    def test_minute_index_is_distinct_from_a_mask(self):
        self.assertNotEqual(SemanticType.MINUTE_INDEX, SemanticType.MINUTE_MASK)

    def test_tree_scope_is_the_bitwise_union_of_its_operators(self):
        registry = build_operator_registry()
        leaf = LeafNode("raw", SemanticType.DAILY_RAW_FACTOR)
        ranked = OperatorNode("cross_section_rank", (leaf,)).bind(registry)
        root = OperatorNode(
            "mean_std_blend", (ranked,), {"window": 5}
        ).bind(registry)
        self.assertEqual(
            root.execution_scope_with(registry),
            ExecutionScope.FULL_CROSS_SECTION | ExecutionScope.HISTORY,
        )
        self.assertEqual(root.history_days_with(registry), 4)

    def test_history_and_lookahead_compose_along_a_path(self):
        registry = build_operator_registry()
        genome = HandbookSkeletonGenome(
            core=Slot.of(
                "handbook_equal_treatment", response_window=5,
                exclude_edges=15, smooth_window=5,
            ),
            cross_section=Slot.of("cross_section_identity"),
            low_frequency=Slot.of("mean_std_blend", window=5),
        )
        self.assertEqual(genome.history_days, 8)
        event = OperatorNode(
            "forward_window_std",
            (LeafNode("ret", SemanticType.MINUTE_RETURN),),
            {"window": 5, "ddof": 0},
        ).bind(registry)
        self.assertEqual(event.intraday_lookahead_with(registry), 4)

    def test_complexity_is_structured_by_axis(self):
        profile = handbook_anchor("hidden_flower").complexity_profile
        # The decomposed peer-correlation branch exposes its quadratic
        # cross-stock cost instead of hiding it inside one whole-factor core.
        self.assertEqual(profile["I"], 2)
        self.assertEqual(profile["P"], 2)


class ChunkInvariantTest(unittest.TestCase):
    def test_cross_section_tree_is_chunked_by_date(self):
        torch.manual_seed(31)
        raw = torch.randn(9, 11)
        raw[0, 3] = float("nan")
        registry = build_operator_registry()
        root = OperatorNode(
            "cross_section_rank",
            (LeafNode("raw", SemanticType.DAILY_RAW_FACTOR),),
        ).bind(registry)
        whole = root.evaluate({"raw": raw}, registry)
        chunked = evaluate_daily_expression(
            root, {"raw": raw}, registry, chunk_rows=18
        )
        self.assertTrue(torch.allclose(whole, chunked, equal_nan=True))

    def test_cross_section_plus_history_matches_whole_grid(self):
        torch.manual_seed(32)
        I, D, M = 6, 14, 48
        volume = torch.rand(I, D, M) * 1000 + 1
        close = 10 * torch.cumprod(
            1 + torch.randn(I, D, M) * 0.001, dim=2
        )
        open_ = close * (1 + torch.randn_like(close) * 0.0002)
        high = torch.maximum(open_, close) * 1.001
        low = torch.minimum(open_, close) * 0.999
        context = {"open": open_, "high": high, "low": low, "volume": volume}
        genome = handbook_anchor(
            "dark_flow", bins=24, lookback=3,
            multiple=1.0, smooth_window=5,
        )
        root, registry = genome.expression()
        whole = root.evaluate(context, registry)
        chunked = genome.evaluate(context, registry, chunk_rows=I * 3)
        self.assertTrue(torch.allclose(
            whole, chunked, atol=1e-6, rtol=1e-6, equal_nan=True
        ))

    def test_independent_tree_may_use_row_chunks(self):
        torch.manual_seed(33)
        I, D = 7, 9
        context = {
            "high_amount": torch.rand(I, D),
            "low_amount": torch.rand(I, D),
            "float_market_cap": torch.rand(I, D) + 1,
        }
        genome = handbook_anchor("water_boat")
        root, registry = genome.expression()
        whole = root.evaluate(context, registry)
        chunked = genome.evaluate(context, registry, chunk_rows=10)
        self.assertTrue(torch.allclose(whole, chunked, equal_nan=True))

    def test_dripping_temporal_tail_receives_history_halo(self):
        torch.manual_seed(34)
        volume = torch.rand(3, 12, 240) * 1000 + 1
        genome = replace(
            dripping_stone_anchor(),
            low_frequency=Slot.of("mean_std_blend", window=5),
        )
        root, registry = genome.expression()
        whole = root.evaluate({"volume": volume}, registry)
        chunked = genome.evaluate(
            {"volume": volume}, chunk_rows=6, registry=registry
        )
        self.assertTrue(torch.allclose(
            whole, chunked, atol=1e-6, rtol=1e-6, equal_nan=True
        ))


if __name__ == "__main__":
    unittest.main()
