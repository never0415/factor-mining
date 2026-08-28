"""The leaf pool must be able to come from the data, not from the seed library.

Harvesting leaves by walking seed expressions makes the reachable leaf set a
property of which expressions happen to exist. A measured run held 37 tensors
in context and let random trees select 15 of them.
"""

import unittest

import torch

from min_gp.dsl import SemanticType
from min_gp.gp.seed_tree import SeedTreeGPConfig, _context_leaf_type


class ContextLeafTypeTest(unittest.TestCase):
    def test_minute_panel_is_legacy_minute(self):
        self.assertIs(
            _context_leaf_type(torch.zeros(4, 3, 241)),
            SemanticType.LEGACY_MINUTE,
        )

    def test_session_mask_is_session_mask(self):
        self.assertIs(
            _context_leaf_type(torch.zeros(241)), SemanticType.SESSION_MASK
        )

    def test_daily_grid_is_skipped(self):
        # (I,D) covers several distinct semantic types, so shape cannot decide
        # between them; those leaves arrive as anchors with declared types.
        self.assertIsNone(_context_leaf_type(torch.zeros(4, 3)))

    def test_non_tensor_is_skipped(self):
        self.assertIsNone(_context_leaf_type("not a tensor"))
        self.assertIsNone(_context_leaf_type(None))

    def test_dtype_does_not_matter(self):
        # Session masks are stored as bfloat16 rather than bool, and w_time is
        # a continuous decay weight sharing that shape.
        for dtype in (torch.bfloat16, torch.float32, torch.bool):
            self.assertIs(
                _context_leaf_type(torch.zeros(241, dtype=dtype)),
                SemanticType.SESSION_MASK,
            )


class ConfigTest(unittest.TestCase):
    def test_default_is_disabled(self):
        self.assertFalse(SeedTreeGPConfig().leaves_from_context)

    def test_flag_is_settable(self):
        self.assertTrue(
            SeedTreeGPConfig(leaves_from_context=True).leaves_from_context
        )


class PoolWideningTest(unittest.TestCase):
    """The widening step itself, exercised without building a real GP."""

    def _widen(self, seed_leaf_names, context):
        from min_gp.dsl import LeafNode
        leaves = {
            LeafNode(name, SemanticType.LEGACY_MINUTE)
            for name in seed_leaf_names
        }
        known = {leaf.name for leaf in leaves}
        for name in sorted(context):
            if name in known:
                continue
            semantic = _context_leaf_type(context[name])
            if semantic is not None:
                leaves.add(LeafNode(name, semantic))
        return {leaf.name for leaf in leaves}

    def test_stranded_tensors_become_reachable(self):
        context = {
            "close": torch.zeros(4, 3, 241),
            "open": torch.zeros(4, 3, 241),      # no seed mentions this one
            "is_jump": torch.zeros(4, 3, 241),   # nor this
            "mask_am": torch.zeros(241),
            "mask_open_5m": torch.zeros(241),    # nor these eighteen
            "w_time": torch.zeros(241),
            "some_daily_factor": torch.zeros(4, 3),  # skipped by design
        }
        widened = self._widen({"close", "mask_am"}, context)
        self.assertEqual(
            widened,
            {"close", "open", "is_jump", "mask_am", "mask_open_5m", "w_time"},
        )
        self.assertNotIn("some_daily_factor", widened)

    def test_seed_leaves_are_never_dropped(self):
        context = {"close": torch.zeros(4, 3, 241)}
        widened = self._widen({"close", "volume", "tp"}, context)
        self.assertTrue({"close", "volume", "tp"} <= widened)


if __name__ == "__main__":
    unittest.main()
