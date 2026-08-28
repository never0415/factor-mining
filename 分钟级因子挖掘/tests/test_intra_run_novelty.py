"""Elite selection must drop near-duplicates in factor space.

NSGA-II measures crowding in objective space and rejects duplicates by genome
structure, so two expressions differing by one constant both survive even when
their factors correlate 0.99. These tests pin the factor-space cut that closes
that gap.
"""

import unittest

import torch

from min_gp.gp.seed_tree import SeedTreeFactorGP, SeedTreeGPConfig


class _Genome:
    """Minimal stand-in carrying a fixed factor panel."""

    def __init__(self, name, factor):
        self.name = name
        self.factor = factor

    def __repr__(self):
        return f"_Genome({self.name})"


class _Harness(SeedTreeFactorGP):
    """Exercises ``_distinct`` alone, without building a real population."""

    def __init__(self, threshold):
        self.config = SeedTreeGPConfig(intra_run_max_correlation=threshold)

    def _factor(self, genome):
        return genome.factor


def _panel(seed, instruments=60, days=12):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(instruments, days, generator=generator)


class IntraRunNoveltyTest(unittest.TestCase):
    def setUp(self):
        base = _panel(1)
        # A monotone rescale leaves every daily ranking identical, so this is a
        # correlation-1.0 duplicate however the levels are compared.
        self.a = _Genome("a", base)
        self.a_twin = _Genome("a_twin", base * 3.0 + 7.0)
        self.b = _Genome("b", _panel(2))
        self.scored = [(None, g) for g in (self.a, self.a_twin, self.b)]
        self.order = [0, 1, 2]

    def test_threshold_none_preserves_historical_behaviour(self):
        kept = _Harness(None)._distinct(self.scored, self.order, 3)
        self.assertEqual(kept, [self.a, self.a_twin, self.b])

    def test_duplicate_is_dropped_and_best_ranked_member_survives(self):
        kept = _Harness(0.9)._distinct(self.scored, self.order, 2)
        self.assertIn(self.a, kept)
        self.assertIn(self.b, kept)
        self.assertNotIn(self.a_twin, kept)

    def test_shortfall_is_topped_up_to_the_requested_size(self):
        # Only two distinct signals exist, but three elites are requested; the
        # population must not shrink.
        kept = _Harness(0.9)._distinct(self.scored, self.order, 3)
        self.assertEqual(len(kept), 3)
        self.assertEqual(kept[:2], [self.a, self.b])
        self.assertIn(self.a_twin, kept)

    def test_uncorrelated_factors_all_survive(self):
        scored = [(None, _Genome(str(i), _panel(10 + i))) for i in range(4)]
        kept = _Harness(0.9)._distinct(scored, list(range(4)), 4)
        self.assertEqual(len(kept), 4)

    def test_threshold_is_on_absolute_correlation(self):
        # An inverted copy carries the same information; direction is settled
        # separately and would erase the sign difference anyway.
        base = _panel(3)
        scored = [(None, _Genome("x", base)), (None, _Genome("neg", -base))]
        kept = _Harness(0.9)._distinct(scored, [0, 1], 1)
        self.assertEqual(len(kept), 1)


class ConfigTest(unittest.TestCase):
    def test_default_is_disabled(self):
        self.assertIsNone(SeedTreeGPConfig().intra_run_max_correlation)


if __name__ == "__main__":
    unittest.main()
