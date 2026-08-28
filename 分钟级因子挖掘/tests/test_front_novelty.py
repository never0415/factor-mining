"""Near-duplicates must not occupy slots on the reported Pareto front.

Deduplicating elites alone left the front untouched: a measured run with the
elite cut active still returned a top fifteen that collapsed into three
clusters, with one pair correlated 0.998. These tests pin the front-level cut.
"""

import unittest

import torch

from min_gp.gp.seed_tree import SeedTreeFactorGP, SeedTreeGPConfig


class _Genome:
    def __init__(self, name, factor):
        self.name = name
        self.factor = factor

    def __repr__(self):
        return f"_Genome({self.name})"


class _Harness(SeedTreeFactorGP):
    def __init__(self, threshold, verbose=False):
        self.config = SeedTreeGPConfig(
            intra_run_max_correlation=threshold, verbose=verbose
        )

    def _factor(self, genome):
        return genome.factor


def _panel(seed, instruments=60, days=12):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(instruments, days, generator=generator)


class FrontNoveltyTest(unittest.TestCase):
    def setUp(self):
        base = _panel(1)
        self.a = _Genome("a", base)
        self.a_twin = _Genome("a_twin", base * 2.5 + 4.0)   # rank-identical
        self.b = _Genome("b", _panel(2))
        self.scored = [(None, g) for g in (self.a, self.a_twin, self.b)]
        self.order = [0, 1, 2]

    def test_duplicate_is_demoted_off_the_front(self):
        ranks = [0, 0, 0]
        out = _Harness(0.9)._demote_duplicate_front(self.scored, ranks, self.order)
        self.assertEqual(out[0], 0, "best-ranked member keeps its slot")
        self.assertEqual(out[1], 1, "its duplicate is pushed to the next rank")
        self.assertEqual(out[2], 0, "an independent signal keeps its slot")

    def test_threshold_none_changes_nothing(self):
        ranks = [0, 0, 0]
        out = _Harness(None)._demote_duplicate_front(self.scored, ranks, self.order)
        self.assertEqual(out, ranks)

    def test_non_front_genomes_are_untouched(self):
        ranks = [0, 3, 0]
        out = _Harness(0.9)._demote_duplicate_front(self.scored, ranks, self.order)
        self.assertEqual(out[1], 3, "a genome already off the front is not re-ranked")

    def test_input_ranks_are_not_mutated(self):
        ranks = [0, 0, 0]
        _Harness(0.9)._demote_duplicate_front(self.scored, ranks, self.order)
        self.assertEqual(ranks, [0, 0, 0])

    def test_order_decides_which_member_survives(self):
        # Walking best-first means the survivor is the one the caller ranked
        # highest, not whichever happens to come first in the list.
        ranks = [0, 0, 0]
        out = _Harness(0.9)._demote_duplicate_front(
            self.scored, ranks, [1, 0, 2]
        )
        self.assertEqual(out[1], 0, "the first one visited survives")
        self.assertEqual(out[0], 1)

    def test_independent_signals_all_survive(self):
        scored = [(None, _Genome(str(i), _panel(30 + i))) for i in range(5)]
        ranks = [0] * 5
        out = _Harness(0.9)._demote_duplicate_front(scored, ranks, list(range(5)))
        self.assertEqual(out, [0] * 5)

    def test_a_broken_factor_is_skipped_not_promoted(self):
        class _Broken(_Genome):
            def __init__(self):
                super().__init__("broken", None)

        broken = _Broken()
        harness = _Harness(0.9)

        def factor(genome):
            if genome is broken:
                raise RuntimeError("cannot evaluate")
            return genome.factor

        harness._factor = factor
        scored = [(None, self.a), (None, broken), (None, self.a_twin)]
        out = harness._demote_duplicate_front(scored, [0, 0, 0], [0, 1, 2])
        self.assertEqual(out[0], 0)
        self.assertEqual(out[1], 0, "an unevaluable genome is left alone")
        self.assertEqual(out[2], 1, "the duplicate is still demoted")


if __name__ == "__main__":
    unittest.main()
