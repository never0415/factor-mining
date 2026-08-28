"""Guards for the shared ranking kernel and its layering.

Step 6a of the legacy retirement: the evaluation layer must not import the old
expression runtime. These tests fail if the dependency creeps back or if a
second copy of the ranking implementation appears.
"""

import ast
import pathlib
import subprocess
import sys
import unittest

import torch

from min_gp.numeric.ranking import cross_section_rank


PACKAGE = pathlib.Path(__file__).resolve().parent.parent


def _imports_from(path, module):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == module:
            return sorted(alias.name for alias in node.names)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == module:
                    return [module]
    return []


class LayeringTest(unittest.TestCase):
    """The numeric layer sits below both DSLs and depends on neither."""

    def test_ranking_module_imports_nothing_from_min_gp(self):
        source = (PACKAGE / "numeric" / "ranking.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertFalse(
                    (node.module or "").startswith("min_gp"),
                    f"numeric layer must not import {node.module}",
                )

    def test_evaluation_layer_no_longer_imports_the_legacy_runtime(self):
        for relative in (
            "fitness.py",
            "evaluation/incremental.py",
            "evaluation/archive.py",
            "operators/cross_section.py",
        ):
            names = _imports_from(PACKAGE / relative, "min_gp.expr")
            self.assertEqual(names, [], f"{relative} still imports min_gp.expr")

    def test_typed_gp_import_does_not_load_the_legacy_runtime(self):
        """Package initialisation must not reintroduce an indirect dependency."""
        probe = (
            "import sys; "
            "import min_gp.gp.event; "
            "raise SystemExit(int('min_gp.expr' in sys.modules))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=PACKAGE.parent,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            completed.stderr or "typed GP import unexpectedly loaded min_gp.expr",
        )

    def test_legacy_package_exports_remain_compatible(self):
        """Old callers may still request the adapter through min_gp.factors."""
        from min_gp.factors import LegacyExpressionGenome
        from min_gp.factors.legacy import LegacyExpressionGenome as direct
        self.assertIs(LegacyExpressionGenome, direct)

    def test_legacy_alias_is_the_same_object_not_a_copy(self):
        """A second implementation would drift; the alias must stay an alias."""
        from min_gp.expr import _cs_rank
        self.assertIs(_cs_rank, cross_section_rank)

    def test_typed_operator_delegates_to_the_shared_kernel(self):
        from min_gp.operators.cross_section import cross_section_rank as op_rank
        x = torch.randn(40, 12)
        self.assertTrue(
            torch.allclose(
                op_rank(x), cross_section_rank(x.float()), equal_nan=True
            )
        )


class RankingBehaviourTest(unittest.TestCase):
    def test_ties_receive_the_average_rank(self):
        x = torch.tensor([[1.0], [1.0], [1.0], [2.0]])
        ranks = cross_section_rank(x)[:, 0]
        # Three-way tie occupies positions 0,1,2 -> mid-rank 1 -> 1/3 after
        # dividing by (count - 1) = 3.
        self.assertTrue(torch.allclose(ranks[:3], torch.full((3,), 1 / 3)))
        self.assertAlmostEqual(float(ranks[3]), 1.0, places=6)

    def test_row_order_alone_cannot_create_a_rank_spread(self):
        constant = torch.ones(50, 8)
        ranks = cross_section_rank(constant)
        self.assertTrue(torch.allclose(ranks, torch.full_like(ranks, 0.5)))

    def test_nan_positions_are_preserved_and_excluded(self):
        x = torch.tensor([[0.0], [float("nan")], [1.0]])
        ranks = cross_section_rank(x)[:, 0]
        self.assertTrue(torch.isnan(ranks[1]))
        self.assertAlmostEqual(float(ranks[0]), 0.0, places=6)
        self.assertAlmostEqual(float(ranks[2]), 1.0, places=6)

    def test_dim_argument_selects_the_ranked_axis(self):
        x = torch.randn(6, 5)
        along_rows = cross_section_rank(x, dim=1)
        self.assertTrue(
            torch.allclose(
                along_rows, cross_section_rank(x.T, dim=0).T, equal_nan=True
            )
        )

    def test_boolean_input_is_promoted_rather_than_rejected(self):
        mask = torch.tensor([[True], [False], [True], [False]])
        ranks = cross_section_rank(mask)[:, 0]
        self.assertTrue(torch.isfinite(ranks).all())
        self.assertGreater(float(ranks[0]), float(ranks[1]))


if __name__ == "__main__":
    unittest.main()
