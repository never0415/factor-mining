import ast
from dataclasses import replace
from pathlib import Path
import unittest

from min_gp.gp.seed_tree import SeedTreeGPConfig


PACKAGE = Path(__file__).resolve().parents[1]


class TypedGPCLITest(unittest.TestCase):
    def test_seed_config_can_select_a_small_anchor_island(self):
        config = replace(SeedTreeGPConfig(), anchor_names=("s27_peak_count",))
        self.assertEqual(config.anchor_names, ("s27_peak_count",))

    def test_preflight_precedes_expensive_loads(self):
        checks = (
            ("seed_tree_gp.py", "check_or_exit", "build_slice"),
            ("handbook_gp.py", "check_or_exit", "build_minute_slice"),
        )
        for filename, check_name, load_name in checks:
            source = (PACKAGE / filename).read_text(encoding="utf-8")
            tree = ast.parse(source)
            calls = {
                node.func.id: node.lineno
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in {check_name, load_name}
            }
            self.assertLess(calls[check_name], calls[load_name], filename)

    def test_console_scripts_are_declared(self):
        project = (PACKAGE.parent / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('min-gp-handbook = "min_gp.handbook_gp:main"', project)
        self.assertIn('min-gp-seed-tree = "min_gp.seed_tree_gp:main"', project)


if __name__ == "__main__":
    unittest.main()
