import random
import unittest

from min_gp.dsl import LeafNode, OperatorNode, SemanticType
from min_gp.gp.organs import (
    EXTERNAL_FACTOR_NAMES, OrganBlock, OrganLibrary,
    build_external_organ_library, graft_organ, required_fields,
    tree_levels, tree_node_count, walk_nodes,
)
from min_gp.gp.typed_tree import TypedTreeGenome, crossover_trees, node_key
from min_gp.operators import build_operator_registry


class OrganReuseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = build_operator_registry()
        cls.library = build_external_organ_library(cls.registry)

    def test_only_external_pdf_factors_are_donors(self):
        self.assertEqual(
            set(self.library.source_names), set(EXTERNAL_FACTOR_NAMES)
        )
        self.assertNotIn("all76", self.library.source_names)
        self.assertFalse(any(
            source.startswith("s") and "_" in source
            for source in self.library.source_names
        ))

    def test_organs_are_internal_two_or_three_level_trees(self):
        self.assertTrue(self.library.blocks)
        keys = [block.structural_key for block in self.library.blocks]
        self.assertEqual(len(keys), len(set(keys)))
        for block in self.library.blocks:
            self.assertIsInstance(block.root, OperatorNode)
            self.assertIn(block.levels, (2, 3))
            self.assertEqual(block.levels, tree_levels(block.root))
            self.assertEqual(block.node_count, tree_node_count(block.root))
            self.assertEqual(block.required_fields, required_fields(block.root))
            self.assertFalse(any(
                self.registry.get(node.name).factor_wrapper
                for node in walk_nodes(block.root)
                if isinstance(node, OperatorNode)
            ))

    def test_available_fields_filter_is_applied_before_gp(self):
        library = build_external_organ_library(
            self.registry, available_fields={"open", "high", "low", "close", "volume"}
        )
        for block in library.blocks:
            self.assertLessEqual(
                set(block.required_fields), {"open", "high", "low", "close", "volume"}
            )

    def test_graft_keeps_a_transparent_mutable_subtree(self):
        leaf = LeafNode("close", SemanticType.MINUTE_CLOSE)
        host_root = OperatorNode(
            "seed_day_mean_minute", (leaf,), {}
        ).bind(self.registry)
        organ_root = OperatorNode(
            "seed_abs_minute", (leaf,), {}
        ).bind(self.registry)
        block = OrganBlock(
            root=organ_root, sources=("climb_mountain",),
            levels=tree_levels(organ_root), node_count=tree_node_count(organ_root),
            required_fields=("close",), structural_key=node_key(organ_root),
        )
        grafted = graft_organ(
            TypedTreeGenome(host_root), self.registry, OrganLibrary((block,)),
            random.Random(2), max_levels=3,
        )
        nodes = list(walk_nodes(grafted.root))
        self.assertTrue(any(node_key(node) == node_key(organ_root) for node in nodes))
        # Its child is still a real genome node, not an opaque daily factor leaf.
        self.assertTrue(any(
            isinstance(node, LeafNode) and node.name == "close" for node in nodes
        ))
        self.assertFalse(any(
            isinstance(node, LeafNode) and node.name == "climb_mountain"
            for node in nodes
        ))

    def test_manifest_states_genetic_semantics(self):
        manifest = self.library.manifest()
        self.assertEqual(
            manifest["genetic_representation"], "transparent_mutable_subtree"
        )
        self.assertEqual(
            manifest["source_policy"], "external_factor_definitions_only"
        )

    def test_crossover_matches_parent_slot_not_replaced_leaf_precision(self):
        close = LeafNode("close", SemanticType.MINUTE_CLOSE)
        volume = LeafNode("volume", SemanticType.MINUTE_VOLUME)
        left = TypedTreeGenome(OperatorNode(
            "seed_day_mean_minute", (close,), {}
        ).bind(self.registry))
        donor_subtree = OperatorNode(
            "volume_transform", (volume,), {"mode": "sqrt"}
        ).bind(self.registry)
        right = TypedTreeGenome(OperatorNode(
            "seed_day_mean_minute", (donor_subtree,), {}
        ).bind(self.registry))
        # MINUTE_VOLUME cannot replace a MINUTE_CLOSE by exact semantic type,
        # but both legally fill the parent's generic LEGACY_MINUTE slot.
        found = False
        rng = random.Random(41)
        for _ in range(100):
            child = crossover_trees(left, right, self.registry, rng)
            if "volume_transform" in str(child.root):
                found = True
                break
        self.assertTrue(found)


if __name__ == "__main__":
    unittest.main()
