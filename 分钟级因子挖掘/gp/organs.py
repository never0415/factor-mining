"""Reusable typed subtrees extracted from externally supplied factor trees.

An organ is deliberately *not* a leaf.  Its complete expression remains in
the genome, so crossover and mutation can enter the organ and change any of
its internal nodes.  The library only provides structural reuse and
provenance; numerical materialisation is an execution optimisation and is not
part of the genetic representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from min_gp.dsl import LeafNode, OperatorNode, SemanticType
from min_gp.gp.typed_tree import node_key


# These are the factor definitions in factor_reproduction_handbook.pdf.  The
# explicit allow-list is important: run logs, all76 and prior Pareto candidates
# must never silently become organ donors.
EXTERNAL_FACTOR_NAMES = (
    "dripping_stone",
    "moderate_risk",
    "wait_rescue",
    "complete_tide",
    "climb_mountain",
    "hidden_flower",
    "long_short_battle",
    "equal_treatment",
    "dark_flow",
    "raw_panic",
    "rushing_forward",
    "water_boat",
    "cooperation_effect",
)


def walk_nodes(node):
    yield node
    if isinstance(node, OperatorNode):
        for child in node.children:
            yield from walk_nodes(child)


def tree_levels(node) -> int:
    """Tree levels with terminals at level 1 (sqrt(volume) has 2 levels)."""
    if not isinstance(node, OperatorNode):
        return 1
    return 1 + max(tree_levels(child) for child in node.children)


def tree_node_count(node) -> int:
    return sum(1 for _ in walk_nodes(node))


def required_fields(node) -> tuple[str, ...]:
    return tuple(sorted({
        item.name for item in walk_nodes(node) if isinstance(item, LeafNode)
    }))


@dataclass(frozen=True)
class OrganBlock:
    """One deduplicated, provenance-carrying reusable expression subtree."""

    root: OperatorNode
    sources: tuple[str, ...]
    levels: int
    node_count: int
    required_fields: tuple[str, ...]
    structural_key: tuple

    @property
    def output_type(self) -> SemanticType:
        return self.root.output_type

    def to_dict(self) -> dict:
        return {
            "sources": list(self.sources),
            "levels": self.levels,
            "node_count": self.node_count,
            "required_fields": list(self.required_fields),
            "output_type": self.output_type.value,
            "expression": str(self.root),
        }


@dataclass(frozen=True)
class OrganLibrary:
    blocks: tuple[OrganBlock, ...]
    source_document: str = "factor_reproduction_handbook.pdf"

    def compatible(
        self, expected_type, registry, *, max_levels=None,
        available_fields=None,
    ) -> tuple[OrganBlock, ...]:
        available = None if available_fields is None else set(available_fields)
        return tuple(
            block for block in self.blocks
            if registry.accepts(block.output_type, expected_type)
            and (max_levels is None or block.levels <= max_levels)
            and (
                available is None
                or set(block.required_fields) <= available
            )
        )

    @property
    def source_names(self) -> tuple[str, ...]:
        return tuple(sorted({name for block in self.blocks for name in block.sources}))

    def manifest(self) -> dict:
        return {
            "source_document": self.source_document,
            "source_policy": "external_factor_definitions_only",
            "genetic_representation": "transparent_mutable_subtree",
            "source_names": list(self.source_names),
            "organ_count": len(self.blocks),
            "organs": [block.to_dict() for block in self.blocks],
        }


def _contains_factor_wrapper(node, registry) -> bool:
    return any(
        registry.get(item.name).factor_wrapper
        for item in walk_nodes(node) if isinstance(item, OperatorNode)
    )


def _external_roots(registry, source_names=None):
    """Load only the typed trees corresponding to the external PDF factors."""
    from min_gp.factors.catalog import build_factor_catalog

    requested = set(EXTERNAL_FACTOR_NAMES if source_names is None else source_names)
    unknown = requested - set(EXTERNAL_FACTOR_NAMES)
    if unknown:
        raise ValueError(f"unknown external organ sources: {sorted(unknown)}")
    roots = []
    for entry in build_factor_catalog():
        if entry.name not in requested or entry.source == "seed_catalog":
            continue
        expression = getattr(entry.genome, "expression", None)
        if expression is None:
            continue
        root, _ = expression(registry)
        roots.append((entry.name, root))
    missing = requested - {name for name, _ in roots}
    if missing:
        raise ValueError(f"external factors have no typed source tree: {sorted(missing)}")
    return tuple(roots)


def build_organ_library(
    registry, donors: Iterable[tuple[str, object]], *, min_levels=2,
    max_levels=3, available_fields=None, source_document="external_input",
) -> OrganLibrary:
    """Generic extractor for any explicitly supplied collection of typed trees."""
    if min_levels < 2 or max_levels < min_levels:
        raise ValueError("organ levels must satisfy 2 <= min_levels <= max_levels")
    available = None if available_fields is None else set(available_fields)
    grouped = {}
    for source, root in donors:
        for node in walk_nodes(root):
            if node is root or not isinstance(node, OperatorNode):
                continue
            levels = tree_levels(node)
            fields = required_fields(node)
            if not min_levels <= levels <= max_levels:
                continue
            if available is not None and not set(fields) <= available:
                continue
            if _contains_factor_wrapper(node, registry):
                continue
            key = node_key(node)
            if key not in grouped:
                grouped[key] = {"root": node, "sources": set()}
            grouped[key]["sources"].add(source)
    blocks = tuple(sorted((
        OrganBlock(
            root=value["root"],
            sources=tuple(sorted(value["sources"])),
            levels=tree_levels(value["root"]),
            node_count=tree_node_count(value["root"]),
            required_fields=required_fields(value["root"]),
            structural_key=key,
        )
        for key, value in grouped.items()
    ), key=lambda block: (block.output_type.value, block.levels, str(block.root))))
    return OrganLibrary(blocks, source_document=source_document)


def build_external_organ_library(
    registry, *, min_levels=2, max_levels=3, source_names=None,
    available_fields=None,
) -> OrganLibrary:
    """Extract and exactly deduplicate depth-2/3 organs from PDF factor trees.

    The donor root itself is excluded even when it is shallow: the objective
    is to reuse an internal mechanism, not to reintroduce a failed whole factor
    under another name. Compatibility wrappers are excluded recursively.
    """
    return build_organ_library(
        registry, _external_roots(registry, source_names),
        min_levels=min_levels, max_levels=max_levels,
        available_fields=available_fields,
        source_document="factor_reproduction_handbook.pdf",
    )


def graft_organ(tree, registry, library, rng, *, max_levels=None):
    """Replace one compatible node with one transparent organ subtree."""
    from min_gp.gp.typed_tree import TypedTreeGenome, _replace, _slots

    choices = []
    for path, node, expected_type in _slots(tree.root, registry):
        blocks = library.compatible(
            expected_type, registry, max_levels=max_levels,
        )
        choices.extend((path, block) for block in blocks)
    if not choices:
        return tree
    path, block = rng.choice(choices)
    return TypedTreeGenome(_replace(tree.root, path, block.root, registry))
