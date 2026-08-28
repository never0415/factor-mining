"""Generic strongly typed tree generation, mutation and crossover."""

from dataclasses import dataclass
import random

from min_gp.dsl import (
    ConstantNode, ExecutionScope, LeafNode, OperatorNode, SemanticType,
)


def node_key(node):
    if isinstance(node, ConstantNode):
        return ("constant", node.value)
    if isinstance(node, LeafNode):
        return ("leaf", node.name, node.output_type.value)
    return (
        "op", node.name, tuple(sorted(node.params.items())),
        tuple(node_key(child) for child in node.children),
    )


def node_to_dict(node):
    if isinstance(node, ConstantNode):
        return {"constant": node.value}
    if isinstance(node, LeafNode):
        return {"leaf": node.name, "type": node.output_type.value}
    return {
        "operator": node.name,
        "params": dict(node.params),
        "children": [node_to_dict(child) for child in node.children],
    }


def node_from_dict(payload, registry):
    if "constant" in payload:
        return ConstantNode(payload["constant"])
    if "leaf" in payload:
        return LeafNode(payload["leaf"], SemanticType(payload["type"]))
    node = OperatorNode(
        payload["operator"],
        tuple(node_from_dict(child, registry) for child in payload["children"]),
        payload.get("params", {}),
    )
    return node.bind(registry)


@dataclass(frozen=True)
class TypedTreeGenome:
    root: LeafNode | ConstantNode | OperatorNode

    def __hash__(self):
        return hash(node_key(self.root))

    def __eq__(self, other):
        return isinstance(other, TypedTreeGenome) and node_key(self.root) == node_key(other.root)

    def evaluate(self, context, registry):
        return self.root.evaluate(context, registry)

    def expression(self, registry):
        return self.root, registry

    def complexity(self, registry):
        return (
            self.root.complexity_with(registry)
            if isinstance(self.root, OperatorNode) else 0
        )

    def execution_scope(self, registry):
        return (
            self.root.execution_scope_with(registry)
            if isinstance(self.root, OperatorNode) else ExecutionScope.LOCAL
        )

    def history_days(self, registry):
        return (
            self.root.history_days_with(registry)
            if isinstance(self.root, OperatorNode) else 0
        )

    def intraday_lookahead_minutes(self, registry):
        return (
            self.root.intraday_lookahead_with(registry)
            if isinstance(self.root, OperatorNode) else 0
        )

    def complexity_profile(self, registry):
        return (
            self.root.complexity_profile_with(registry)
            if isinstance(self.root, OperatorNode) else {}
        )

    def to_dict(self):
        return node_to_dict(self.root)

    @classmethod
    def from_dict(cls, payload, registry):
        return cls(node_from_dict(payload, registry))

    def __str__(self):
        return str(self.root)


def _params(spec, rng):
    return {
        name: rng.choice(tuple(domain))
        for name, domain in spec.parameter_domains.items()
    }


def random_tree(
    registry, output_type, leaves, max_depth, rng: random.Random,
    allowed_operators=None,
):
    if output_type == SemanticType.SCALAR:
        return ConstantNode(rng.choice((0, 1, 2)))
    # Leaves fill a slot on the same directional rule the registry applies to
    # operators. An exact-equality test kept every economically precise leaf --
    # amount_share, volume_share, a price-state mask -- out of the generic
    # LEGACY_MINUTE slots the migrated seed arithmetic is built from, so those
    # leaves could only ever enter a tree by crossover from a seed that already
    # contained them, and never by generation.
    leaf_options = [
        leaf for leaf in leaves
        if registry.accepts(leaf.output_type, output_type)
    ]
    operators = [
        spec for spec in registry.matching(output_type, compatible=True)
        if allowed_operators is None or spec.name in allowed_operators
    ]
    if leaf_options and (max_depth <= 0 or not operators or rng.random() < 0.35):
        return rng.choice(leaf_options)
    if max_depth <= 0 or not operators:
        if leaf_options:
            return rng.choice(leaf_options)
        raise ValueError(f"cannot produce terminal of type {output_type.value}")
    rng.shuffle(operators)
    for spec in operators:
        try:
            children = tuple(
                random_tree(
                    registry, kind, leaves, max_depth - 1, rng,
                    allowed_operators,
                )
                for kind in spec.input_types
            )
            return OperatorNode(spec.name, children, _params(spec, rng)).bind(registry)
        except ValueError:
            continue
    if leaf_options:
        return rng.choice(leaf_options)
    raise ValueError(f"no feasible tree for {output_type.value} at depth {max_depth}")


def _nodes(node, path=()):
    found = [(path, node)]
    if isinstance(node, OperatorNode):
        for index, child in enumerate(node.children):
            found.extend(_nodes(child, path + (index,)))
    return found


def _slots(node, registry, path=(), expected_type=None):
    """Nodes paired with the type required by their parent input slot."""
    expected_type = node.output_type if expected_type is None else expected_type
    found = [(path, node, expected_type)]
    if isinstance(node, OperatorNode):
        spec = registry.get(node.name)
        for index, (child, child_expected) in enumerate(
            zip(node.children, spec.input_types)
        ):
            found.extend(_slots(
                child, registry, path + (index,), child_expected
            ))
    return found


def _replace(node, path, replacement, registry):
    if not path:
        return replacement
    children = list(node.children)
    children[path[0]] = _replace(children[path[0]], path[1:], replacement, registry)
    return OperatorNode(node.name, tuple(children), dict(node.params)).bind(registry)


def mutate_tree(
    genome, registry, leaves, max_depth, rng, allowed_operators=None,
    organ_library=None, organ_probability=0.0,
):
    path, old, expected_type = rng.choice(_slots(genome.root, registry))
    replacement = None
    if organ_library and rng.random() < organ_probability:
        organs = organ_library.compatible(
            expected_type, registry, max_levels=max_depth + 1,
        )
        if organs:
            replacement = rng.choice(organs).root
    if replacement is None:
        replacement = random_tree(
            registry, expected_type, leaves, max_depth, rng,
            allowed_operators,
        )
    return TypedTreeGenome(_replace(genome.root, path, replacement, registry))


def crossover_trees(a, b, registry, rng):
    left = _slots(a.root, registry)
    right = _nodes(b.root)
    compatible = [
        (path, node, [
            donor for _, donor in right
            if registry.accepts(donor.output_type, expected_type)
        ])
        for path, node, expected_type in left
    ]
    compatible = [item for item in compatible if item[2]]
    if not compatible:
        return a
    path, _, donors = rng.choice(compatible)
    donor = rng.choice(donors)
    return TypedTreeGenome(_replace(a.root, path, donor, registry))
