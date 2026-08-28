"""Serializable genome adapter for the original 59 expression-tree seeds.

The original expression runtime already has a registered operator table and
type-aware subtree mutation/crossover.  This adapter gives those trees the
same genome metadata contract as the newer slot skeletons without changing
their numerical implementation.
"""

from dataclasses import dataclass
import random as random_module

from min_gp.expr import Leaf, Op, OP, parse


def _all_nodes(node):
    yield node
    if isinstance(node, Op):
        for child in node.args:
            yield from _all_nodes(child)


@dataclass(frozen=True)
class LegacyExpressionGenome:
    name: str
    expression: str

    def node(self):
        return parse(self.expression)

    def evaluate(self, context):
        """Evaluate with the legacy ``Ctx`` used by the original GA."""
        return context.eval(self.node())

    @property
    def required_fields(self):
        return tuple(sorted({
            item.name for item in _all_nodes(self.node())
            if isinstance(item, Leaf)
        }))

    @property
    def operator_slots(self):
        return tuple(
            item.name for item in _all_nodes(self.node())
            if isinstance(item, Op)
        )

    def validate_registered(self):
        missing = sorted(set(self.operator_slots) - set(OP))
        if missing:
            raise ValueError(f"legacy genome has unregistered operators {missing}")
        return True

    def mutate(self, context, rng=None, max_depth=6):
        """Type the anchor in ``context`` then use the legacy GP mutator."""
        from min_gp.engine import mutate
        node = self.node()
        context.eval(node)
        state = random_module.getstate()
        if rng is not None:
            random_module.setstate(rng.getstate())
        try:
            changed = mutate(node, max_depth)
            if rng is not None:
                rng.setstate(random_module.getstate())
        finally:
            random_module.setstate(state)
        return LegacyExpressionGenome(self.name, str(changed))

    def crossover(self, other, context, rng=None, max_depth=6):
        from min_gp.engine import crossover
        left, right = self.node(), other.node()
        context.eval(left)
        context.eval(right)
        state = random_module.getstate()
        if rng is not None:
            random_module.setstate(rng.getstate())
        try:
            child, _ = crossover(left, right, max_depth)
            if rng is not None:
                rng.setstate(random_module.getstate())
        finally:
            random_module.setstate(state)
        return LegacyExpressionGenome(self.name, str(child))

    def to_dict(self):
        return {
            "kind": "legacy_expression", "name": self.name,
            "expression": self.expression,
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(payload["name"], payload["expression"])

    def __str__(self):
        return str(self.node())


def legacy_seed_population():
    from min_gp.seeds import SEEDS
    genomes = tuple(
        LegacyExpressionGenome(name, expression)
        for name, expression in SEEDS.items()
    )
    for genome in genomes:
        genome.validate_registered()
    return genomes
