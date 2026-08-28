"""Immutable typed expression nodes."""

from dataclasses import dataclass, field
from typing import Any, Mapping

from min_gp.dsl.registry import OperatorRegistry
from min_gp.dsl.types import ExecutionScope, SemanticType


def _add_extents(left: int | None, right: int | None) -> int | None:
    return None if left is None or right is None else left + right


@dataclass(frozen=True)
class LeafNode:
    name: str
    output_type: SemanticType

    def evaluate(
        self, context: Mapping[str, Any], registry: OperatorRegistry,
        _cache=None,
    ):
        del registry
        del _cache
        if self.name not in context:
            raise KeyError(f"missing leaf {self.name!r}")
        return context[self.name]

    @property
    def complexity(self) -> int:
        return 0

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class ConstantNode:
    """A literal arithmetic terminal, not an operator configuration value."""

    value: int | float
    output_type: SemanticType = field(default=SemanticType.SCALAR, init=False)

    def __post_init__(self):
        if self.value not in (0, 1, 2):
            raise ValueError("literal constants are restricted to {0, 1, 2}")

    def evaluate(self, context, registry, _cache=None):
        del context, registry, _cache
        return self.value

    @property
    def complexity(self) -> int:
        return 0

    def __str__(self) -> str:
        return repr(self.value)


@dataclass(frozen=True)
class OperatorNode:
    name: str
    children: tuple["LeafNode | ConstantNode | OperatorNode", ...]
    params: Mapping[str, Any] = field(default_factory=dict)
    output_type: SemanticType = field(init=False)

    def bind(self, registry: OperatorRegistry) -> "OperatorNode":
        spec = registry.get(self.name)
        for child in self.children:
            if isinstance(child, OperatorNode) and not hasattr(child, "output_type"):
                child.bind(registry)
        child_types = tuple(child.output_type for child in self.children)
        if not all(
            registry.accepts(actual, expected)
            for actual, expected in zip(child_types, spec.input_types)
        ) or len(child_types) != len(spec.input_types):
            raise TypeError(
                f"{self.name}: expected {spec.input_types}, got {child_types}"
            )
        spec.validate_params(self.params)
        object.__setattr__(self, "output_type", spec.output_type)
        return self

    def evaluate(
        self, context: Mapping[str, Any], registry: OperatorRegistry,
        _cache=None,
    ):
        spec = registry.get(self.name)
        if not hasattr(self, "output_type"):
            self.bind(registry)
        cache = {} if _cache is None else _cache
        identity = id(self)
        if identity in cache:
            return cache[identity]
        args = [child.evaluate(context, registry, cache) for child in self.children]
        kwargs = dict(self.params)
        if spec.passes_context:
            kwargs["_context"] = context
        value = spec.implementation(*args, **kwargs)
        cache[identity] = value
        return value

    def complexity_with(self, registry: OperatorRegistry, _seen=None) -> int:
        seen = set() if _seen is None else _seen
        if id(self) in seen:
            return 0
        seen.add(id(self))
        return registry.get(self.name).cost + sum(
            child.complexity_with(registry, seen)
            if isinstance(child, OperatorNode) else child.complexity
            for child in self.children
        )

    def execution_scope_with(self, registry: OperatorRegistry) -> ExecutionScope:
        scope = registry.get(self.name).execution_scope
        for child in self.children:
            if isinstance(child, OperatorNode):
                scope |= child.execution_scope_with(registry)
        return scope

    def history_days_with(self, registry: OperatorRegistry) -> int | None:
        """Effective daily prefix along the longest dependency path."""
        own = registry.get(self.name).resolved_history_days(self.params)
        children = [
            child.history_days_with(registry)
            for child in self.children if isinstance(child, OperatorNode)
        ]
        if not children:
            return own
        child = None if any(value is None for value in children) else max(children)
        return _add_extents(own, child)

    def intraday_lookahead_with(self, registry: OperatorRegistry) -> int | None:
        """Maximum forward-minute dependency along any expression path."""
        own = registry.get(self.name).resolved_intraday_lookahead(self.params)
        children = [
            child.intraday_lookahead_with(registry)
            for child in self.children if isinstance(child, OperatorNode)
        ]
        if not children:
            return own
        child = None if any(value is None for value in children) else max(children)
        return _add_extents(own, child)

    def complexity_profile_with(self, registry: OperatorRegistry) -> dict[str, float]:
        """Dominant exponent for each logical axis across the expression."""
        profile = dict(registry.get(self.name).complexity)
        for child in self.children:
            if not isinstance(child, OperatorNode):
                continue
            for axis, exponent in child.complexity_profile_with(registry).items():
                profile[axis] = max(profile.get(axis, 0.0), exponent)
        return profile

    def __str__(self) -> str:
        args = [str(child) for child in self.children]
        args.extend(f"{k}={v!r}" for k, v in sorted(self.params.items()))
        return f"{self.name}({', '.join(args)})"
