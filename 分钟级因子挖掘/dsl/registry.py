"""Operator metadata and validation for the strongly typed GP grammar."""

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping

from min_gp.dsl.types import ExecutionScope, SemanticType, is_assignable


Extent = int | None | Callable[[Mapping[str, Any]], int | None]


def _resolve_extent(value: Extent, params: Mapping[str, Any]) -> int | None:
    resolved = value(params) if callable(value) else value
    if resolved is not None and resolved < 0:
        raise ValueError(f"execution extent must be non-negative, got {resolved}")
    return resolved


@dataclass(frozen=True)
class CostCalibration:
    """One measured runtime/memory point used for asymptotic extrapolation."""

    reference_shape: Mapping[str, int]
    seconds: float
    peak_bytes: int | None = None
    device: str = "unknown"
    source: str = "benchmark"
    parameter_values: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.seconds < 0:
            raise ValueError("calibrated seconds must be non-negative")
        if self.peak_bytes is not None and self.peak_bytes < 0:
            raise ValueError("calibrated peak_bytes must be non-negative")
        if any(value <= 0 for value in self.reference_shape.values()):
            raise ValueError("calibration reference dimensions must be positive")

    def matches(self, params: Mapping[str, Any]) -> bool:
        return all(params.get(name) == value for name, value in self.parameter_values.items())


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    input_types: tuple[SemanticType, ...]
    output_type: SemanticType
    implementation: Callable[..., Any]
    cost: int = 1
    # A very small number of shape-changing operators (notably daily ->
    # minute broadcast) need the current evaluation context to recover an
    # axis length.  They receive it as the keyword-only ``_context``.
    passes_context: bool = False
    parameter_domains: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)
    needs_full_cross_section: bool = False
    needs_history: bool = False
    # Number of preceding daily observations required.  None means the entire
    # prefix (for example a recursively seeded EMA).
    history_days: Extent = 0
    # Kept separate from execution scope: this is an information-set audit,
    # not a chunking requirement.
    intraday_lookahead_minutes: Extent = 0
    # Asymptotic exponents by logical axis, e.g. {"I": 2, "D": 1}.
    # Runtime calibration remains separate from this structural description.
    complexity: Mapping[str, float] = field(default_factory=dict)
    memory_complexity: Mapping[str, float] = field(default_factory=dict)
    calibration: CostCalibration | None = None
    # True only for compatibility wrappers that evaluate an entire published
    # factor.  Such a node is valid as a numerical baseline, but does not count
    # as evidence of fine-grained decomposition.
    factor_wrapper: bool = False

    @property
    def execution_scope(self) -> ExecutionScope:
        scope = ExecutionScope.LOCAL
        if self.needs_full_cross_section:
            scope |= ExecutionScope.FULL_CROSS_SECTION
        if self.needs_history:
            scope |= ExecutionScope.HISTORY
        return scope

    def resolved_history_days(self, params: Mapping[str, Any]) -> int | None:
        if not self.needs_history:
            return 0
        return _resolve_extent(self.history_days, params)

    def resolved_intraday_lookahead(
        self, params: Mapping[str, Any]
    ) -> int | None:
        return _resolve_extent(self.intraday_lookahead_minutes, params)

    def validate_params(self, params: Mapping[str, Any]) -> None:
        unknown = set(params) - set(self.parameter_domains)
        if unknown:
            raise TypeError(f"{self.name}: unknown parameters {sorted(unknown)}")
        for name, domain in self.parameter_domains.items():
            if name not in params:
                raise TypeError(f"{self.name}: missing parameter {name!r}")
            if domain and params[name] not in domain:
                raise ValueError(
                    f"{self.name}: {name}={params[name]!r} not in {domain!r}"
                )


class OperatorRegistry:
    def __init__(self):
        self._specs: dict[str, OperatorSpec] = {}

    def register(self, spec: OperatorSpec) -> None:
        if spec.name in self._specs:
            raise KeyError(f"operator already registered: {spec.name}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> OperatorSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise KeyError(f"unknown typed operator: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def specs(self) -> tuple[OperatorSpec, ...]:
        return tuple(self._specs[name] for name in sorted(self._specs))

    def replace(self, spec: OperatorSpec) -> None:
        if spec.name not in self._specs:
            raise KeyError(f"cannot replace unknown operator: {spec.name}")
        self._specs[spec.name] = spec

    def apply_calibration(
        self, name: str, calibration: CostCalibration,
        *, complexity: Mapping[str, float] | None = None,
        memory_complexity: Mapping[str, float] | None = None,
    ) -> None:
        spec = self.get(name)
        self.replace(replace(
            spec,
            calibration=calibration,
            complexity=(
                spec.complexity if complexity is None else dict(complexity)
            ),
            memory_complexity=(
                spec.memory_complexity
                if memory_complexity is None else dict(memory_complexity)
            ),
        ))

    def matching(
        self,
        output_type: SemanticType,
        input_types: tuple[SemanticType, ...] | None = None,
        arity: int | None = None,
        compatible: bool = False,
    ) -> tuple[OperatorSpec, ...]:
        """Every operator that can fill a slot with this shape.

        Slot candidates are derived from the registry rather than listed by
        hand, so registering a new operator puts it into the search space
        automatically and no separate table can fall out of sync.
        """
        found = []
        for spec in self.specs():
            if (
                spec.output_type != output_type
                and not (compatible and is_assignable(spec.output_type, output_type))
            ):
                continue
            if input_types is not None and spec.input_types != input_types:
                continue
            if arity is not None and len(spec.input_types) != arity:
                continue
            found.append(spec)
        return tuple(found)

    @staticmethod
    def accepts(actual: SemanticType, expected: SemanticType) -> bool:
        return is_assignable(actual, expected)
