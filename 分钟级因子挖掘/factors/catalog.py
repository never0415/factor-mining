"""Auditable catalog spanning all published factors in the typed DSL.

Every progress field is derived from the executable tree or a production GP's
own acceptance predicate. There are no completion literals.
"""

from dataclasses import dataclass

from min_gp.dsl import OperatorNode, types_compatible
from min_gp.factors.dripping_skeleton import dripping_stone_anchor
from min_gp.factors.event_skeleton import (
    EventSkeletonGenome, moderate_risk_anchor, wait_rescue_anchor,
)
from min_gp.factors.handbook_skeleton import handbook_seed_population
from min_gp.factors.seed_tree import typed_seed_population
from min_gp.operators import build_operator_registry


@dataclass(frozen=True)
class FactorCatalogEntry:
    name: str
    source: str
    backend: str
    gp_family: str
    genome: object
    leaves: tuple[str, ...]
    operator_slots: tuple[str, ...]

    @property
    def evolvable(self):
        return bool(self.operator_slots)


def _event_slots(genome: EventSkeletonGenome):
    branches = [genome.primary] + (
        [genome.secondary] if genome.secondary is not None else []
    )
    values = [genome.detector.operator]
    for branch in branches:
        values.extend((
            branch.statistic.operator, branch.aggregator.operator,
            branch.cross_section.operator, branch.low_frequency.operator,
        ))
    if genome.combiner is not None:
        values.append(genome.combiner.operator)
    return tuple(values)


def build_factor_catalog():
    entries = []
    dripping = dripping_stone_anchor()
    entries.append(FactorCatalogEntry(
        "dripping_stone", "dripping_stone", "typed_slots",
        "dripping_stone", dripping, dripping.required_fields,
        dripping.operator_slots,
    ))
    for name, genome in (
        ("moderate_risk", moderate_risk_anchor()),
        ("wait_rescue", wait_rescue_anchor()),
    ):
        entries.append(FactorCatalogEntry(
            name, "event_handbook", "typed_slots", "event", genome,
            genome.required_fields, _event_slots(genome),
        ))
    for genome in handbook_seed_population():
        entries.append(FactorCatalogEntry(
            genome.anchor_name, "reproduction_handbook", "typed_slots",
            "handbook", genome, genome.required_fields,
            genome.operator_slots,
        ))
    registry = build_operator_registry()
    for genome in typed_seed_population(registry):
        entries.append(FactorCatalogEntry(
            genome.name, "seed_catalog", "typed_tree", "seed_tree", genome,
            genome.required_fields, genome.operator_slots,
        ))
    return tuple(entries)


def _nodes(node):
    yield node
    if isinstance(node, OperatorNode):
        for child in node.children:
            yield from _nodes(child)


def _root(entry, registry):
    expression = getattr(entry.genome, "expression", None)
    if expression is None:
        raise TypeError(f"{entry.name}: genome has no typed expression")
    return expression(registry)[0]


def is_fine_grained(entry, registry=None):
    """Derive decomposition from the tree, never from catalog metadata."""
    registry = registry or build_operator_registry()
    operators = [
        node for node in _nodes(_root(entry, registry))
        if isinstance(node, OperatorNode)
    ]
    return bool(operators) and not any(
        registry.get(node.name).factor_wrapper for node in operators
    )


def production_gp_accepts(entry):
    """Ask the production engine itself whether this genome is accepted."""
    from min_gp.gp.dripping_stone import DrippingStoneGP
    from min_gp.gp.event import EventFactorGP
    from min_gp.gp.handbook import HandbookFactorGP
    from min_gp.gp.seed_tree import SeedTreeFactorGP

    engines = {
        "dripping_stone": DrippingStoneGP,
        "event": EventFactorGP,
        "handbook": HandbookFactorGP,
        "seed_tree": SeedTreeFactorGP,
    }
    engine = engines.get(entry.gp_family)
    return engine is not None and engine.accepts_genome(entry.genome)


def recombination_components(entries=None, registry=None):
    """Weak components of the safe subtree-substitution type graph."""
    entries = tuple(entries or build_factor_catalog())
    registry = registry or build_operator_registry()
    value_types = [
        {node.output_type for node in _nodes(_root(entry, registry))}
        for entry in entries
    ]
    adjacency = [set() for _ in entries]
    for left in range(len(entries)):
        adjacency[left].add(left)
        for right in range(left + 1, len(entries)):
            if any(
                types_compatible(a, b)
                for a in value_types[left] for b in value_types[right]
            ):
                adjacency[left].add(right)
                adjacency[right].add(left)
    unseen = set(range(len(entries)))
    components = []
    while unseen:
        pending = [unseen.pop()]
        found = []
        while pending:
            index = pending.pop()
            found.append(index)
            neighbours = adjacency[index] & unseen
            unseen -= neighbours
            pending.extend(neighbours)
        components.append(tuple(entries[index].name for index in sorted(found)))
    return tuple(sorted(components, key=lambda item: (-len(item), item)))


def directed_recombination_audit(entries=None, registry=None):
    """Describe the actual directional ``donor -> recipient`` crossover graph.

    An edge exists when at least one donor-node output can fill one node type
    in the recipient according to ``registry.accepts``. Strong components are
    the sets that can exchange material in both directions; source/sink
    components are calculated on the SCC condensation graph.
    """
    entries = tuple(entries or build_factor_catalog())
    registry = registry or build_operator_registry()
    value_types = [
        {node.output_type for node in _nodes(_root(entry, registry))}
        for entry in entries
    ]
    adjacency = [set() for _ in entries]
    for donor in range(len(entries)):
        for recipient in range(len(entries)):
            if donor == recipient:
                continue
            if any(
                registry.accepts(donor_type, recipient_type)
                for donor_type in value_types[donor]
                for recipient_type in value_types[recipient]
            ):
                adjacency[donor].add(recipient)

    index = 0
    indices, lowlinks = {}, {}
    stack, on_stack, components = [], set(), []

    def visit(vertex):
        nonlocal index
        indices[vertex] = lowlinks[vertex] = index
        index += 1
        stack.append(vertex)
        on_stack.add(vertex)
        for neighbour in adjacency[vertex]:
            if neighbour not in indices:
                visit(neighbour)
                lowlinks[vertex] = min(lowlinks[vertex], lowlinks[neighbour])
            elif neighbour in on_stack:
                lowlinks[vertex] = min(lowlinks[vertex], indices[neighbour])
        if lowlinks[vertex] != indices[vertex]:
            return
        component = []
        while True:
            member = stack.pop()
            on_stack.remove(member)
            component.append(member)
            if member == vertex:
                break
        components.append(tuple(sorted(component)))

    for vertex in range(len(entries)):
        if vertex not in indices:
            visit(vertex)
    components.sort(key=lambda value: (-len(value), value))
    membership = {
        member: component_index
        for component_index, component in enumerate(components)
        for member in component
    }
    incoming = [set() for _ in components]
    outgoing = [set() for _ in components]
    for donor, recipients in enumerate(adjacency):
        for recipient in recipients:
            left, right = membership[donor], membership[recipient]
            if left != right:
                outgoing[left].add(right)
                incoming[right].add(left)
    source_components = tuple(
        index for index in range(len(components)) if not incoming[index]
    )
    sink_components = tuple(
        index for index in range(len(components)) if not outgoing[index]
    )
    cross_component_edges = sum(
        membership[donor] != membership[recipient]
        for donor, recipients in enumerate(adjacency)
        for recipient in recipients
    )
    return {
        "strong_components": tuple(
            tuple(entries[index].name for index in component)
            for component in components
        ),
        "source_components": tuple(
            tuple(entries[member].name for member in components[index])
            for index in source_components
        ),
        "sink_components": tuple(
            tuple(entries[member].name for member in components[index])
            for index in sink_components
        ),
        "directed_edges": sum(map(len, adjacency)),
        "cross_component_edges": cross_component_edges,
    }


def migration_audit(entries=None):
    """Return concrete migration violations; an empty tuple means complete."""
    entries = tuple(entries or build_factor_catalog())
    registry = build_operator_registry()
    typed = set(registry.names())
    issues = []
    seen = set()
    for entry in entries:
        identity = (entry.source, entry.name)
        if identity in seen:
            issues.append(f"duplicate factor identity: {identity}")
        seen.add(identity)
        if not entry.leaves:
            issues.append(f"{entry.name}: no leaves")
        if not entry.operator_slots:
            issues.append(f"{entry.name}: no evolvable operator slots")
        missing = sorted(set(entry.operator_slots) - typed)
        if missing:
            issues.append(f"{entry.name}: unregistered operators {missing}")
        payload = getattr(entry.genome, "to_dict", lambda: None)()
        if not payload:
            issues.append(f"{entry.name}: genome is not serializable")
        try:
            fine_grained = is_fine_grained(entry, registry)
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(f"{entry.name}: invalid typed expression: {exc}")
        else:
            if not fine_grained:
                issues.append(f"{entry.name}: still contains a whole-factor wrapper")
        if not production_gp_accepts(entry):
            issues.append(f"{entry.name}: production GP rejects its genome type")
    return tuple(issues)


def migration_progress(entries=None):
    entries = tuple(entries or build_factor_catalog())
    registry = build_operator_registry()
    components = recombination_components(entries, registry)
    directed = directed_recombination_audit(entries, registry)
    strong = directed["strong_components"]
    used = {
        node.name
        for entry in entries
        for node in _nodes(_root(entry, registry))
        if isinstance(node, OperatorNode)
    }
    calibrated = {
        name for name in used if registry.get(name).calibration is not None
    }
    zero_cost_exempt = {
        name for name in used if registry.get(name).cost == 0
    }
    calibration_required = used - zero_cost_exempt
    calibration_missing = calibration_required - calibrated
    registry_required = {
        spec.name for spec in registry.specs() if spec.cost > 0
    }
    registry_calibrated = {
        spec.name for spec in registry.specs() if spec.calibration is not None
    }
    return {
        "catalog_complete": len(entries),
        "typed_dsl_migrated": sum(
            entry.backend in {"typed_slots", "typed_tree"} for entry in entries
        ),
        "fine_grained_recombinable": sum(
            is_fine_grained(entry, registry) for entry in entries
        ),
        "production_gp_connected": sum(
            production_gp_accepts(entry) for entry in entries
        ),
        "weak_recombination_components": len(components),
        "weak_component_sizes": tuple(map(len, components)),
        "strong_recombination_components": len(strong),
        "strong_component_sizes": tuple(map(len, strong)),
        "directed_source_factor_count": sum(
            map(len, directed["source_components"])
        ),
        "directed_sink_factor_count": sum(
            map(len, directed["sink_components"])
        ),
        "cross_component_directed_edges": directed["cross_component_edges"],
        "cost_calibrated_operators": len(calibrated),
        "cost_calibration_required": len(calibration_required),
        "cost_zero_cost_exempt": len(zero_cost_exempt),
        "cost_calibration_missing": len(calibration_missing),
        "cost_used_operators": len(used),
        "registry_cost_calibration_required": len(registry_required),
        "registry_cost_calibrated": len(registry_calibrated),
        "registry_cost_calibration_missing": len(
            registry_required - registry_calibrated
        ),
        "registry_zero_cost_exempt": len(registry.specs()) - len(registry_required),
        "total": len(entries),
    }


# Rebuilders keyed by the ``kind`` every genome writes into its export sidecar.
# Registering the family here keeps the gate, the report scripts and the search
# from each carrying their own dispatch and drifting apart.
def _rushing(payload, _registry):
    from min_gp.factors.rushing_skeleton import RushingForwardSkeletonGenome
    return RushingForwardSkeletonGenome.from_dict(payload)


def _event(payload, _registry):
    return EventSkeletonGenome.from_dict(payload)


def _seed_tree(payload, registry):
    from min_gp.factors.seed_tree import SeedTreeGenome
    return SeedTreeGenome.from_dict(payload, registry)


GENOME_REBUILDERS = {
    "rushing_forward_skeleton": _rushing,
    "event_skeleton": _event,
    "typed_seed_tree": _seed_tree,
}


def infer_genome_kind(payload) -> str | None:
    """Name the family of a genome dict written before ``kind`` was stored.

    ``EventSkeletonGenome.to_dict`` still omits the tag, so every record in an
    older run log -- all76.jsonl among them -- arrives untagged. Structure is
    unambiguous enough to recover it: only the event skeleton carries a
    detector beside a primary branch.
    """
    kind = payload.get("kind")
    if kind is not None:
        return kind
    if "detector" in payload and "primary" in payload:
        return "event_skeleton"
    if "root" in payload:
        return "typed_seed_tree"
    return None


def genome_from_export(payload, registry=None):
    """Rebuild whichever genome family produced an exported candidate."""
    kind = infer_genome_kind(payload)
    try:
        rebuild = GENOME_REBUILDERS[kind]
    except KeyError:
        raise ValueError(
            f"no rebuilder registered for genome kind {kind!r}; "
            f"known kinds are {sorted(GENOME_REBUILDERS)}"
        ) from None
    return rebuild(payload, registry or build_operator_registry())


def load_exported_genome(path, registry=None):
    """Read an export sidecar and return (genome, expression, fitness)."""
    import json
    from pathlib import Path

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return (
        genome_from_export(payload["genome"], registry),
        payload.get("expression", ""),
        payload.get("fitness", {}),
    )
