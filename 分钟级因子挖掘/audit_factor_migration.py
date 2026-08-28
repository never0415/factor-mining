"""Print the complete leaf/operator/slot migration inventory."""

from collections import Counter

from min_gp.factors.catalog import (
    build_factor_catalog, directed_recombination_audit, is_fine_grained, migration_audit,
    migration_progress, production_gp_accepts, recombination_components,
)
from min_gp.operators import build_operator_registry


def main():
    entries = build_factor_catalog()
    issues = migration_audit(entries)
    sources = Counter(entry.source for entry in entries)
    backends = Counter(entry.backend for entry in entries)
    registry = build_operator_registry()
    print(f"factors={len(entries)} sources={dict(sources)} backends={dict(backends)}")
    print(f"progress={migration_progress(entries)}")
    for entry in entries:
        print(
            f"{entry.source:22s} {entry.name:28s} "
            f"leaves={len(entry.leaves):2d} slots={len(entry.operator_slots):2d} "
            f"fine_grained={is_fine_grained(entry, registry)} "
            f"gp_connected={production_gp_accepts(entry)}"
        )
    components = recombination_components(entries, registry)
    directed = directed_recombination_audit(entries, registry)
    print(
        f"weak_recombination_components={len(components)} "
        f"sizes={[len(component) for component in components]}"
    )
    print(
        f"strong_recombination_components={len(directed['strong_components'])} "
        f"sizes={[len(component) for component in directed['strong_components']]} "
        f"source_sizes={[len(component) for component in directed['source_components']]} "
        f"sink_sizes={[len(component) for component in directed['sink_components']]} "
        f"cross_component_edges={directed['cross_component_edges']}"
    )
    if issues:
        for issue in issues:
            print(f"ERROR: {issue}")
        raise SystemExit(1)
    print("migration audit: PASS")


if __name__ == "__main__":
    main()
