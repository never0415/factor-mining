"""Registry-driven variation for the event-factor operator genome."""

from dataclasses import replace
import random

from min_gp.factors.event_skeleton import (
    Branch,
    EventSkeletonGenome,
    Slot,
    moderate_risk_anchor,
    slot_candidates,
    wait_rescue_anchor,
)
from min_gp.operators import build_operator_registry


SLOT_KINDS = (
    "statistic", "aggregator", "cross_section", "low_frequency"
)


def random_slot(
    kind: str, rng: random.Random, registry=None, available_fields=None
) -> Slot:
    registry = registry or build_operator_registry()
    candidates = slot_candidates(registry, kind, available_fields)
    if not candidates:
        raise ValueError(
            f"no {kind} operator is compatible with fields "
            f"{sorted(available_fields or ())}"
        )
    spec = rng.choice(candidates)
    params = {
        name: rng.choice(tuple(domain))
        for name, domain in spec.parameter_domains.items()
    }
    return Slot.of(spec.name, **params)


def random_branch(rng: random.Random, registry=None, available_fields=None) -> Branch:
    registry = registry or build_operator_registry()
    return Branch(**{
        kind: random_slot(kind, rng, registry, available_fields)
        for kind in SLOT_KINDS
    })


def random_skeleton(
    rng: random.Random, registry=None, available_fields=None
) -> EventSkeletonGenome:
    registry = registry or build_operator_registry()
    secondary = rng.random() < 0.5
    return EventSkeletonGenome(
        detector=random_slot("detector", rng, registry, available_fields),
        primary=random_branch(rng, registry, available_fields),
        secondary=(
            random_branch(rng, registry, available_fields) if secondary else None
        ),
        combiner=(
            random_slot("combiner", rng, registry, available_fields)
            if secondary else None
        ),
    )


def _replace_branch_slot(branch: Branch, kind: str, slot: Slot) -> Branch:
    return replace(branch, **{kind: slot})


def mutate_skeleton(
    genome: EventSkeletonGenome, rng: random.Random, registry=None,
    available_fields=None,
) -> EventSkeletonGenome:
    """Mutate an operator choice or one of that operator's parameters."""
    registry = registry or build_operator_registry()
    choices = ["detector", "primary"]
    if genome.secondary is not None:
        choices.extend(("secondary", "combiner", "drop_secondary"))
    else:
        choices.append("add_secondary")
    target = rng.choice(choices)
    if target == "detector":
        return replace(genome, detector=random_slot(
            "detector", rng, registry, available_fields
        ))
    if target in ("primary", "secondary"):
        branch = getattr(genome, target)
        kind = rng.choice(SLOT_KINDS)
        changed = _replace_branch_slot(
            branch, kind, random_slot(kind, rng, registry, available_fields)
        )
        return replace(genome, **{target: changed})
    if target == "combiner":
        return replace(genome, combiner=random_slot(
            "combiner", rng, registry, available_fields
        ))
    if target == "drop_secondary":
        return replace(genome, secondary=None, combiner=None)
    return replace(
        genome,
        secondary=random_branch(rng, registry, available_fields),
        combiner=random_slot("combiner", rng, registry, available_fields),
    )


def crossover_skeleton(
    a: EventSkeletonGenome, b: EventSkeletonGenome, rng: random.Random
) -> EventSkeletonGenome:
    """Uniform semantic-slot crossover; every resulting fill is type-valid."""
    primary = Branch(**{
        kind: rng.choice((getattr(a.primary, kind), getattr(b.primary, kind)))
        for kind in SLOT_KINDS
    })
    candidates = [g for g in (a, b) if g.secondary is not None]
    use_secondary = bool(candidates) and rng.random() < 0.5
    if use_secondary:
        source = rng.choice(candidates)
        other = rng.choice(candidates)
        secondary = Branch(**{
            kind: rng.choice((
                getattr(source.secondary, kind), getattr(other.secondary, kind)
            ))
            for kind in SLOT_KINDS
        })
        combiner = rng.choice((source.combiner, other.combiner))
    else:
        secondary = combiner = None
    return EventSkeletonGenome(
        detector=rng.choice((a.detector, b.detector)),
        primary=primary,
        secondary=secondary,
        combiner=combiner,
    )


def skeleton_seed_population(available_fields=None) -> tuple[EventSkeletonGenome, ...]:
    seeds = (moderate_risk_anchor(), wait_rescue_anchor())
    if available_fields is None:
        return seeds
    available_fields = set(available_fields)
    return tuple(
        genome for genome in seeds
        if set(genome.required_fields) <= available_fields
    )
