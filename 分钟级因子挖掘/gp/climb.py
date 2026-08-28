"""Variation operators for the fine-grained Climb-Mountain genome."""

from min_gp.factors.climb_skeleton import (
    CLIMB_SLOT_OPERATORS, ClimbMountainSkeletonGenome,
    climb_slot_candidates,
)
from min_gp.factors.event_skeleton import Slot
from min_gp.operators import build_operator_registry


def _random_slot(kind, rng, registry):
    spec = rng.choice(climb_slot_candidates(registry, kind))
    return Slot.of(spec.name, **{
        name: rng.choice(tuple(domain))
        for name, domain in spec.parameter_domains.items()
    })


def random_climb_genome(rng, registry=None):
    registry = registry or build_operator_registry()
    return ClimbMountainSkeletonGenome(
        **{
            kind: _random_slot(kind, rng, registry)
            for kind in CLIMB_SLOT_OPERATORS
        },
        anchor_name=None,
    )


def mutate_climb_genome(genome, rng, registry=None):
    registry = registry or build_operator_registry()
    target = rng.choice(tuple(CLIMB_SLOT_OPERATORS))
    return ClimbMountainSkeletonGenome(
        **{
            kind: _random_slot(kind, rng, registry)
            if kind == target else getattr(genome, kind)
            for kind in CLIMB_SLOT_OPERATORS
        },
        anchor_name=None,
    )


def crossover_climb_genomes(left, right, rng):
    return ClimbMountainSkeletonGenome(
        **{
            kind: rng.choice((getattr(left, kind), getattr(right, kind)))
            for kind in CLIMB_SLOT_OPERATORS
        },
        anchor_name=None,
    )
