"""Variation operators for the structure-preserving Rushing-Forward island."""

from min_gp.factors.event_skeleton import Slot
from min_gp.factors.rushing_skeleton import (
    RUSHING_SLOT_OPERATORS,
    RushingForwardSkeletonGenome,
    rushing_slot_candidates,
)
from min_gp.operators import build_operator_registry


def _random_slot(kind, rng, registry):
    spec = rng.choice(rushing_slot_candidates(registry, kind))
    params = {
        name: rng.choice(tuple(domain))
        for name, domain in spec.parameter_domains.items()
    }
    if spec.name == "smooth_daily" and params.get("method") == "none":
        params["window"] = 1
    return Slot.of(spec.name, **params)


def random_rushing_genome(rng, registry=None):
    registry = registry or build_operator_registry()
    return RushingForwardSkeletonGenome(
        invert_mask=rng.choice((False, True)),
        **{
            kind: _random_slot(kind, rng, registry)
            for kind in RUSHING_SLOT_OPERATORS
        },
        anchor_name=None,
    )


def mutate_rushing_genome(genome, rng, registry=None):
    registry = registry or build_operator_registry()
    target = rng.choice(("invert_mask", *RUSHING_SLOT_OPERATORS))
    return RushingForwardSkeletonGenome(
        invert_mask=(
            not genome.invert_mask
            if target == "invert_mask" else genome.invert_mask
        ),
        **{
            kind: (
                _random_slot(kind, rng, registry)
                if kind == target else getattr(genome, kind)
            )
            for kind in RUSHING_SLOT_OPERATORS
        },
        anchor_name=None,
    )


def crossover_rushing_genomes(left, right, rng):
    return RushingForwardSkeletonGenome(
        invert_mask=rng.choice((left.invert_mask, right.invert_mask)),
        **{
            kind: rng.choice((getattr(left, kind), getattr(right, kind)))
            for kind in RUSHING_SLOT_OPERATORS
        },
        anchor_name=None,
    )
