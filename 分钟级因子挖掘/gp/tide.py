"""Variation operators for the fine-grained Complete-Tide genome."""

from min_gp.factors.event_skeleton import Slot
from min_gp.factors.tide_skeleton import (
    CompleteTideSkeletonGenome, TideBranch, tide_slot_candidates,
)
from min_gp.operators import build_operator_registry


def _random_slot(kind, rng, registry):
    spec = rng.choice(tide_slot_candidates(registry, kind))
    return Slot.of(spec.name, **{
        name: rng.choice(tuple(domain))
        for name, domain in spec.parameter_domains.items()
    })


def _random_branch(rng, registry):
    return TideBranch(
        selector=_random_slot("selector", rng, registry),
        cross_section=_random_slot("cross_section", rng, registry),
        temporal=_random_slot("temporal", rng, registry),
    )


def random_tide_genome(rng, registry=None):
    registry = registry or build_operator_registry()
    return CompleteTideSkeletonGenome(
        **{
            kind: _random_slot(kind, rng, registry)
            for kind in (
                "activity", "pivot", "left_locator", "right_locator",
                "left_speed", "right_speed", "left_activity",
                "right_activity", "combiner",
            )
        },
        strong=_random_branch(rng, registry),
        weak=_random_branch(rng, registry),
        anchor_name=None,
    )


def mutate_tide_genome(genome, rng, registry=None):
    registry = registry or build_operator_registry()
    common = (
        "activity", "pivot", "left_locator", "right_locator",
        "left_speed", "right_speed", "left_activity", "right_activity",
        "combiner",
    )
    branch_slots = tuple(
        (branch, kind)
        for branch in ("strong", "weak")
        for kind in ("selector", "cross_section", "temporal")
    )
    target = rng.choice(tuple(common) + branch_slots)
    values = {name: getattr(genome, name) for name in common}
    strong, weak = genome.strong, genome.weak
    if isinstance(target, str):
        values[target] = _random_slot(target, rng, registry)
    else:
        branch_name, kind = target
        branch = getattr(genome, branch_name)
        changed = TideBranch(**{
            name: _random_slot(kind, rng, registry)
            if name == kind else getattr(branch, name)
            for name in ("selector", "cross_section", "temporal")
        })
        if branch_name == "strong":
            strong = changed
        else:
            weak = changed
    return CompleteTideSkeletonGenome(
        **values, strong=strong, weak=weak, anchor_name=None
    )


def crossover_tide_genomes(left, right, rng):
    def choose(name):
        return rng.choice((getattr(left, name), getattr(right, name)))

    def branch(a, b):
        return TideBranch(
            selector=rng.choice((a.selector, b.selector)),
            cross_section=rng.choice((a.cross_section, b.cross_section)),
            temporal=rng.choice((a.temporal, b.temporal)),
        )

    return CompleteTideSkeletonGenome(
        **{
            name: choose(name) for name in (
                "activity", "pivot", "left_locator", "right_locator",
                "left_speed", "right_speed", "left_activity",
                "right_activity", "combiner",
            )
        },
        strong=branch(left.strong, right.strong),
        weak=branch(left.weak, right.weak),
        anchor_name=None,
    )
