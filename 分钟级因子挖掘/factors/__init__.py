"""Exact factor templates and their constrained variants."""

from min_gp.factors.dripping_stone import DrippingStoneTemplate
from min_gp.factors.dripping_skeleton import (
    DrippingSkeletonGenome, dripping_stone_anchor,
)
from min_gp.factors.event_factors import ModerateRiskTemplate, WaitRescueTemplate
from min_gp.factors.event_skeleton import EventSkeletonGenome
from min_gp.factors.handbook import (
    HANDBOOK_FACTORS, LOCAL_MINUTE_FACTORS, evaluate_local_minute_factor,
)
from min_gp.factors.handbook_skeleton import (
    HandbookSkeletonGenome, handbook_anchor, handbook_seed_population,
)
from min_gp.factors.tide_skeleton import (
    CompleteTideSkeletonGenome, TideBranch, complete_tide_anchor,
)
from min_gp.factors.climb_skeleton import (
    ClimbMountainSkeletonGenome, climb_mountain_anchor,
)


# The legacy adapter and migration catalog both import ``min_gp.expr``.  Keep
# their historical package-level exports, but resolve them only when an old
# caller explicitly asks for one.  Typed factor imports must not pull the old
# expression runtime into memory as a side effect of importing this package.
_LAZY_LEGACY_EXPORTS = {
    "LegacyExpressionGenome": ("min_gp.factors.legacy", "LegacyExpressionGenome"),
    "legacy_seed_population": ("min_gp.factors.legacy", "legacy_seed_population"),
    "build_factor_catalog": ("min_gp.factors.catalog", "build_factor_catalog"),
    "migration_audit": ("min_gp.factors.catalog", "migration_audit"),
}


def __getattr__(name):
    target = _LAZY_LEGACY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value

__all__ = [
    "DrippingStoneTemplate", "ModerateRiskTemplate", "WaitRescueTemplate",
    "DrippingSkeletonGenome", "dripping_stone_anchor",
    "EventSkeletonGenome",
    "HANDBOOK_FACTORS",
    "LOCAL_MINUTE_FACTORS",
    "evaluate_local_minute_factor",
    "HandbookSkeletonGenome",
    "handbook_anchor",
    "handbook_seed_population",
    "CompleteTideSkeletonGenome",
    "TideBranch",
    "complete_tide_anchor",
    "ClimbMountainSkeletonGenome",
    "climb_mountain_anchor",
    "LegacyExpressionGenome",
    "legacy_seed_population",
    "build_factor_catalog",
    "migration_audit",
]
