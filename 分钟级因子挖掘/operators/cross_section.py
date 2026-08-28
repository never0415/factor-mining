"""Cross-sectional transforms kept separate from raw factor construction."""

import torch

from min_gp.dsl.registry import OperatorRegistry, OperatorSpec
from min_gp.dsl.types import SemanticType
from min_gp.numeric.ranking import cross_section_rank as _fractional_rank


def cross_section_distance(x: torch.Tensor, standardize: bool = False) -> torch.Tensor:
    x = x.float()
    valid = torch.isfinite(x)
    clean = torch.nan_to_num(x)
    count = valid.sum(0, keepdim=True).clamp(min=1)
    mean = clean.sum(0, keepdim=True) / count
    base = x
    if standardize:
        var = (((clean - mean) ** 2) * valid).sum(0, keepdim=True) / count
        std = torch.sqrt(var.clamp(min=0)).clamp(min=1e-12)
        base = (x - mean) / std
        clean_base = torch.nan_to_num(base)
        mean = (clean_base * valid).sum(0, keepdim=True) / count
    out = torch.abs(base - mean)
    return torch.where(valid, out, torch.full_like(out, float("nan")))


def cross_section_rank(x: torch.Tensor) -> torch.Tensor:
    """Daily cross-sectional rank in [0, 1], NaN preserved."""
    return _fractional_rank(x.float())


def cross_section_identity(x: torch.Tensor) -> torch.Tensor:
    """No cross-sectional step.

    待著而救 applies none at all, so "skip it" has to be a choice the search can
    make rather than a structural difference between two hand-written factors.
    """
    return x


def register_cross_section_operators(registry: OperatorRegistry) -> None:
    registry.register(OperatorSpec(
        "cross_section_distance", (SemanticType.DAILY_RAW_FACTOR,),
        SemanticType.DAILY_RAW_FACTOR, cross_section_distance,
        parameter_domains={"standardize": (False, True)},
        needs_full_cross_section=True,
        complexity={"I": 1, "D": 1},
    ))
    registry.register(OperatorSpec(
        "cross_section_rank", (SemanticType.DAILY_RAW_FACTOR,),
        SemanticType.DAILY_RAW_FACTOR, cross_section_rank,
        needs_full_cross_section=True,
        complexity={"I": 1, "D": 1, "logI": 1},
    ))
    registry.register(OperatorSpec(
        "cross_section_identity", (SemanticType.DAILY_RAW_FACTOR,),
        SemanticType.DAILY_RAW_FACTOR, cross_section_identity, cost=0,
        complexity={"I": 1, "D": 1},
    ))
