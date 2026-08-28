"""Correlation-aware archive for retaining genuinely distinct factors."""

from dataclasses import dataclass, field

import torch

from min_gp.numeric.ranking import cross_section_rank as _cs_rank


def factor_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach().cpu().float(), b.detach().cpu().float()
    ra, rb = _cs_rank(a), _cs_rank(b)
    valid = torch.isfinite(ra) & torch.isfinite(rb)
    if int(valid.sum()) < 2:
        return float("nan")
    x, y = ra[valid], rb[valid]
    x, y = x - x.mean(), y - y.mean()
    denom = torch.sqrt((x * x).sum() * (y * y).sum())
    if denom <= 0:
        return float("nan")
    return float(((x * y).sum() / denom).item())


@dataclass
class FactorArchive:
    max_abs_correlation: float = 0.85
    entries: list[tuple[object, torch.Tensor]] = field(default_factory=list)

    def add(self, metadata, factor: torch.Tensor) -> tuple[bool, float]:
        correlations = [
            abs(factor_correlation(factor, existing))
            for _, existing in self.entries
        ]
        finite = [value for value in correlations if value == value]
        maximum = max(finite, default=0.0)
        if maximum >= self.max_abs_correlation:
            return False, maximum
        self.entries.append((metadata, factor.detach().cpu()))
        return True, maximum
