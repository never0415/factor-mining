"""Cross-sectional ranking, free of any expression-system dependency.

This lives below both DSLs on purpose. Fitness, incremental evaluation, the
correlation archive and the typed cross-section operators all need fractional
ranks, and importing them from `min_gp.expr` made the evaluation layer depend
on the legacy expression runtime -- the wrong direction, and a blocker for
retiring that runtime later.
"""

import torch


def _as_float(t: torch.Tensor) -> torch.Tensor:
    """Promote bool/int inputs, matching the legacy default precision.

    bfloat16 is kept rather than float32 so the move is numerically identical
    to the implementation this replaced; comparisons below use exact equality
    to detect ties, and a wider dtype would split ties differently.
    """
    if t.dtype.is_floating_point:
        return t
    return t.to(torch.bfloat16)


def cross_section_rank(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """NaN-aware fractional rank in [0, 1] with average ranks for ties.

    A double-``argsort`` assigns arbitrary distinct ranks to equal values.
    Event factors contain many ties (event counts, masked medians), so that
    behaviour can manufacture a small IC out of row order alone. Every member
    of a tie group gets its mid-rank instead. Fully vectorised.
    """
    x = _as_float(x)   # defensive: and_/or_ may produce bool
    moved = x.movedim(dim, -1)
    width = moved.shape[-1]
    rows = moved.reshape(-1, width)
    valid = torch.isfinite(rows)
    clean = torch.where(valid, rows, torch.full_like(rows, float("inf")))
    values, order = torch.sort(clean, dim=1, stable=True)
    sorted_valid = torch.gather(valid, 1, order)
    pos = torch.arange(width, device=x.device).expand_as(order)

    new_group = sorted_valid.clone()
    if width > 1:
        new_group[:, 1:] &= (
            ~sorted_valid[:, :-1] | (values[:, 1:] != values[:, :-1])
        )
    starts = torch.where(new_group, pos, torch.zeros_like(pos))
    starts = torch.cummax(starts, dim=1).values

    end_group = sorted_valid.clone()
    if width > 1:
        end_group[:, :-1] &= (
            ~sorted_valid[:, 1:] | (values[:, :-1] != values[:, 1:])
        )
    ends = torch.where(end_group, pos, torch.full_like(pos, width - 1))
    ends = torch.flip(
        torch.cummin(torch.flip(ends, dims=(1,)), dim=1).values, dims=(1,)
    )
    sorted_rank = 0.5 * (starts + ends).to(rows.dtype)
    rank = torch.empty_like(sorted_rank).scatter(1, order, sorted_rank)
    count = valid.sum(1, keepdim=True)
    rank = rank / (count - 1).clamp(min=1).to(rank.dtype)
    rank = torch.where(valid, rank, torch.full_like(rank, float("nan")))
    return rank.reshape(moved.shape).movedim(-1, dim)
