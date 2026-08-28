"""Point-in-time cross-sectional preprocessing shared by fitness and reports."""

from __future__ import annotations

import torch


def remove_outliers(
    factor: torch.Tensor,
    n_mad: float = 5.0,
    dim: int = 0,
) -> torch.Tensor:
    """Winsorize each cross-section to median +/- ``n_mad`` raw MAD.

    The operation is date-local and therefore introduces no look-ahead. Only
    finite factor observations enter the median and MAD; missing/non-finite
    inputs remain missing. ``MAD`` is the unscaled median absolute deviation
    requested by the research protocol (there is deliberately no 1.4826
    normal-consistency multiplier).
    """
    if n_mad < 0:
        raise ValueError("n_mad must be non-negative")
    value = factor.float()
    finite = torch.isfinite(value)
    clean = torch.where(finite, value, torch.full_like(value, float("nan")))
    median = torch.nanquantile(clean, 0.5, dim=dim, keepdim=True)
    deviation = torch.abs(clean - median)
    mad = torch.nanquantile(deviation, 0.5, dim=dim, keepdim=True)
    lower, upper = median - n_mad * mad, median + n_mad * mad
    clipped = torch.maximum(torch.minimum(clean, upper), lower)
    return torch.where(finite, clipped, torch.full_like(clipped, float("nan")))


def industry_one_hot(
    industry: torch.Tensor,
    levels: int | None = None,
) -> torch.Tensor:
    """Expand an ``(I,D)`` integer industry grid into ``(I,D,K)`` dummies.

    ``load_daily_exposures`` already turns the ``sw_level1`` strings into
    point-in-time integer codes via a pandas categorical, so the string side of
    the mapping lives there and this function only widens codes into columns.
    Negative codes mark a stock with no industry on that date and produce an
    all-zero row rather than a column of their own -- such rows are excluded
    from the fit by the caller's validity mask instead of forming a residual
    "unknown industry" bucket that would absorb genuine signal.

    ``levels`` fixes the column count for one call. It does not need to agree
    between evaluation windows -- each date is fitted on its own, no
    coefficient is carried across stages, and a window that happens to contain
    36 industries rather than 38 simply has two all-zero columns fewer. What it
    must cover is every code present in the grid it is given, which is why the
    caller passes the level count from the same ``load_daily_exposures`` call
    that produced the codes rather than recomputing a maximum here.
    """
    if industry.ndim != 2:
        raise ValueError("industry grid must have shape (instrument, date)")
    codes = industry.long()
    width = int(codes.max().item()) + 1 if levels is None else int(levels)
    if width < 1:
        raise ValueError("industry design needs at least one level")
    positions = torch.arange(width, device=codes.device).view(1, 1, -1)
    return (codes.unsqueeze(2) == positions).float()


def neutralize(
    factor: torch.Tensor,
    industry: torch.Tensor | None = None,
    levels: int | None = None,
    continuous: tuple[torch.Tensor, ...] = (),
    min_cross_section: int = 30,
) -> torch.Tensor:
    """Replace a factor with its daily cross-sectional OLS residual.

    ``Y`` is the factor and ``X`` is the industry dummy block beside every
    continuous exposure (log float market cap in the standard protocol). The
    normal equations are solved with ``pinv`` rather than a ridge-regularised
    ``solve`` because the design is routinely rank-deficient: a full industry
    one-hot spans the intercept, and on any given date a CSI 500 slice can hold
    no members of some industry at all, leaving that column exactly zero. The
    pseudo-inverse returns the minimum-norm solution in both cases, which fits
    the identical subspace and therefore yields the same residual, whereas a
    plain solve would fail or return noise scaled by the ridge term.

    Because the industry dummies already span the intercept, no separate
    constant column is added unless there are no industries at all.

    A stock is dropped (returned as NaN) on any date where the factor, its
    industry, or any exposure is missing: a name that cannot be neutralised
    must not be ranked in the same cross-section as names that were, or it
    keeps the very exposure the residual is meant to remove. Dates with fewer
    than ``min_cross_section`` usable stocks are dropped entirely.
    """
    if factor.ndim != 2:
        raise ValueError("factor must have shape (instrument, date)")
    if industry is None and not continuous:
        raise ValueError("neutralization needs an industry or a continuous exposure")
    if min_cross_section < 1:
        raise ValueError("min_cross_section must be positive")
    value = factor.float()
    if industry is not None and industry.shape != factor.shape:
        raise ValueError("industry grid does not align with the factor")
    for exposure in continuous:
        if exposure.shape != factor.shape:
            raise ValueError("continuous exposure does not align with the factor")

    # Validity is settled before the design is built, because the continuous
    # columns are standardised over exactly the rows that enter the fit.
    valid = torch.isfinite(value)
    if industry is not None:
        valid &= industry >= 0
    for exposure in continuous:
        valid &= torch.isfinite(exposure.float())

    columns = []
    if industry is not None:
        columns.append(industry_one_hot(industry, levels))
    else:
        columns.append(torch.ones(*factor.shape, 1, device=value.device))
    weight = valid.to(value.dtype)
    count = weight.sum(0, keepdim=True).clamp(min=1)
    for exposure in continuous:
        # Centre and scale each date's exposure. The industry dummies span the
        # intercept, so this leaves the column space -- and therefore the
        # residual -- unchanged, but it keeps the normal matrix well
        # conditioned. Raw ln(float market cap) sits near 23 while a dummy is
        # 1, and squaring that ratio into X'X costs enough precision that a
        # factor exactly linear in the exposure kept a residual correlation of
        # -0.03 instead of 0.
        finite = torch.nan_to_num(exposure.float())
        mean = (finite * weight).sum(0, keepdim=True) / count
        # Zeroing must follow the fill: a non-finite exposure times a zero
        # weight is still non-finite, and one such cell makes the whole date's
        # SVD fail rather than simply dropping that row from the fit.
        centered = (finite - mean) * weight
        scale = (centered.square().sum(0, keepdim=True) / count).sqrt()
        columns.append((centered / scale.clamp(min=1e-8)).unsqueeze(2))
    design = torch.cat(columns, dim=2)

    # Solved in double precision: the pseudo-inverse of a rank-deficient normal
    # matrix separates kept from discarded directions by singular-value
    # magnitude, and float32 leaves that threshold too close to the noise.
    weighted = (design * weight.unsqueeze(2)).double()
    target = torch.where(valid, value, torch.zeros_like(value)).double()
    normal = torch.einsum("idp,idq->dpq", weighted, weighted)
    moment = torch.einsum("idp,id->dp", weighted, target)
    beta = torch.linalg.pinv(normal) @ moment.unsqueeze(2)
    fitted = torch.einsum("idp,dp->id", design.double(), beta.squeeze(2))
    enough = valid.sum(0) >= min_cross_section
    keep = valid & enough.unsqueeze(0)
    residual = (value.double() - fitted).to(value.dtype)
    return torch.where(keep, residual, torch.full_like(value, float("nan")))


def align_signal(factor: torch.Tensor, offset: str = "close") -> torch.Tensor:
    """Delay a close-formed signal by one trading day.

    ``offset="close"`` shifts the panel one column to the right, so the value
    computed from date ``t``'s closing data is scored against date ``t+1``'s
    label. The first column becomes NaN because no prior signal exists.

    This is an execution assumption, not a look-ahead repair: intraday
    look-ahead is audited structurally by ``intraday_lookahead_minutes`` and
    cross-day look-ahead is prevented inside the operators themselves. Shifting
    trades one day of signal freshness for the guarantee that nothing formed at
    a close is ever credited with that same close's fill.
    """
    if offset in (None, "none"):
        return factor
    if offset != "close":
        raise ValueError(f"unknown alignment offset: {offset!r}")
    if factor.ndim != 2:
        raise ValueError("factor must have shape (instrument, date)")
    shifted = torch.full_like(factor, float("nan"))
    shifted[:, 1:] = factor[:, :-1]
    return shifted
