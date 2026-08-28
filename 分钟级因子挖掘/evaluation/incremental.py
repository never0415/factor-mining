"""Robust walk-forward and baseline-incremental fitness."""

from dataclasses import dataclass, replace

import numpy as np
import torch

from min_gp.numeric.ranking import cross_section_rank as _cs_rank
from min_gp.fitness import daily_spearman_ic

DEFAULT_COST_BPS = 30.0


@dataclass(frozen=True)
class WalkForwardConfig:
    min_train_days: int = 504
    valid_days: int = 126
    n_splits: int = 4
    embargo_days: int = 1
    min_cross_section: int = 30
    min_valid_ic_days: int = 40
    cost_bps: float = DEFAULT_COST_BPS
    quantile: float = 0.2
    holding_period: int = 1
    signal_average_days: int = 1
    direction_mode: str = "paper"  # paper: fixed +1; discovery: infer on train
    paper_direction: int = 1
    # Hard gates. A genome failing either one is rejected outright rather than
    # being pushed onto the Pareto front with a poor objective value, because a
    # factor that only works in some folds is not a weaker factor - it is an
    # unstable one, and NSGA-II would happily trade its instability for another
    # objective.
    # Rebalance cost, not rank quality, is what separates the hold-out winners
    # from the losers: across 76 v1 candidates corr(turnover, net) = -0.62, and
    # every candidate above 0.7 lost money. Turnover is charged inside
    # net_long_short, but that objective trades against IC, so the cap is a
    # gate rather than a term. Units match net_long_short_return: 1.0 is a full
    # replacement of the long-short book.
    max_turnover: float | None = 0.5
    min_fold_consistency: float = 0.75
    min_folds: int = 3
    rank_residual: bool = True
    # Mandatory point-in-time factor winsorization before IC: for every date,
    # clip the stock cross-section to median +/- outlier_mad * raw MAD.
    outlier_mad: float = 5.0
    # Optional hard cut on correlation with the accepted-factor pool. Novelty
    # is carried as a Pareto objective by default (see IncrementalFitness),
    # not as a rejection: a smoothed variant of a pooled factor can correlate
    # 0.9 with it and still beat it on IC and turnover, and throwing that away
    # loses a better factor to keep a diversity rule. The objective lets it
    # survive on the front and be judged against what it actually improves.
    # Set this only to forbid a region outright - for example when mining
    # deliberately for a decorrelated second leg.
    max_pool_correlation: float | None = None

    def __post_init__(self):
        if self.direction_mode not in ("paper", "discovery"):
            raise ValueError("direction_mode must be 'paper' or 'discovery'")
        if self.holding_period < 1:
            raise ValueError("holding_period must be >= 1")
        if self.signal_average_days < 1:
            raise ValueError("signal_average_days must be >= 1")
        if self.paper_direction not in (-1, 1):
            raise ValueError("paper_direction must be -1 or +1")
        if self.outlier_mad < 0:
            raise ValueError("outlier_mad must be non-negative")
        if not 0.0 <= self.min_fold_consistency <= 1.0:
            raise ValueError("min_fold_consistency must lie in [0, 1]")
        if self.min_folds < 1:
            raise ValueError("min_folds must be >= 1")


@dataclass(frozen=True)
class IncrementalFitness:
    robust_ic: float
    incremental_ic: float
    net_long_short: float
    complexity: float
    fold_consistency: float
    coverage: float
    valid: bool = True
    # The tradable sign settled during training. Downstream holdout evaluation
    # must reuse it verbatim; re-inferring the sign out of sample turns a
    # failed factor into a passing one.
    direction: int = 1
    # Median per-rebalance turnover, in the same units as ``max_turnover``.
    turnover: float = float("nan")
    # Largest absolute rank correlation against the accepted-factor pool.
    # 0.0 when no pool was supplied, which makes the novelty objective constant
    # and therefore inert, so an unpooled run behaves exactly as before.
    pool_correlation: float = 0.0

    @classmethod
    def invalid(cls) -> "IncrementalFitness":
        # Novelty is the one objective an invalid genome could otherwise win:
        # its default correlation of 0.0 is the best possible value, which
        # would leave a rejected candidate non-dominated whenever every valid
        # one overlaps the pool. Rejection means worst on every axis.
        return cls(-1e9, -1e9, -1e9, 1e9, 0.0, 0.0, False, 1, float("nan"), 1.0)

    @property
    def objectives(self) -> tuple[float, float, float, float, float]:
        return (
            self.robust_ic,
            self.incremental_ic,
            self.net_long_short,
            -self.complexity,
            # Novelty against the accepted pool, as an objective rather than a
            # cut. A candidate that overlaps the pool is not disqualified; it
            # simply has to be better on IC, net return or cost to earn its
            # place, which is the trade a portfolio actually makes.
            -self.pool_correlation,
        )


def walk_forward_splits(D: int, cfg: WalkForwardConfig):
    splits = []
    train_end = cfg.min_train_days
    for _ in range(cfg.n_splits):
        valid_start = train_end + cfg.embargo_days
        valid_end = valid_start + cfg.valid_days
        if valid_end > D:
            break
        splits.append((slice(0, train_end), slice(valid_start, valid_end)))
        train_end += cfg.valid_days
    return splits


def cross_section_residual(
    y: torch.Tensor, x: torch.Tensor, use_ranks: bool = True
) -> torch.Tensor:
    """Daily cross-sectional residual of y on intercept + x.

    Residualization runs on cross-sectional ranks by default, matching the
    Spearman IC that consumes the result. On raw levels a monotone but
    non-linear relationship between candidate and anchor leaves a large
    residual that still carries the anchor's information, which inflates the
    incremental IC - it can even exceed the candidate's own IC.
    """
    if use_ranks:
        y, x = _cs_rank(y.float()), _cs_rank(x.float())
    valid_bool = torch.isfinite(y) & torch.isfinite(x)
    valid = valid_bool.to(torch.float32)
    yf, xf = y.float(), x.float()
    yc, xc = torch.nan_to_num(yf), torch.nan_to_num(xf)
    n = valid.sum(0, keepdim=True).clamp(min=2.0)
    mx = (xc * valid).sum(0, keepdim=True) / n
    my = (yc * valid).sum(0, keepdim=True) / n
    cov = ((xc - mx) * (yc - my) * valid).sum(0, keepdim=True) / n
    var = ((xc - mx) ** 2 * valid).sum(0, keepdim=True) / n
    beta = cov / var.clamp(min=1e-12)
    residual = yf - (my + beta * (xf - mx))
    residual = torch.where(
        valid_bool, residual, torch.full_like(residual, float("nan"))
    )
    # Identical ranks should leave an exactly constant residual.  Floating
    # point noise would otherwise receive ranks of its own and create a fake
    # incremental IC.
    return torch.where(
        valid_bool & (residual.abs() < 1e-7), torch.zeros_like(residual), residual
    )


def cross_section_residual_multi(
    y: torch.Tensor, anchors, use_ranks: bool = True, ridge: float = 1e-6
) -> torch.Tensor:
    """Daily cross-sectional residual of y on an intercept and every anchor.

    The univariate helper answers what a candidate adds beyond one baseline.
    Once an accepted-factor pool exists the honest question is what it adds
    beyond all of them jointly, because a candidate can sit below the pairwise
    correlation cut against every pool member individually and still be close
    to a linear combination of two of them.

    The single-anchor case delegates to ``cross_section_residual`` so the
    established path keeps its exact numerics.
    """
    anchors = tuple(anchors)
    if not anchors:
        return y.float()
    if len(anchors) == 1:
        return cross_section_residual(y, anchors[0], use_ranks)

    target = _cs_rank(y.float()) if use_ranks else y.float()
    columns = [
        _cs_rank(anchor.float()) if use_ranks else anchor.float()
        for anchor in anchors
    ]
    valid_bool = torch.isfinite(target)
    for column in columns:
        valid_bool &= torch.isfinite(column)
    valid = valid_bool.to(torch.float32)
    design = torch.stack(
        [torch.ones_like(target)] + [torch.nan_to_num(c) for c in columns], dim=2
    )
    weighted = design * valid.unsqueeze(2)
    xtx = torch.einsum("idp,idq->dpq", weighted, weighted)
    xty = torch.einsum("idp,id->dp", weighted, torch.nan_to_num(target) * valid)
    identity = torch.eye(
        xtx.shape[-1], dtype=xtx.dtype, device=xtx.device
    ).expand(xtx.shape[0], -1, -1)
    beta = torch.linalg.solve(xtx + ridge * identity, xty.unsqueeze(2)).squeeze(2)
    fitted = torch.einsum("idp,dp->id", design, beta)
    residual = torch.where(
        valid_bool, target - fitted, torch.full_like(target, float("nan"))
    )
    return torch.where(
        valid_bool & (residual.abs() < 1e-7), torch.zeros_like(residual), residual
    )


def mean_rank_correlation(
    a: torch.Tensor, b: torch.Tensor, min_n: int = 30, pre_ranked: bool = False
) -> float:
    """Mean daily cross-sectional Spearman correlation between two factors.

    Averaging day-local correlations, rather than pooling every cell, matches
    the way IC is measured: two factors that rank names identically each day
    are the same factor regardless of how their levels drift across dates.

    ``pre_ranked`` skips the ranking step for callers that already hold ranked
    panels. Ranking is day-local, so a caller comparing N factors pairwise can
    rank each once instead of ``N(N-1)`` times - at 76 factors that is 5700
    rankings of the full panel reduced to 76. Centering still happens on the
    jointly valid rows of each pair, so the result is unchanged.
    """
    ranked_a = a.float() if pre_ranked else _cs_rank(a.float())
    ranked_b = b.float() if pre_ranked else _cs_rank(b.float())
    valid = torch.isfinite(ranked_a) & torch.isfinite(ranked_b)
    weight = valid.to(torch.float32)
    count = weight.sum(0)
    n = count.clamp(min=1.0)
    av = torch.nan_to_num(ranked_a) * weight
    bv = torch.nan_to_num(ranked_b) * weight
    da = (av - av.sum(0) / n) * weight
    db = (bv - bv.sum(0) / n) * weight
    covariance = (da * db).sum(0) / n
    scale = torch.sqrt((da * da).sum(0) / n) * torch.sqrt((db * db).sum(0) / n)
    daily = covariance / scale.clamp(min=1e-12)
    daily = daily[(count >= min_n) & torch.isfinite(daily)]
    return float(daily.mean().item()) if daily.numel() else float("nan")


def max_pool_correlation(candidate: torch.Tensor, pool, min_n: int = 30) -> float:
    """Largest absolute rank correlation against any factor already accepted.

    Sign is discarded: a candidate correlated -0.95 with a pool member carries
    the same information as that member, merely inverted, and the direction
    step downstream would erase the distinction anyway.
    """
    correlations = [
        abs(mean_rank_correlation(candidate, existing, min_n)) for existing in pool
    ]
    finite = [value for value in correlations if np.isfinite(value)]
    return max(finite) if finite else 0.0


def _mean_ic(
    factor: torch.Tensor,
    returns: torch.Tensor,
    min_cross_section: int,
) -> tuple[float, int]:
    # Weekly labels are deliberately sparse. Ranking the non-rebalance
    # columns cannot affect the result and only multiplies GPU work.
    eligible_days = torch.isfinite(returns).any(dim=0)
    factor, returns = factor[:, eligible_days], returns[:, eligible_days]
    if returns.shape[1] == 0:
        return float("nan"), 0
    ic = daily_spearman_ic(factor, returns, min_n=min_cross_section)
    valid = ic[torch.isfinite(ic)]
    if valid.numel() == 0:
        return float("nan"), 0
    return float(valid.mean().item()), int(valid.numel())


def _window_mean(series, day_indices, window):
    """Mean a precomputed sparse daily series inside an original-grid slice."""
    start = 0 if window.start is None else window.start
    stop = int(day_indices.max().item()) + 1 if window.stop is None else window.stop
    selected = series[(day_indices >= start) & (day_indices < stop)]
    selected = selected[torch.isfinite(selected)]
    if selected.numel() == 0:
        return float("nan"), 0
    return float(selected.mean().item()), int(selected.numel())


def trailing_signal_mean(factor: torch.Tensor, window: int) -> torch.Tensor:
    """Strict trailing trading-day mean, including the current day."""
    if factor.ndim != 2:
        raise ValueError("factor must have shape (I,D)")
    if window < 1:
        raise ValueError("window must be >= 1")
    value = factor.float()
    if window == 1:
        return value
    valid = torch.isfinite(value)
    clean = torch.nan_to_num(value)
    sums = torch.cumsum(clean, dim=1)
    counts = torch.cumsum(valid.to(torch.int32), dim=1)
    prefix_sum = torch.zeros_like(sums[:, :1])
    prefix_count = torch.zeros_like(counts[:, :1])
    sums = torch.cat((prefix_sum, sums), dim=1)
    counts = torch.cat((prefix_count, counts), dim=1)
    rolling_sum = sums[:, window:] - sums[:, :-window]
    rolling_count = counts[:, window:] - counts[:, :-window]
    tail = rolling_sum / rolling_count.clamp(min=1).to(rolling_sum.dtype)
    tail = torch.where(
        rolling_count == window, tail, torch.full_like(tail, float("nan"))
    )
    head = torch.full(
        (factor.shape[0], window - 1), float("nan"),
        device=factor.device, dtype=tail.dtype,
    )
    return torch.cat((head, tail), dim=1)


def net_long_short_return(
    factor: torch.Tensor,
    returns: torch.Tensor,
    direction: int,
    quantile: float,
    cost_bps: float,
    min_cross_section: int,
    rebalance_period: int = 1,
) -> float:
    """Equal-weight long-short return after target-weight turnover cost.

    ``returns`` contains horizon returns, not one-day returns.  For a horizon
    longer than one day we therefore score non-overlapping rebalance dates;
    otherwise the same future price moves and a fresh transaction cost would
    both be counted on every adjacent day.
    """
    if rebalance_period < 1:
        raise ValueError("rebalance_period must be >= 1")
    I, D = factor.shape
    prev = torch.zeros(I, device=factor.device, dtype=torch.float32)
    daily, traded = [], []
    eligible_days = torch.nonzero(
        torch.isfinite(returns).any(dim=0), as_tuple=False
    ).squeeze(1)
    if rebalance_period > 1:
        eligible_days = eligible_days[
            eligible_days.remainder(rebalance_period) == 0
        ]
    for d in eligible_days.tolist():
        f, r = factor[:, d].float() * direction, returns[:, d].float()
        valid = torch.isfinite(f) & torch.isfinite(r)
        n = int(valid.sum().item())
        if n < min_cross_section:
            continue
        idx_valid = torch.nonzero(valid, as_tuple=False).squeeze(1)
        k = max(1, int(n * quantile))
        order = torch.argsort(f[idx_valid])
        short_idx = idx_valid[order[:k]]
        long_idx = idx_valid[order[-k:]]
        weights = torch.zeros_like(prev)
        weights[long_idx] = 1.0 / k
        weights[short_idx] = -1.0 / k
        gross = (weights * torch.nan_to_num(r)).sum()
        # Half the absolute weight change: one unit is a full replacement of a
        # book whose gross exposure is 2 (one long leg, one short leg).
        turnover = 0.5 * (weights - prev).abs().sum()
        net = gross - turnover * (cost_bps * 1e-4)
        daily.append(float(net.item()))
        traded.append(float(turnover.item()))
        prev = weights
    if not daily:
        return float("nan"), float("nan")
    return float(np.mean(daily)), float(np.mean(traded))


def evaluate_incremental_fitness(
    candidate: torch.Tensor,
    baseline: torch.Tensor,
    fwd_ret: torch.Tensor,
    complexity: float,
    cfg: WalkForwardConfig,
    neutralizer=None,
    pool=(),
) -> IncrementalFitness:
    """Score one candidate against a baseline anchor and an accepted pool.

    ``pool`` holds factors already kept for trading. They join the baseline in
    the residual that defines ``incremental_ic``, and the candidate's largest
    absolute rank correlation against them is gated separately, so a rediscovery
    of something already held cannot occupy a slot on the Pareto front.
    """
    pool = tuple(pool)
    if candidate.shape != baseline.shape or candidate.shape != fwd_ret.shape:
        raise ValueError("candidate, baseline and fwd_ret must have equal shapes")
    if any(existing.shape != candidate.shape for existing in pool):
        raise ValueError("pool factors must share the candidate grid")
    candidate = trailing_signal_mean(candidate, cfg.signal_average_days)
    baseline = trailing_signal_mean(baseline, cfg.signal_average_days)
    pool = tuple(
        trailing_signal_mean(existing, cfg.signal_average_days) for existing in pool
    )
    if neutralizer is not None:
        # Strip the style exposure before anything is scored. Applied after the
        # signal average because that is the order a live signal would run in:
        # form, smooth, then neutralise at trade time. The baseline is stripped
        # too, so the incremental IC asks what is new beyond the anchor once
        # both have given up the same exposure rather than rewarding a
        # candidate for simply carrying more of it.
        candidate = neutralizer(candidate)
        baseline = neutralizer(baseline)
        pool = tuple(neutralizer(existing) for existing in pool)
    valid_pairs = torch.isfinite(candidate) & torch.isfinite(fwd_ret)
    eligible = torch.isfinite(fwd_ret)
    coverage = float(
        valid_pairs.sum().float().div(eligible.sum().clamp(min=1)).item()
    )
    if coverage < 0.25:
        return IncrementalFitness.invalid()
    # Train-time health gate: reject constant/nearly constant daily sections
    # before rank evaluation, otherwise double-argsort tie ordering can create a
    # false IC from an economically degenerate factor.
    valid_f = torch.isfinite(candidate)
    count = valid_f.sum(0)
    clean = torch.nan_to_num(candidate.float())
    mean = clean.sum(0) / count.clamp(min=1)
    variance = (
        ((clean - mean.unsqueeze(0)) ** 2) * valid_f
    ).sum(0) / count.clamp(min=1)
    usable_days = (count >= cfg.min_cross_section) & eligible.any(0)
    healthy_days = usable_days & torch.isfinite(variance) & (variance > 1e-12)
    healthy_ratio = healthy_days.sum() / usable_days.sum().clamp(min=1)
    if usable_days.any() and healthy_ratio.item() < 0.5:
        return IncrementalFitness.invalid()
    splits = walk_forward_splits(candidate.shape[1], cfg)
    if not splits:
        return IncrementalFitness.invalid()

    # Cross-sectional ranks and residuals are day-local. Compute them once on
    # the sparse label dates, then slice the resulting IC series by fold. The
    # former implementation recomputed overlapping train ranks in every fold.
    eligible_days = eligible.any(0)
    day_indices = torch.nonzero(
        eligible_days, as_tuple=False
    ).squeeze(1)
    weekly_candidate = candidate[:, eligible_days]
    weekly_returns = fwd_ret[:, eligible_days]
    weekly_baseline = baseline[:, eligible_days]
    weekly_pool = tuple(existing[:, eligible_days] for existing in pool)
    # Rejected before the fold loop: a rediscovery is cheap to detect and
    # expensive to score.
    pool_correlation = (
        max_pool_correlation(weekly_candidate, weekly_pool, cfg.min_cross_section)
        if weekly_pool else 0.0
    )
    if cfg.max_pool_correlation is not None and not (
        pool_correlation <= cfg.max_pool_correlation
    ):
        # Carry the measured value out with the rejection so a run log can
        # distinguish a rediscovery from a genuinely broken candidate.
        return replace(
            IncrementalFitness.invalid(), pool_correlation=pool_correlation
        )
    residual = cross_section_residual_multi(
        weekly_candidate, (weekly_baseline,) + weekly_pool,
        use_ranks=cfg.rank_residual,
    )
    ic_series = daily_spearman_ic(
        weekly_candidate, weekly_returns, min_n=cfg.min_cross_section,
        outlier_mad=cfg.outlier_mad,
    )
    incremental_series = daily_spearman_ic(
        residual, weekly_returns, min_n=cfg.min_cross_section,
        outlier_mad=cfg.outlier_mad,
    )
    fold_ic, fold_inc, fold_net, fold_dir = [], [], [], []
    fold_turnover = []
    for train, valid in splits:
        if cfg.direction_mode == "paper":
            direction = cfg.paper_direction
        else:
            train_ic, train_days = _window_mean(
                ic_series, day_indices, train
            )
            if train_days < cfg.min_valid_ic_days or not np.isfinite(train_ic):
                continue
            direction = 1 if train_ic >= 0 else -1

        valid_ic, n_days = _window_mean(ic_series, day_indices, valid)
        if n_days < cfg.min_valid_ic_days or not np.isfinite(valid_ic):
            continue
        inc_ic, _ = _window_mean(incremental_series, day_indices, valid)
        # The anchor itself has zero residual; treat its incremental signal as 0.
        if not np.isfinite(inc_ic):
            inc_ic = 0.0
        net, traded = net_long_short_return(
            candidate[:, valid], fwd_ret[:, valid], direction,
            cfg.quantile, cfg.cost_bps, cfg.min_cross_section,
            cfg.holding_period,
        )
        if not np.isfinite(net):
            continue
        fold_turnover.append(traded)
        fold_ic.append(direction * valid_ic)
        fold_inc.append(direction * inc_ic)
        fold_net.append(net)
        fold_dir.append(direction)

    if len(fold_ic) < cfg.min_folds:
        return IncrementalFitness.invalid()
    consistency = float(np.mean(np.asarray(fold_ic) > 0))
    if consistency < cfg.min_fold_consistency:
        return IncrementalFitness.invalid()
    turnover = float(np.median(fold_turnover)) if fold_turnover else float("nan")
    if cfg.max_turnover is not None and not (turnover <= cfg.max_turnover):
        return IncrementalFitness.invalid()
    if cfg.direction_mode == "paper":
        settled_direction = cfg.paper_direction
    else:
        finite_ic = ic_series[torch.isfinite(ic_series)]
        full_ic = float(finite_ic.mean().item())
        settled_direction = 1 if full_ic >= 0 else -1
    return IncrementalFitness(
        robust_ic=float(np.median(fold_ic)),
        incremental_ic=float(np.median(fold_inc)),
        net_long_short=float(np.median(fold_net)),
        complexity=float(complexity),
        fold_consistency=consistency,
        coverage=coverage,
        valid=True,   # every hard gate above already passed
        direction=settled_direction,
        turnover=turnover,
        pool_correlation=pool_correlation,
    )
