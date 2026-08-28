"""Prepared, population-batched walk-forward fitness.

The scalar reference implementation in :mod:`min_gp.evaluation.incremental`
is intentionally kept intact.  This module implements the same semantics for
a population while sharing every candidate-independent calculation:

* the rebalance-date return grid and its cross-sectional ranks;
* the trailing-mean baseline and (when requested) its ranks;
* walk-forward train/validation membership.

Candidates are reduced from ``(I, D)`` to ``(I, W)`` immediately after their
strict trailing signal mean, where ``W`` is the number of labelled rebalance
dates.  A population therefore costs ``N * I * W`` rather than ``N * I * D``
of retained memory.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch

from min_gp.evaluation.incremental import (
    IncrementalFitness,
    WalkForwardConfig,
    trailing_signal_mean,
    walk_forward_splits,
)
from min_gp.numeric.preprocessing import remove_outliers
from min_gp.numeric.ranking import cross_section_rank


ProgressCallback = Callable[[int, int], None]


def _ranked_cross_section_ic(
    ranked_factor: torch.Tensor,
    ranked_return: torch.Tensor,
    min_cross_section: int,
) -> torch.Tensor:
    """Pearson correlation of pre-ranked ``(N,I,W)`` and ``(I,W)`` grids."""
    valid = torch.isfinite(ranked_factor) & torch.isfinite(
        ranked_return
    ).unsqueeze(0)
    n_raw = valid.sum(dim=1)
    n = n_raw.clamp(min=1).to(ranked_factor.dtype)
    factor = torch.nan_to_num(ranked_factor)
    returns = torch.nan_to_num(ranked_return).unsqueeze(0)
    factor_mean = (factor * valid).sum(dim=1) / n
    return_mean = (returns * valid).sum(dim=1) / n
    factor_centered = factor - factor_mean.unsqueeze(1)
    return_centered = returns - return_mean.unsqueeze(1)
    covariance = (
        factor_centered * return_centered * valid
    ).sum(dim=1) / n
    factor_scale = torch.sqrt(
        (factor_centered.square() * valid).sum(dim=1) / n
    ).clamp(min=1e-12)
    return_scale = torch.sqrt(
        (return_centered.square() * valid).sum(dim=1) / n
    ).clamp(min=1e-12)
    result = covariance / (factor_scale * return_scale)
    return torch.where(
        n_raw >= min_cross_section,
        result,
        torch.full_like(result, float("nan")),
    )


def _cross_section_residual(
    candidate: torch.Tensor,
    baseline: torch.Tensor,
) -> torch.Tensor:
    """Residualise ``(N,I,W)`` candidates on one ``(I,W)`` baseline."""
    baseline = baseline.unsqueeze(0)
    valid_bool = torch.isfinite(candidate) & torch.isfinite(baseline)
    valid = valid_bool.to(torch.float32)
    candidate = candidate.float()
    baseline = baseline.float()
    clean_candidate = torch.nan_to_num(candidate)
    clean_baseline = torch.nan_to_num(baseline)
    n = valid.sum(dim=1, keepdim=True).clamp(min=2.0)
    baseline_mean = (clean_baseline * valid).sum(dim=1, keepdim=True) / n
    candidate_mean = (clean_candidate * valid).sum(dim=1, keepdim=True) / n
    covariance = (
        (clean_baseline - baseline_mean)
        * (clean_candidate - candidate_mean)
        * valid
    ).sum(dim=1, keepdim=True) / n
    variance = (
        (clean_baseline - baseline_mean).square() * valid
    ).sum(dim=1, keepdim=True) / n
    beta = covariance / variance.clamp(min=1e-12)
    residual = candidate - (
        candidate_mean + beta * (baseline - baseline_mean)
    )
    residual = torch.where(
        valid_bool, residual, torch.full_like(residual, float("nan"))
    )
    return torch.where(
        valid_bool & (residual.abs() < 1e-7),
        torch.zeros_like(residual),
        residual,
    )


def _slice_mask(day_indices: torch.Tensor, window: slice) -> torch.Tensor:
    start = 0 if window.start is None else int(window.start)
    stop = (
        int(day_indices.max().item()) + 1
        if window.stop is None
        else int(window.stop)
    )
    return (day_indices >= start) & (day_indices < stop)


def _portfolio_batch(
    factors: torch.Tensor,
    returns: torch.Tensor,
    day_indices: torch.Tensor,
    validation_windows: Sequence[slice],
    directions: torch.Tensor,
    cfg: WalkForwardConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch equal-weight long-short net return and realised turnover.

    The portfolio is reset to cash at the start of every validation fold, just
    like the scalar ``net_long_short_return`` call.  Candidate-specific dates
    with too few valid stocks are skipped without resetting the previous book.
    """
    population = factors.shape[0]
    folds = len(validation_windows)
    net_result = torch.full(
        (population, folds), float("nan"),
        dtype=torch.float32, device=factors.device,
    )
    turnover_result = torch.full_like(net_result, float("nan"))
    instrument_positions = torch.arange(
        factors.shape[1], device=factors.device
    ).view(1, 1, -1)

    for fold, window in enumerate(validation_windows):
        selected = _slice_mask(day_indices, window)
        if cfg.holding_period > 1:
            start = 0 if window.start is None else int(window.start)
            selected &= (day_indices - start).remainder(
                cfg.holding_period
            ) == 0
        indices = torch.nonzero(selected, as_tuple=False).squeeze(1)
        if indices.numel() == 0:
            continue

        factor = factors[:, :, indices].permute(0, 2, 1).float()
        future = returns[:, indices].T.unsqueeze(0).float()
        oriented = factor * directions[:, fold].view(-1, 1, 1)
        valid = torch.isfinite(oriented) & torch.isfinite(future)
        count = valid.sum(dim=2)
        k = torch.floor(count * cfg.quantile).to(torch.long).clamp(min=1)
        eligible = count >= cfg.min_cross_section

        ordered = torch.argsort(
            torch.where(
                valid, oriented, torch.full_like(oriented, float("inf"))
            ),
            dim=2,
        )
        short = instrument_positions < k.unsqueeze(2)
        long = (
            instrument_positions >= (count - k).unsqueeze(2)
        ) & (instrument_positions < count.unsqueeze(2))
        sorted_weights = torch.zeros_like(oriented)
        unit = k.clamp(min=1).to(oriented.dtype).reciprocal().unsqueeze(2)
        sorted_weights = torch.where(short, -unit, sorted_weights)
        # Match scalar assignment order: in a pathological overlapping book,
        # the long assignment overwrites the short assignment.
        sorted_weights = torch.where(long, unit, sorted_weights)
        weights = torch.zeros_like(sorted_weights).scatter(
            2, ordered, sorted_weights
        )
        weights = torch.where(
            eligible.unsqueeze(2), weights, torch.zeros_like(weights)
        )

        previous = torch.zeros(
            (population, factors.shape[1]),
            dtype=torch.float32, device=factors.device,
        )
        net_sum = torch.zeros(population, device=factors.device)
        turnover_sum = torch.zeros_like(net_sum)
        observations = torch.zeros(
            population, dtype=torch.long, device=factors.device
        )
        clean_future = torch.nan_to_num(future)
        for position in range(indices.numel()):
            usable = eligible[:, position]
            current = weights[:, position]
            gross = (current * clean_future[:, position]).sum(dim=1)
            turnover = 0.5 * (current - previous).abs().sum(dim=1)
            net = gross - turnover * (cfg.cost_bps * 1e-4)
            net_sum += torch.where(usable, net, torch.zeros_like(net))
            turnover_sum += torch.where(
                usable, turnover, torch.zeros_like(turnover)
            )
            observations += usable
            previous = torch.where(usable.unsqueeze(1), current, previous)

        has_observations = observations > 0
        denominator = observations.clamp(min=1).to(net_sum.dtype)
        net_result[:, fold] = torch.where(
            has_observations,
            net_sum / denominator,
            torch.full_like(net_sum, float("nan")),
        )
        turnover_result[:, fold] = torch.where(
            has_observations,
            turnover_sum / denominator,
            torch.full_like(turnover_sum, float("nan")),
        )

    return (
        net_result.detach().cpu().numpy(),
        turnover_result.detach().cpu().numpy(),
    )


class BatchedIncrementalEvaluator:
    """Reusable population evaluator with candidate-independent caches."""

    def __init__(
        self,
        baseline: torch.Tensor,
        fwd_ret: torch.Tensor,
        cfg: WalkForwardConfig,
        neutralizer=None,
    ):
        if baseline.shape != fwd_ret.shape or baseline.ndim != 2:
            raise ValueError("baseline and fwd_ret must share shape (I,D)")
        self.shape = tuple(fwd_ret.shape)
        self.cfg = cfg
        self.neutralizer = neutralizer
        self.splits = walk_forward_splits(self.shape[1], cfg)
        self.eligible_days = torch.isfinite(fwd_ret).any(dim=0)
        self.day_indices = torch.nonzero(
            self.eligible_days, as_tuple=False
        ).squeeze(1)
        self.returns = fwd_ret[:, self.eligible_days].float()
        self.return_ranks = cross_section_rank(self.returns)
        self.return_valid = torch.isfinite(self.returns)
        self.return_pair_denominator = self.return_valid.sum().clamp(min=1)

        prepared_baseline = trailing_signal_mean(
            baseline, cfg.signal_average_days
        )
        if neutralizer is not None:
            prepared_baseline = neutralizer(prepared_baseline)
        self.baseline = prepared_baseline[:, self.eligible_days].float()
        self.baseline_ranks = (
            cross_section_rank(self.baseline)
            if cfg.rank_residual else None
        )

    @property
    def weekly_shape(self) -> tuple[int, int]:
        return self.shape[0], int(self.day_indices.numel())

    def prepare_candidate(self, candidate: torch.Tensor) -> torch.Tensor:
        """Smooth one daily candidate and retain labelled rebalance dates."""
        if tuple(candidate.shape) != self.shape:
            raise ValueError(
                f"candidate shape {tuple(candidate.shape)} != {self.shape}"
            )
        prepared = trailing_signal_mean(
            candidate, self.cfg.signal_average_days
        )
        if self.neutralizer is not None:
            prepared = self.neutralizer(prepared)
        return prepared[:, self.eligible_days].float()

    def evaluate_batch(
        self,
        candidates: torch.Tensor | Sequence[torch.Tensor],
        complexities: Sequence[float],
        *,
        batch_size: int | None = None,
        progress: ProgressCallback | None = None,
    ) -> list[IncrementalFitness]:
        """Score prepared ``(N,I,W)`` candidates in memory-safe chunks."""
        if isinstance(candidates, torch.Tensor):
            values = candidates
        else:
            values = torch.stack(tuple(candidates), dim=0)
        if values.ndim == 2:
            values = values.unsqueeze(0)
        if values.ndim != 3 or tuple(values.shape[1:]) != self.weekly_shape:
            raise ValueError(
                "prepared candidates must have shape "
                f"(N,{self.weekly_shape[0]},{self.weekly_shape[1]})"
            )
        if values.shape[0] != len(complexities):
            raise ValueError("one complexity is required per candidate")
        if not self.splits or self.day_indices.numel() == 0:
            return [IncrementalFitness.invalid() for _ in complexities]
        batch_size = values.shape[0] if batch_size is None else int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        results: list[IncrementalFitness] = []
        for start in range(0, values.shape[0], batch_size):
            stop = min(start + batch_size, values.shape[0])
            chunk = values[start:stop].to(
                device=self.returns.device, dtype=torch.float32
            )
            results.extend(self._evaluate_chunk(
                chunk, complexities[start:stop]
            ))
            if progress is not None:
                progress(stop, values.shape[0])
        return results

    def _evaluate_chunk(
        self,
        candidates: torch.Tensor,
        complexities: Sequence[float],
    ) -> list[IncrementalFitness]:
        cfg = self.cfg
        population = candidates.shape[0]
        invalid = [IncrementalFitness.invalid() for _ in range(population)]

        valid_pairs = torch.isfinite(candidates) & self.return_valid.unsqueeze(0)
        coverage = valid_pairs.sum(dim=(1, 2)).float() / (
            self.return_pair_denominator.to(torch.float32)
        )
        finite = torch.isfinite(candidates)
        count = finite.sum(dim=1)
        clean = torch.nan_to_num(candidates.float())
        mean = clean.sum(dim=1) / count.clamp(min=1)
        variance = (
            (clean - mean.unsqueeze(1)).square() * finite
        ).sum(dim=1) / count.clamp(min=1)
        usable = count >= cfg.min_cross_section
        healthy = usable & torch.isfinite(variance) & (variance > 1e-12)
        usable_count = usable.sum(dim=1)
        healthy_ratio = healthy.sum(dim=1).float() / usable_count.clamp(
            min=1
        ).float()
        active = (coverage >= 0.25) & (
            (usable_count == 0) | (healthy_ratio >= 0.5)
        )
        active_indices = torch.nonzero(active, as_tuple=False).squeeze(1)
        if active_indices.numel() == 0:
            return invalid

        active_candidates = candidates[active_indices]
        # Winsorization sits inside ``daily_spearman_ic`` on the scalar path,
        # so it applies to what IC ranks and not to what the residual is fitted
        # on. Reproducing that split matters: residualising the clipped panel
        # instead of the raw one changes the incremental IC, which is what the
        # scalar-reference test measures.
        ic_ranks = cross_section_rank(
            remove_outliers(active_candidates, n_mad=cfg.outlier_mad, dim=1),
            dim=1,
        )
        ic_series = _ranked_cross_section_ic(
            ic_ranks, self.return_ranks, cfg.min_cross_section
        )
        if cfg.rank_residual:
            residual = _cross_section_residual(
                cross_section_rank(active_candidates, dim=1),
                self.baseline_ranks,
            )
        else:
            residual = _cross_section_residual(
                active_candidates, self.baseline
            )
        # The scalar path feeds the residual through ``daily_spearman_ic`` too,
        # so it is winsorized in turn - on its own scale, not the candidate's.
        residual_ranks = cross_section_rank(
            remove_outliers(residual, n_mad=cfg.outlier_mad, dim=1), dim=1
        )
        incremental_series = _ranked_cross_section_ic(
            residual_ranks, self.return_ranks, cfg.min_cross_section
        )

        ic_numpy = ic_series.detach().cpu().numpy()
        incremental_numpy = incremental_series.detach().cpu().numpy()
        day_numpy = self.day_indices.detach().cpu().numpy()
        validation_windows = [valid for _train, valid in self.splits]
        directions = torch.ones(
            (active_indices.numel(), len(self.splits)),
            dtype=torch.float32, device=candidates.device,
        )
        fold_usable = np.zeros(
            (active_indices.numel(), len(self.splits)), dtype=bool
        )
        fold_ic = np.full_like(fold_usable, np.nan, dtype=np.float64)
        fold_incremental = np.full_like(
            fold_usable, np.nan, dtype=np.float64
        )

        for row in range(active_indices.numel()):
            for fold, (train, valid) in enumerate(self.splits):
                train_mask = (
                    (day_numpy >= (0 if train.start is None else train.start))
                    & (day_numpy < (
                        self.shape[1] if train.stop is None else train.stop
                    ))
                )
                train_values = ic_numpy[row, train_mask]
                train_values = train_values[np.isfinite(train_values)]
                if cfg.direction_mode == "paper":
                    direction = cfg.paper_direction
                else:
                    if train_values.size < cfg.min_valid_ic_days:
                        continue
                    train_ic = float(train_values.mean())
                    if not np.isfinite(train_ic):
                        continue
                    direction = 1 if train_ic >= 0 else -1

                valid_mask = (
                    (day_numpy >= (0 if valid.start is None else valid.start))
                    & (day_numpy < (
                        self.shape[1] if valid.stop is None else valid.stop
                    ))
                )
                valid_values = ic_numpy[row, valid_mask]
                valid_values = valid_values[np.isfinite(valid_values)]
                if valid_values.size < cfg.min_valid_ic_days:
                    continue
                valid_ic = float(valid_values.mean())
                if not np.isfinite(valid_ic):
                    continue
                incremental_values = incremental_numpy[row, valid_mask]
                incremental_values = incremental_values[
                    np.isfinite(incremental_values)
                ]
                incremental_ic = (
                    float(incremental_values.mean())
                    if incremental_values.size else 0.0
                )
                directions[row, fold] = direction
                fold_usable[row, fold] = True
                fold_ic[row, fold] = direction * valid_ic
                fold_incremental[row, fold] = direction * incremental_ic

        fold_net, fold_turnover = _portfolio_batch(
            active_candidates, self.returns, self.day_indices,
            validation_windows, directions, cfg,
        )
        for row, original_index_tensor in enumerate(active_indices):
            original_index = int(original_index_tensor.item())
            usable_folds = fold_usable[row] & np.isfinite(fold_net[row])
            if usable_folds.sum() < cfg.min_folds:
                continue
            aligned_ic = fold_ic[row, usable_folds]
            consistency = float(np.mean(aligned_ic > 0))
            if consistency < cfg.min_fold_consistency:
                continue
            turnover_values = fold_turnover[row, usable_folds]
            turnover = (
                float(np.median(turnover_values))
                if turnover_values.size else float("nan")
            )
            if cfg.max_turnover is not None and not (
                turnover <= cfg.max_turnover
            ):
                continue
            if cfg.direction_mode == "paper":
                settled_direction = cfg.paper_direction
            else:
                finite_ic = ic_numpy[row, np.isfinite(ic_numpy[row])]
                if finite_ic.size == 0:
                    continue
                settled_direction = 1 if float(finite_ic.mean()) >= 0 else -1
            invalid[original_index] = IncrementalFitness(
                robust_ic=float(np.median(aligned_ic)),
                incremental_ic=float(np.median(
                    fold_incremental[row, usable_folds]
                )),
                net_long_short=float(np.median(fold_net[row, usable_folds])),
                complexity=float(complexities[original_index]),
                fold_consistency=consistency,
                coverage=float(coverage[original_index].item()),
                valid=True,
                direction=settled_direction,
                turnover=turnover,
            )
        return invalid
