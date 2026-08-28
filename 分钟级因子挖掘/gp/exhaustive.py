"""Exhaustive search over event-factor template spaces.

The handbook event families are small: WaitRescue has 324 genomes and
ModerateRisk 768, both at or below a single GP island's evaluation budget. A
GP would therefore spend its budget rediscovering a space it could simply
enumerate, while adding sampling noise and no coverage guarantee.

Enumeration is also what makes staging safe. The parameters factor into an
expensive intraday stage and a cheap daily stage:

    ModerateRisk   768 = 96 intraday x 8 daily
    WaitRescue     324 = 81 intraday x 4 daily

Because the whole space is known up front, every distinct intraday result is
computed once and reused, which is where the ~4x speedup comes from. Genome
ordering in a GP is random, so the same reuse would need an unbounded cache.
"""

from dataclasses import asdict, dataclass, replace
from itertools import product
import time
from pathlib import Path
import json
import os

import torch

from min_gp.evaluation import IncrementalFitness, WalkForwardConfig
from min_gp.factors.event_factors import ModerateRiskTemplate, WaitRescueTemplate
from min_gp.gp.dripping_stone import _pareto_rank_and_crowding
from min_gp.gp.event import EVENT_FAMILIES, EventFactorEvaluator
from min_gp.operators.event import (
    DDOF_VALUES,
    EDGE_EXCLUSIONS,
    EVENT_SIGMAS,
    EXCLUDE_OPEN_MINUTES,
    FORWARD_WINDOWS,
    MIN_EVENT_GAPS,
    TOP_K_EVENTS,
)
from min_gp.operators.temporal import SMOOTH_WINDOWS
from min_gp.experiment import (
    append_failure, atomic_json, configuration_fingerprint,
)
from min_gp.gp.cost_control import (
    DEFAULT_MAX_COST_UNITS, DEFAULT_MAX_ESTIMATED_SECONDS,
    DEFAULT_MAX_UNCALIBRATED_COST_UNITS,
)


DAILY_SMOOTH_WINDOWS = SMOOTH_WINDOWS[1:]


@dataclass(frozen=True)
class ExhaustiveConfig:
    family: str = "moderate_risk"
    chunk_rows: int = 4096
    # Intraday results are (I, D) per parameter set and there can be ~100 of
    # them at once, so they are parked off the accelerator by default; the
    # daily stage moves them back one at a time.
    store_device: str = "cpu"
    verbose: bool = True
    progress_every: int = 50
    checkpoint_path: str | None = None
    error_log_path: str | None = None
    resume: bool = False
    max_cost_units: int | None = DEFAULT_MAX_COST_UNITS
    max_estimated_seconds: float | None = DEFAULT_MAX_ESTIMATED_SECONDS
    max_peak_bytes: int | None = None
    max_uncalibrated_cost_units: int | None = DEFAULT_MAX_UNCALIBRATED_COST_UNITS
    allow_uncalibrated_cost: bool = False

    def __post_init__(self):
        if self.family not in EVENT_FAMILIES:
            raise ValueError(f"unknown event family: {self.family}")


def enumerate_event_templates(family: str):
    """Every valid genome in a family's parameter space, in a stable order."""
    if family == "moderate_risk":
        return [
            ModerateRiskTemplate(
                sigma=sigma, response_window=response, exclude_edges=edges,
                ddof=ddof, standardize_before_distance=standardize,
                smooth_window=smooth,
            )
            for sigma, edges, ddof, response, standardize, smooth in product(
                EVENT_SIGMAS, EDGE_EXCLUSIONS, DDOF_VALUES, FORWARD_WINDOWS,
                (False, True), DAILY_SMOOTH_WINDOWS,
            )
        ]
    if family == "wait_rescue":
        return [
            WaitRescueTemplate(
                top_k=top_k, exclude_before=before, min_gap=gap,
                follow_window=follow, smooth_window=smooth,
            )
            for top_k, before, gap, follow, smooth in product(
                TOP_K_EVENTS, EXCLUDE_OPEN_MINUTES, MIN_EVENT_GAPS,
                FORWARD_WINDOWS, DAILY_SMOOTH_WINDOWS,
            )
        ]
    raise ValueError(f"unknown event family: {family}")


def _intraday_key(genome):
    """The parameters that drive the expensive minute-level stage."""
    if isinstance(genome, ModerateRiskTemplate):
        return (
            genome.sigma, genome.exclude_edges, genome.ddof,
            genome.response_window,
        )
    return (
        genome.top_k, genome.exclude_before, genome.min_gap,
        genome.follow_window,
    )


class ExhaustiveEventSearch(EventFactorEvaluator):
    """Enumerate a family, sharing every distinct intraday computation."""

    def __init__(
        self,
        minute_tensors: dict[str, torch.Tensor],
        fwd_ret: torch.Tensor,
        pool_mask: torch.Tensor | None = None,
        config: ExhaustiveConfig | None = None,
        fitness_config: WalkForwardConfig | None = None,
    ):
        super().__init__(
            minute_tensors, fwd_ret, pool_mask,
            config or ExhaustiveConfig(), fitness_config,
        )
        self.device = self.minute_tensors["volume"].device

    # ── stage 1: minute level, one result per distinct intraday key ──

    def _moderate_risk_stage1(self, keys):
        from min_gp.operators.event import (
            forward_window_std, intraday_sigma_event, masked_daily_mean,
            minute_delta_volume, minute_return,
        )
        close = self.minute_tensors["close"]
        volume = self.minute_tensors["volume"]
        I, D, M = close.shape
        rows = I * D
        close_rows, volume_rows = close.reshape(rows, M), volume.reshape(rows, M)
        store = self.config.store_device
        raw_vol = {
            key: torch.full((rows,), float("nan"), device=store) for key in keys
        }
        # The spike-minute return mean ignores the response window, so it is
        # keyed one dimension shorter and shared across response windows.
        ret_keys = {key[:3] for key in keys}
        raw_ret = {
            key: torch.full((rows,), float("nan"), device=store)
            for key in ret_keys
        }
        for start in range(0, rows, self.config.chunk_rows):
            stop = min(start + self.config.chunk_rows, rows)
            ret = minute_return(close_rows[start:stop], horizon=1)
            delta = minute_delta_volume(volume_rows[start:stop], horizon=1)
            responses = {}
            spikes = {}
            for sigma, edges, ddof, window in sorted(keys):
                spike_key = (sigma, edges, ddof)
                if spike_key not in spikes:
                    spikes[spike_key] = intraday_sigma_event(
                        delta, sigma=sigma, direction="above",
                        exclude_edges=edges, ddof=ddof,
                    )
                    raw_ret[spike_key][start:stop] = masked_daily_mean(
                        ret, spikes[spike_key]
                    ).to(store)
                if (window, ddof) not in responses:
                    responses[(window, ddof)] = forward_window_std(
                        ret, window=window, ddof=ddof
                    )
                raw_vol[(sigma, edges, ddof, window)][start:stop] = (
                    masked_daily_mean(
                        responses[(window, ddof)], spikes[spike_key]
                    ).to(store)
                )
            del ret, delta, responses, spikes
        return {
            key: (raw_vol[key].reshape(I, D), raw_ret[key[:3]].reshape(I, D))
            for key in keys
        }

    def _wait_rescue_stage1(self, keys):
        from min_gp.operators.event import (
            follow_ratio_series, masked_daily_mean, topk_separated_events,
        )
        volume = self.minute_tensors["volume"]
        I, D, M = volume.shape
        rows = I * D
        volume_rows = volume.reshape(rows, M)
        store = self.config.store_device
        raw = {key: torch.full((rows,), float("nan"), device=store) for key in keys}
        for start in range(0, rows, self.config.chunk_rows):
            stop = min(start + self.config.chunk_rows, rows)
            chunk = volume_rows[start:stop]
            events, series = {}, {}
            for top_k, before, gap, follow in sorted(keys):
                event_key = (top_k, before, gap)
                if event_key not in events:
                    events[event_key] = topk_separated_events(
                        chunk, k=top_k, exclude_before=before, min_gap=gap
                    )
                if follow not in series:
                    series[follow] = follow_ratio_series(chunk, window=follow)
                raw[(top_k, before, gap, follow)][start:stop] = masked_daily_mean(
                    series[follow], events[event_key]
                ).to(store)
            del events, series
        return {key: raw[key].reshape(I, D) for key in keys}

    # ── stage 2: daily level, cheap, per genome ──

    def _moderate_risk_factor(self, genome, stage1):
        from min_gp.operators.cross_section import cross_section_distance
        from min_gp.operators.temporal import equal_blend, mean_std_blend
        raw_vol, raw_ret = stage1[_intraday_key(genome)]
        kwargs = {"standardize": genome.standardize_before_distance}
        bright_vol = mean_std_blend(
            cross_section_distance(raw_vol.to(self.device), **kwargs),
            genome.smooth_window,
        )
        bright_ret = mean_std_blend(
            cross_section_distance(raw_ret.to(self.device), **kwargs),
            genome.smooth_window,
        )
        return equal_blend(bright_vol, bright_ret)

    def _wait_rescue_factor(self, genome, stage1):
        from min_gp.operators.temporal import mean_std_blend
        raw = stage1[_intraday_key(genome)].to(self.device)
        return mean_std_blend(raw, genome.smooth_window)

    def build_factor(self, genome, stage1):
        """Rebuild a genome's factor from shared intraday results."""
        if isinstance(genome, ModerateRiskTemplate):
            factor = self._moderate_risk_factor(genome, stage1)
        else:
            factor = self._wait_rescue_factor(genome, stage1)
        return self._mask(factor)

    def run(self):
        genomes = [
            genome for genome in enumerate_event_templates(self.config.family)
            if self._admissible(genome)
        ]
        if not genomes:
            raise ValueError("no exhaustive genome fits the configured cost budget")
        space_fingerprint = configuration_fingerprint({
            "family": self.config.family,
            "ordered_genomes": [genome.to_dict() for genome in genomes],
            "fitness": asdict(self.fitness_config),
        })
        keys = {_intraday_key(genome) for genome in genomes}
        started = time.time()
        if self.config.verbose:
            print(
                f"[exhaustive {self.config.family}] {len(genomes)} genomes = "
                f"{len(keys)} intraday x {len(genomes) // max(len(keys), 1)} daily",
                flush=True,
            )
        checkpoint = Path(self.config.checkpoint_path) if self.config.checkpoint_path else None
        progress_path = checkpoint.with_suffix(".progress.json") if checkpoint else None
        if self.config.resume and checkpoint and checkpoint.exists():
            payload = torch.load(checkpoint, map_location=self.config.store_device, weights_only=False)
            if payload["family"] != self.config.family:
                raise ValueError("exhaustive checkpoint family mismatch")
            if payload.get("space_fingerprint") != space_fingerprint:
                raise ValueError(
                    "exhaustive checkpoint search-space mismatch; "
                    "start a new run instead of resuming"
                )
            stage1 = payload["stage1"]
        else:
            if self.config.family == "moderate_risk":
                stage1 = self._moderate_risk_stage1(keys)
            else:
                stage1 = self._wait_rescue_stage1(keys)
            if checkpoint:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
                torch.save({
                    "family": self.config.family,
                    "space_fingerprint": space_fingerprint,
                    "stage1": stage1,
                }, temporary)
                os.replace(temporary, checkpoint)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self.config.verbose:
            print(
                f"[exhaustive {self.config.family}] intraday stage done "
                f"({time.time() - started:.0f}s)", flush=True,
            )

        scored = []
        if self.config.resume and progress_path and progress_path.exists():
            with open(progress_path, encoding="utf-8") as handle:
                saved = json.load(handle)
            if saved.get("family") != self.config.family:
                raise ValueError("exhaustive progress family mismatch")
            if saved.get("space_fingerprint") != space_fingerprint:
                raise ValueError(
                    "exhaustive progress search-space mismatch; "
                    "start a new run instead of resuming"
                )
            cls = ModerateRiskTemplate if self.config.family == "moderate_risk" else WaitRescueTemplate
            scored = [
                (IncrementalFitness(**row["score"]), cls(**row["genome"]))
                for row in saved.get("scored", [])
            ]
            if len(scored) > len(genomes):
                raise ValueError("exhaustive progress contains too many genomes")
            if [genome for _, genome in scored] != genomes[:len(scored)]:
                raise ValueError(
                    "exhaustive progress genome prefix does not match the "
                    "current enumeration order"
                )
        for index, genome in enumerate(genomes[len(scored):], start=len(scored)):
            try:
                factor = self.build_factor(genome, stage1)
            except (RuntimeError, ValueError, TypeError) as exc:
                append_failure(self.config.error_log_path, "exhaustive_factor", genome, exc)
                scored.append((IncrementalFitness.invalid(), genome))
            else:
                score = self.score(factor, genome)
                del factor
                scored.append((score, genome))
            if self.config.verbose and (index + 1) % self.config.progress_every == 0:
                best = max(
                    (s for s, _ in scored if s.valid),
                    key=lambda s: s.robust_ic, default=None,
                )
                best_text = "none yet" if best is None else f"{best.robust_ic:+.4f}"
                print(
                    f"[exhaustive {self.config.family}] {index + 1}/{len(genomes)} "
                    f"valid={sum(s.valid for s, _ in scored)} bestIC={best_text} "
                    f"({time.time() - started:.0f}s)", flush=True,
                )
            if progress_path and (
                (index + 1) % self.config.progress_every == 0
                or index + 1 == len(genomes)
            ):
                atomic_json(progress_path, {
                    "family": self.config.family,
                    "space_fingerprint": space_fingerprint,
                    "scored": [
                        {"score": score.__dict__, "genome": item.to_dict()}
                        for score, item in scored
                    ],
                })
        del stage1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ranks, crowd = _pareto_rank_and_crowding(scored)
        order = sorted(range(len(scored)), key=lambda i: (ranks[i], -crowd[i]))
        if self.config.verbose:
            print(
                f"[exhaustive {self.config.family}] done: "
                f"{sum(s.valid for s, _ in scored)}/{len(scored)} valid, "
                f"front0={sum(r == 0 for r in ranks)} "
                f"({time.time() - started:.0f}s)", flush=True,
            )
        return [
            (scored[i][0], scored[i][1], ranks[i], crowd[i]) for i in order
        ]
