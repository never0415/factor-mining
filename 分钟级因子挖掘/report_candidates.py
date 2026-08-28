"""Full IC / ICIR / RankIC / RankICIR and quantile-curve report for candidates.

The GP prints one line per generation for the highest-IC genome only. This
walks the whole Pareto front on both the search window and the untouched
hold-out, so the trade-off between rank quality and tradability is visible per
candidate instead of inferred from a single objective.

  python -m min_gp.report_candidates \
      --candidates .../event_skeleton_gp.jsonl --out report.json
"""

import argparse
import json

import numpy as np
import torch

from min_gp.config import (
    ADJUSTED_CLOSE_PARQUET, MINUTE_PARQUET, ZZ500_PIT_PARQUET, require_path,
)
from min_gp.data import load_pit_codes, load_pit_daily_mask, load_pit_dates
from min_gp.evaluation.incremental import trailing_signal_mean
from min_gp.factors.event_skeleton import EventSkeletonGenome
from min_gp.label import tensor_rebalance_fwd_ret
from min_gp.numeric.ranking import cross_section_rank
from min_gp.numeric.preprocessing import remove_outliers
from min_gp.spectral_data import build_minute_slice, load_daily_close_tensor


def _pearson(a, b, valid, preprocess=True):
    """Column-wise correlation over the valid mask; NaN where too few names."""
    if preprocess:
        a = remove_outliers(a, n_mad=5.0, dim=0)
        valid = valid & torch.isfinite(a)
    count = valid.sum(0)
    n = count.clamp(min=1).to(a.dtype)
    av, bv = torch.nan_to_num(a) * valid, torch.nan_to_num(b) * valid
    ma, mb = av.sum(0) / n, bv.sum(0) / n
    da, db = (av - ma) * valid, (bv - mb) * valid
    cov = (da * db).sum(0) / n
    sd = torch.sqrt((da * da).sum(0) / n) * torch.sqrt((db * db).sum(0) / n)
    out = cov / sd.clamp(min=1e-12)
    return torch.where(count >= 30, out, torch.full_like(out, float("nan")))


def series_stats(series, periods_per_year):
    finite = series[np.isfinite(series)]
    if finite.size < 2:
        return dict(mean=float("nan"), ir=float("nan"), ir_annual=float("nan"),
                    positive=float("nan"), n=int(finite.size))
    mean, sd = float(finite.mean()), float(finite.std())
    ir = mean / sd if sd > 0 else float("nan")
    return dict(
        mean=mean, ir=ir, ir_annual=ir * float(np.sqrt(periods_per_year)),
        positive=float((finite > 0).mean()), n=int(finite.size),
    )


def quantile_curves(factor, fwd, direction, groups, cost_bps):
    """Equal-weight group returns per rebalance date, plus a net long-short.

    The fifth value is the date indices the series actually cover. Dates whose
    cross-section falls below 30 names are skipped, so this is a subset of the
    rebalance dates -- and it differs per factor, which is what makes two
    candidates' curves impossible to correlate without it.
    """
    ranks = cross_section_rank(factor * direction)
    dates = torch.nonzero(torch.isfinite(fwd).any(0), as_tuple=False).squeeze(1)
    per_group = [[] for _ in range(groups)]
    long_short, net_long_short, turnovers, kept = [], [], [], []
    previous = None
    for day in dates.tolist():
        r, y = ranks[:, day], fwd[:, day]
        valid = torch.isfinite(r) & torch.isfinite(y)
        if int(valid.sum()) < 30:
            continue
        index = torch.nonzero(valid, as_tuple=False).squeeze(1)
        order = torch.argsort(r[index])
        chunks = torch.chunk(order, groups)
        means = [float(y[index[chunk]].mean()) for chunk in chunks]
        for slot, value in enumerate(means):
            per_group[slot].append(value)
        spread = means[-1] - means[0]
        weights = torch.zeros_like(ranks[:, day])
        weights[index[chunks[-1]]] = 1.0 / len(chunks[-1])
        weights[index[chunks[0]]] = -1.0 / len(chunks[0])
        turnover = (
            float(weights.abs().sum()) if previous is None
            else float((weights - previous).abs().sum())
        )
        previous = weights
        kept.append(day)
        long_short.append(spread)
        net_long_short.append(spread - 0.5 * turnover * cost_bps * 1e-4)
        turnovers.append(turnover)
    return (
        [np.asarray(values) for values in per_group],
        np.asarray(long_short), np.asarray(net_long_short),
        np.asarray(turnovers), kept,
    )


def evaluate_window(genomes, directions, minute, close, pool_mask, dates,
                    rule, period, average_days, groups, cost_bps, chunk_rows):
    fwd = tensor_rebalance_fwd_ret(close, dates, rule, period)
    fwd = torch.where(pool_mask, fwd, torch.full_like(fwd, float("nan")))
    label_dates = [
        dates[i] for i in
        torch.nonzero(torch.isfinite(fwd).any(0), as_tuple=False).squeeze(1).tolist()
    ]
    periods_per_year = 52 if rule == "week_end" else 244
    rank_y = cross_section_rank(fwd)
    results = []
    for index, genome in enumerate(genomes):
        direction = directions[index]
        factor = genome.evaluate(minute, chunk_rows)
        factor = trailing_signal_mean(factor, average_days)
        factor = torch.where(pool_mask, factor, torch.full_like(factor, float("nan")))
        factor = remove_outliers(factor, n_mad=5.0, dim=0)
        valid = torch.isfinite(factor) & torch.isfinite(fwd)
        ic = _pearson(factor, fwd, valid, preprocess=False) * direction
        rank_ic = _pearson(
            cross_section_rank(factor), rank_y, valid, preprocess=False
        ) * direction
        groups_series, gross, net, turnover, days = quantile_curves(
            factor, fwd, direction, groups, cost_bps
        )
        results.append(dict(
            index=index,
            direction=direction,
            # Dates the curves actually cover. Weeks whose cross-section falls
            # below 30 names are skipped, and which weeks those are differs per
            # factor -- without this, two candidates' curves cannot be aligned.
            days=days,
            ic=series_stats(ic.cpu().numpy(), periods_per_year),
            rank_ic=series_stats(rank_ic.cpu().numpy(), periods_per_year),
            coverage=float(valid.sum().div(torch.isfinite(fwd).sum().clamp(min=1))),
            turnover=float(np.nanmean(turnover)) if turnover.size else float("nan"),
            group_cumulative=[
                np.cumprod(1.0 + values).tolist() for values in groups_series
            ],
            long_short_cumulative=np.cumprod(1.0 + gross).tolist(),
            net_long_short_cumulative=np.cumprod(1.0 + net).tolist(),
            long_short_mean=float(gross.mean()) if gross.size else float("nan"),
            net_long_short_mean=float(net.mean()) if net.size else float("nan"),
        ))
        del factor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # ``days`` on each candidate indexes the full trading-day axis, not
    # ``label_dates`` -- ship that axis so the two can be joined without
    # reconstructing the rebalance calendar outside.
    return dict(dates=label_dates, all_dates=list(dates), candidates=results)


def load_front(path, max_rank):
    records, seen = [], set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if not record["fitness"]["valid"]:
                continue
            if record.get("pareto_rank", 0) > max_rank:
                continue
            key = record["expression"]
            if key in seen:
                continue
            seen.add(key)
            records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--max-rank", type=int, default=0)
    # The walk-forward folds only score up to 2022-03-07 (four 126-day
    # validation windows after a 504-day minimum train). Everything between
    # that date and the hold-out was loaded but never entered any objective,
    # so it is a genuinely untouched validation block -- report it separately
    # rather than folding it into "train", where it would look in-sample.
    parser.add_argument("--select-start", default="2018-01-02")
    parser.add_argument("--select-end", default="2022-03-07")
    parser.add_argument("--valid-start", default="2022-03-08")
    parser.add_argument("--valid-end", default="2024-12-31")
    parser.add_argument("--holdout-start", default="2025-01-02")
    parser.add_argument("--holdout-end", default="2026-07-31")
    parser.add_argument("--rebalance", default="week_end")
    parser.add_argument("--period", type=int, default=1)
    parser.add_argument("--signal-average-days", type=int, default=5)
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--cost-bps", type=float, default=30.0)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--max-stocks", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = "cpu" if args.cpu else "cuda"
    minute_path = require_path(str(MINUTE_PARQUET), "minute parquet")
    daily_path = require_path(str(ADJUSTED_CLOSE_PARQUET), "adjusted close")
    pit_path = require_path(str(ZZ500_PIT_PARQUET), "PIT parquet")

    records = load_front(args.candidates, args.max_rank)
    genomes = [EventSkeletonGenome.from_dict(r["genome"]) for r in records]
    directions = [int(r["fitness"].get("direction", 1)) for r in records]
    fields = tuple(sorted({f for g in genomes for f in g.required_fields}))
    print(f"[report] {len(genomes)} candidates, fields={fields}", flush=True)

    report = {"expressions": [r["expression"] for r in records],
              "train_fitness": [r["fitness"] for r in records], "windows": {}}
    for name, start, end in (
        ("select", args.select_start, args.select_end),
        ("valid", args.valid_start, args.valid_end),
        ("holdout", args.holdout_start, args.holdout_end),
    ):
        instruments = load_pit_codes(pit_path, start, end)
        if args.max_stocks:
            instruments = instruments[:args.max_stocks]
        dates = load_pit_dates(pit_path, start, end)
        minute, meta = build_minute_slice(
            minute_path, start, end, fields=fields,
            instruments=instruments, dates=dates, device=device,
        )
        close = load_daily_close_tensor(
            daily_path, meta["dates"], meta["instruments"], device=device
        )
        pool = load_pit_daily_mask(
            pit_path, meta["dates"], meta["instruments"], device=device
        ) & torch.isfinite(close)
        print(f"[report] {name}: {meta['I']}x{meta['D']}", flush=True)
        report["windows"][name] = evaluate_window(
            genomes, directions, minute, close, pool, meta["dates"],
            args.rebalance, args.period, args.signal_average_days,
            args.groups, args.cost_bps, args.chunk_rows,
        )
        del minute, close, pool
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False)
    print(f"[report] written -> {args.out}")


if __name__ == "__main__":
    main()
