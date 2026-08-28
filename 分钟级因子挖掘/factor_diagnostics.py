"""Weekly IC and portfolio diagnostics for exported daily factor leaves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from min_gp.config import (
    ADJUSTED_CLOSE_PARQUET, INDUSTRY_VALUE_EXPOSURES_PARQUET,
    ZZ500_PIT_PARQUET, require_path,
)
from min_gp.data import load_pit_codes, load_pit_daily_mask, load_pit_dates
from min_gp.evaluation import (
    BatchedNeutralizer, DEFAULT_COST_BPS, trailing_signal_mean,
)
from min_gp.label import tensor_weekly_fwd_ret
from min_gp.numeric.ranking import cross_section_rank
from min_gp.numeric.preprocessing import remove_outliers
from min_gp.spectral_data import (
    load_daily_close_tensor, load_daily_exposures, load_daily_factor_leaves,
)


def parse_factor(value):
    """Parse NAME=PATH:DIRECTION while preserving a Windows drive colon."""
    name, source = value.split("=", 1)
    path, direction = source.rsplit(":", 1)
    direction = int(direction)
    if direction not in (-1, 1):
        raise ValueError("factor direction must be -1 or +1")
    return name, path, direction


def prepare_signal(factor, average_days, direction, neutralizer=None):
    """Build the actual ranking signal in its declared execution order."""
    signal = trailing_signal_mean(factor, average_days)
    if neutralizer is not None:
        signal = neutralizer(signal)
    return signal * direction


def _daily_pearson_prepared(factor, returns, min_cross_section=30):
    valid = torch.isfinite(factor) & torch.isfinite(returns)
    count = valid.sum(0)
    x = torch.nan_to_num(factor.float())
    y = torch.nan_to_num(returns.float())
    n = count.clamp(min=1).float()
    mx = (x * valid).sum(0) / n
    my = (y * valid).sum(0) / n
    dx, dy = x - mx.unsqueeze(0), y - my.unsqueeze(0)
    covariance = (dx * dy * valid).sum(0) / n
    sx = torch.sqrt((dx.square() * valid).sum(0) / n)
    sy = torch.sqrt((dy.square() * valid).sum(0) / n)
    corr = covariance / (sx * sy).clamp(min=1e-12)
    return torch.where(
        count >= min_cross_section, corr,
        torch.full_like(corr, float("nan")),
    )


def daily_pearson_ic(factor, returns, min_cross_section=30):
    factor = remove_outliers(factor, n_mad=5.0, dim=0)
    return _daily_pearson_prepared(factor, returns, min_cross_section)


def daily_rank_ic(factor, returns, min_cross_section=30):
    factor = remove_outliers(factor, n_mad=5.0, dim=0)
    return _daily_pearson_prepared(
        cross_section_rank(factor), cross_section_rank(returns),
        min_cross_section,
    )


def summarize(series):
    values = series[np.isfinite(series)]
    if not len(values):
        return {"mean": float("nan"), "std": float("nan"),
                "ir": float("nan"), "count": 0}
    std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    mean = float(values.mean())
    return {
        "mean": mean, "std": std,
        "ir": mean / std if std > 0 else float("nan"),
        "count": int(len(values)),
    }


def max_drawdown(nav):
    if not len(nav):
        return float("nan")
    peak = np.maximum.accumulate(nav)
    return float(np.max((peak - nav) / np.maximum(peak, 1e-12)))


def weekly_portfolios(
    signal, returns, dates, groups=5, cost_bps=DEFAULT_COST_BPS,
    min_cross_section=30,
):
    """Build equal-weight weekly quantiles and gross/net long-short series."""
    if groups < 2:
        raise ValueError("groups must be >= 2")
    instruments, days = signal.shape
    previous = torch.zeros(instruments, dtype=torch.float32)
    used_dates, day_indices = [], []
    group_returns, gross_ls, net_ls, turnovers = [], [], [], []
    for day in range(days):
        score, future = signal[:, day].float(), returns[:, day].float()
        valid = torch.isfinite(score) & torch.isfinite(future)
        indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
        if indices.numel() < max(min_cross_section, groups):
            continue
        ordered = indices[torch.argsort(score[indices])]
        partitions = torch.tensor_split(ordered, groups)
        values = [float(future[index].mean().item()) for index in partitions]
        weights = torch.zeros_like(previous)
        weights[partitions[-1]] = 1.0 / len(partitions[-1])
        weights[partitions[0]] = -1.0 / len(partitions[0])
        gross = float((weights * torch.nan_to_num(future)).sum().item())
        turnover = float((0.5 * (weights - previous).abs().sum()).item())
        net = gross - turnover * cost_bps * 1e-4
        used_dates.append(str(dates[day])[:10])
        day_indices.append(day)
        group_returns.append(values)
        gross_ls.append(gross)
        net_ls.append(net)
        turnovers.append(turnover)
        previous = weights
    return {
        "dates": np.asarray(used_dates),
        "day_indices": np.asarray(day_indices, dtype=np.int64),
        "groups": np.asarray(group_returns, dtype=np.float64),
        "gross_ls": np.asarray(gross_ls, dtype=np.float64),
        "net_ls": np.asarray(net_ls, dtype=np.float64),
        "turnover": np.asarray(turnovers, dtype=np.float64),
    }


def diagnose(signal, returns, dates, cost_bps=DEFAULT_COST_BPS,
             min_cross_section=30, groups=5):
    ic = daily_pearson_ic(signal, returns, min_cross_section).cpu().numpy()
    rank_ic = daily_rank_ic(signal, returns, min_cross_section).cpu().numpy()
    portfolios = weekly_portfolios(
        signal.cpu(), returns.cpu(), dates, groups, cost_bps,
        min_cross_section,
    )
    weekly_ic = ic[portfolios["day_indices"]]
    weekly_rank_ic = rank_ic[portfolios["day_indices"]]
    group_nav = np.cumprod(1.0 + portfolios["groups"], axis=0)
    gross_nav = np.cumprod(1.0 + portfolios["gross_ls"])
    net_nav = np.cumprod(1.0 + portfolios["net_ls"])
    ic_summary, rank_summary = summarize(weekly_ic), summarize(weekly_rank_ic)
    metrics = {
        "ic": ic_summary["mean"], "icir": ic_summary["ir"],
        "rank_ic": rank_summary["mean"], "rank_icir": rank_summary["ir"],
        "weeks": rank_summary["count"],
        "mean_weekly_gross_ls": float(np.mean(portfolios["gross_ls"])),
        "mean_weekly_net_ls": float(np.mean(portfolios["net_ls"])),
        "mean_turnover": float(np.mean(portfolios["turnover"])),
        "gross_terminal_nav": float(gross_nav[-1]),
        "net_terminal_nav": float(net_nav[-1]),
        "net_max_drawdown": max_drawdown(net_nav),
    }
    return metrics, {
        **portfolios, "ic": weekly_ic, "rank_ic": weekly_rank_ic,
        "group_nav": group_nav, "gross_nav": gross_nav, "net_nav": net_nav,
    }


def plot_diagnostics(results, destination):
    figure, axes = plt.subplots(
        2, len(results), figsize=(8 * len(results), 9), squeeze=False,
    )
    for column, (name, metrics, series) in enumerate(results):
        dates = pd.to_datetime(series["dates"])
        top, bottom = axes[0, column], axes[1, column]
        for group in range(series["group_nav"].shape[1]):
            top.plot(dates, series["group_nav"][:, group],
                     label=f"Q{group + 1}", linewidth=1.4)
        top.axhline(1.0, color="black", linewidth=.7)
        top.set_title(
            f"{name}: weekly quintile NAV\n"
            f"IC={metrics['ic']:.4f} ICIR={metrics['icir']:.3f}  "
            f"RankIC={metrics['rank_ic']:.4f} RankICIR={metrics['rank_icir']:.3f}"
        )
        top.set_ylabel("Gross NAV")
        top.grid(alpha=.25)
        top.legend(ncol=5, fontsize=8)
        bottom.plot(dates, series["gross_nav"], label="Long-short gross", linewidth=1.5)
        bottom.plot(dates, series["net_nav"], label="Long-short net (30bp)", linewidth=2)
        bottom.axhline(1.0, color="black", linewidth=.7)
        bottom.set_title(
            f"Q5 - Q1 | mean net/week={metrics['mean_weekly_net_ls']:.3%}, "
            f"turnover={metrics['mean_turnover']:.2f}, MDD={metrics['net_max_drawdown']:.1%}"
        )
        bottom.set_ylabel("NAV")
        bottom.grid(alpha=.25)
        bottom.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", action="append", required=True,
                        metavar="NAME=PATH:DIRECTION")
    parser.add_argument("--start", default="2018-01-02")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--signal-average-days", type=int, default=5)
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument("--min-cross-section", type=int, default=30)
    parser.add_argument("--daily-parquet", default=str(ADJUSTED_CLOSE_PARQUET))
    parser.add_argument("--pit", default=str(ZZ500_PIT_PARQUET))
    parser.add_argument("--neutralize", action="store_true",
                        help="residualize the averaged signal on PIT industry and log float cap")
    parser.add_argument("--exposures", default=str(INDUSTRY_VALUE_EXPOSURES_PARQUET))
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    pit = require_path(args.pit, "PIT universe")
    daily = require_path(args.daily_parquet, "adjusted daily close")
    instruments = load_pit_codes(pit, args.start, args.end)
    dates = load_pit_dates(pit, args.start, args.end)
    close = load_daily_close_tensor(daily, dates, instruments, device="cpu")
    returns = tensor_weekly_fwd_ret(close, dates)
    pool = load_pit_daily_mask(pit, dates, instruments, device="cpu")
    pool &= torch.isfinite(close)
    returns = torch.where(pool, returns, torch.full_like(returns, float("nan")))

    neutralizer, exposure_names = None, []
    if args.neutralize:
        styles, industry, _levels = load_daily_exposures(
            require_path(args.exposures, "industry/market-cap exposures"),
            dates, instruments, device="cpu",
        )
        exposure_names = [*styles, "sw_level1"]
        neutralizer = BatchedNeutralizer(
            close.shape, tuple(styles.values()), industry,
            args.min_cross_section,
        )

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    results, all_metrics = [], {}
    for name, path, direction in map(parse_factor, args.factor):
        factor = load_daily_factor_leaves(
            require_path(path, f"factor {name}"), dates, instruments,
            ["factor"], device="cpu",
        )["factor"]
        signal = prepare_signal(
            factor, args.signal_average_days, direction, neutralizer
        )
        signal = torch.where(pool, signal, torch.full_like(signal, float("nan")))
        metrics, series = diagnose(
            signal, returns, dates, args.cost_bps,
            args.min_cross_section, args.groups,
        )
        metrics.update({
            "name": name, "direction": direction,
            "signal_average_days": args.signal_average_days,
            "cost_bps": args.cost_bps,
            "neutralized": args.neutralize,
            "neutralization_exposures": exposure_names,
        })
        all_metrics[name] = metrics
        results.append((name, metrics, series))
        weekly = pd.DataFrame({
            "trade_date": series["dates"],
            "ic": series["ic"],
            "rank_ic": series["rank_ic"],
            **{f"q{i + 1}_return": series["groups"][:, i]
               for i in range(args.groups)},
            "gross_long_short": series["gross_ls"],
            "net_long_short": series["net_ls"],
            "turnover": series["turnover"],
            **{f"q{i + 1}_nav": series["group_nav"][:, i]
               for i in range(args.groups)},
            "gross_long_short_nav": series["gross_nav"],
            "net_long_short_nav": series["net_nav"],
        })
        weekly.to_csv(output / f"{name}_weekly.csv", index=False)
    (output / "metrics.json").write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_diagnostics(results, output / "factor_diagnostics.png")
    print(json.dumps(all_metrics, ensure_ascii=False, indent=2))
    print(f"[diagnostics] -> {output}")


if __name__ == "__main__":
    main()
