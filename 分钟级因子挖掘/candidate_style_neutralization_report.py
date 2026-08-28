"""Individually neutralise one handbook GP factor against ten style factors."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from min_gp.climb_candidate_neutralization import exposure_corr
from min_gp.config import (
    ADJUSTED_CLOSE_PARQUET,
    MINUTE_PARQUET,
    RISK_EXPOSURES_PARQUET,
    ZZ500_PIT_PARQUET,
)
from min_gp.data import load_pit_codes, load_pit_daily_mask, load_pit_dates
from min_gp.evaluation.incremental import trailing_signal_mean
from min_gp.evaluation.neutralize import BatchedNeutralizer, trailing_volatility
from min_gp.factors.handbook_skeleton import HandbookSkeletonGenome
from min_gp.label import tensor_rebalance_fwd_ret
from min_gp.long_short_battle_html_report import WINDOWS, evaluate_period
from min_gp.numeric.ranking import cross_section_rank
from min_gp.spectral_data import build_minute_slice, load_daily_close_tensor, load_daily_exposures


VALUE_EXPOSURES = r"F:\fangzheng research\data\interim\zz500_industry_value_exposures.parquet"
FUNDAMENTAL_EXPOSURES = r"F:\fangzheng research\data\interim\zz500_fundamental_style_exposures.parquet"

STYLE_META = {
    "size": ("市值", "流通市值对数 ln(float market cap)"),
    "nonlinear_size": (
        "非线性市值",
        "逐日标准化市值的三次项对标准化市值回归后的残差",
    ),
    "bp": ("价值/BP", "账面价值与总市值之比 book-to-price"),
    "growth": (
        "成长",
        "净资产、营业总收入、归母净利润同比增速的逐日截面秩均值（至少两项有效）",
    ),
    "momentum": ("3个月动量", "截至信号日的过去60交易日复权收盘累计收益"),
    "earnings_yield": ("盈利收益", "AkShare 每日 PE(TTM) 的倒数"),
    "leverage": ("杠杆", "AkShare 财报总负债/总资产，公告日后一个交易日生效"),
    "liquidity": ("流动性", "项目PIT风险暴露 ln_amount"),
    "beta": ("市场Beta", "项目PIT市场Beta暴露"),
    "volatility": ("60日波动率", "截至信号日的过去60交易日日收益标准差"),
}


def _fmt(value, pct: bool = False, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{digits}%}" if pct else f"{value:.{digits}f}"


def _trailing_return(close: torch.Tensor, window: int) -> torch.Tensor:
    output = torch.full_like(close, float("nan"))
    previous = close[:, :-window]
    current = close[:, window:]
    valid = torch.isfinite(previous) & torch.isfinite(current) & previous.ne(0)
    values = current / previous - 1.0
    output[:, window:] = torch.where(valid, values, torch.full_like(values, float("nan")))
    return output


def _nonlinear_size(size: torch.Tensor) -> torch.Tensor:
    valid = torch.isfinite(size)
    safe = torch.where(valid, size, torch.zeros_like(size))
    count = valid.sum(dim=0).clamp(min=1)
    mean = safe.sum(dim=0) / count
    centered = torch.where(valid, size - mean, torch.zeros_like(size))
    sigma = torch.sqrt(centered.square().sum(dim=0) / count).clamp(min=1e-8)
    zscore = torch.where(valid, (size - mean) / sigma, torch.full_like(size, float("nan")))
    zscore = zscore.clamp(min=-3.0, max=3.0)
    cubic = zscore.pow(3)
    projector = BatchedNeutralizer(
        size.shape, continuous=(zscore,), rank_space=False, min_cross_section=30
    )
    return projector(cubic)


def _growth_composite(*components: torch.Tensor) -> torch.Tensor:
    ranked = torch.stack([cross_section_rank(value.float()) for value in components])
    valid = torch.isfinite(ranked)
    count = valid.sum(dim=0)
    total = torch.where(valid, ranked, torch.zeros_like(ranked)).sum(dim=0)
    result = total / count.clamp(min=1)
    return torch.where(count >= 2, result, torch.full_like(result, float("nan")))


def _coverage(exposure: torch.Tensor, pool: torch.Tensor, date_mask: torch.Tensor) -> float:
    eligible = pool & date_mask.unsqueeze(0)
    denominator = eligible.sum()
    if not denominator:
        return float("nan")
    return float((eligible & torch.isfinite(exposure)).sum() / denominator)


def _write_html(report: dict, destination: Path) -> None:
    summary_rows = []
    group_rows = []
    labels = report["window_labels"]
    for style_key, entry in report["styles"].items():
        for stage in ("select", "valid", "test"):
            window = entry["windows"][stage]
            raw, neutral = window["raw"], window["neutral"]
            net_delta = neutral["net_ls_total"] - raw["net_ls_total"]
            summary_rows.append(
                f"<tr><td>{entry['label']}</td><td>{labels[stage]}</td>"
                f"<td>{_fmt(window['coverage'], True, 1)}</td>"
                f"<td>{_fmt(raw['ic']['mean'])} → {_fmt(neutral['ic']['mean'])}</td>"
                f"<td>{_fmt(raw['ic']['ir_annual'], digits=2)} → {_fmt(neutral['ic']['ir_annual'], digits=2)}</td>"
                f"<td>{_fmt(raw['rank_ic']['mean'])} → {_fmt(neutral['rank_ic']['mean'])}</td>"
                f"<td>{_fmt(raw['rank_ic']['ir_annual'], digits=2)} → {_fmt(neutral['rank_ic']['ir_annual'], digits=2)}</td>"
                f"<td>{_fmt(raw['net_ls_total'], True, 2)} → {_fmt(neutral['net_ls_total'], True, 2)}</td>"
                f"<td class='{'good' if net_delta >= 0 else 'bad'}'>{_fmt(net_delta, True, 2)}</td>"
                f"<td>{_fmt(raw['turnover'], True, 1)} → {_fmt(neutral['turnover'], True, 1)}</td>"
                f"<td>{_fmt(window['raw_exposure_corr'])} → {_fmt(window['neutral_exposure_corr'])}</td></tr>"
            )
            groups = "".join(f"<td>{_fmt(value, True, 2)}</td>" for value in neutral["group_total"])
            group_rows.append(
                f"<tr><td>{entry['label']}</td><td>{labels[stage]}</td>{groups}"
                f"<td>{_fmt(neutral['gross_ls_total'], True, 2)}</td>"
                f"<td>{_fmt(neutral['net_ls_total'], True, 2)}</td></tr>"
            )
    definitions = "".join(
        f"<tr><td>{entry['label']}</td><td>{entry['definition']}</td></tr>"
        for entry in report["styles"].values()
    )
    destination.write_text(
        f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{report['id']} 单风格中性化</title>
<style>body{{margin:0;background:#f3f5f8;color:#172033;font:14px/1.5 system-ui,'Microsoft YaHei'}}main{{max-width:1700px;margin:auto;padding:28px}}.card{{background:#fff;border:1px solid #dde3ec;border-radius:12px;padding:18px;margin:14px 0}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}th,td{{padding:8px;border-bottom:1px solid #e4e8ef;text-align:right}}th:first-child,td:first-child{{text-align:left}}.good{{color:#087f5b}}.bad{{color:#c33}}p{{color:#667085}}code{{word-break:break-all;white-space:normal}}</style></head><body><main>
<h1>{report['id']}：10项风格逐个单独中性化</h1><p>每日截面秩空间 OLS，每次只回归一个风格暴露；原始指标也限制在该暴露有效的匹配样本中。方向固定为 {report['direction']:+d}，周频调仓，5日信号均值，Q5−Q1，30bps成本。财报按公告日后一个交易日生效。</p>
<section class="card"><h2>核心对比</h2><div class="scroll"><table><thead><tr><th>单独剔除</th><th>阶段</th><th>暴露覆盖</th><th>IC 原→中</th><th>ICIR 原→中</th><th>RankIC 原→中</th><th>RankICIR 原→中</th><th>净多空 原→中</th><th>净收益变化</th><th>换手 原→中</th><th>暴露相关 原→中</th></tr></thead><tbody>{''.join(summary_rows)}</tbody></table></div></section>
<section class="card"><h2>中性化后五分组与多空</h2><div class="scroll"><table><thead><tr><th>单独剔除</th><th>阶段</th><th>Q1</th><th>Q2</th><th>Q3</th><th>Q4</th><th>Q5</th><th>毛多空</th><th>净多空</th></tr></thead><tbody>{''.join(group_rows)}</tbody></table></div></section>
<section class="card"><h2>暴露定义</h2><table><thead><tr><th>风格</th><th>定义</th></tr></thead><tbody>{definitions}</tbody></table></section>
<section class="card"><h2>因子表达式</h2><code>{report['expression']}</code></section></main></body></html>""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--minute-parquet", default=str(MINUTE_PARQUET))
    parser.add_argument("--daily-parquet", default=str(ADJUSTED_CLOSE_PARQUET))
    parser.add_argument("--pit", default=str(ZZ500_PIT_PARQUET))
    parser.add_argument("--risk-exposures", default=str(RISK_EXPOSURES_PARQUET))
    parser.add_argument("--value-exposures", default=VALUE_EXPOSURES)
    parser.add_argument("--fundamental-exposures", default=FUNDAMENTAL_EXPOSURES)
    parser.add_argument("--momentum-window", type=int, default=60)
    parser.add_argument("--vol-window", type=int, default=60)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = "cpu" if args.cpu else "cuda"
    sidecar = Path(args.candidate)
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    genome = HandbookSkeletonGenome.from_dict(record["genome"])
    direction = int(record["fitness"]["direction"])
    start, end = WINDOWS[0][2], WINDOWS[-1][3]
    instruments = load_pit_codes(args.pit, start, end)
    dates = load_pit_dates(args.pit, start, end)
    date_index = pd.Index([str(value) for value in dates], name="trade_date")
    instrument_index = pd.Index([str(value) for value in instruments], name="instrument")

    print("[style-neutral] load cached 2018-2024 factor", flush=True)
    cached = pd.read_parquet(sidecar.with_suffix(""), columns=["instrument", "trade_date", "factor"])
    cached["trade_date"] = cached["trade_date"].astype(str)
    frame = cached.pivot(index="instrument", columns="trade_date", values="factor")
    frame = frame.reindex(index=instrument_index, columns=date_index)
    raw = torch.as_tensor(frame.to_numpy(copy=True), device=device, dtype=torch.float32)
    raw = trailing_signal_mean(raw, 5)

    print("[style-neutral] rebuild 2025-2026 holdout factor", flush=True)
    warmup_dates = [str(value) for value in dates if str(value) >= "2024-01-02"]
    context, meta = build_minute_slice(
        args.minute_parquet, "2024-01-02", end,
        fields=("open", "high", "low", "close", "volume"),
        instruments=instruments, dates=warmup_dates, device=device,
    )
    holdout = trailing_signal_mean(genome.evaluate(context, chunk_rows=args.chunk_rows), 5)
    holdout_dates = np.asarray(meta["dates"])
    use = np.flatnonzero(holdout_dates >= WINDOWS[2][2])
    full_positions = np.array([date_index.get_loc(str(holdout_dates[index])) for index in use])
    raw[:, full_positions] = holdout[:, use]

    close = load_daily_close_tensor(args.daily_parquet, dates, instruments, device=device)
    pool = load_pit_daily_mask(args.pit, dates, instruments, device=device) & torch.isfinite(close)
    fwd = tensor_rebalance_fwd_ret(close, dates, "week_end", 1)
    fwd = torch.where(pool, fwd, torch.full_like(fwd, float("nan")))
    raw = torch.where(pool, raw, torch.full_like(raw, float("nan")))

    value_styles, _industry, _levels = load_daily_exposures(
        args.value_exposures, dates, instruments,
        continuous_columns=("ln_float_market_cap", "book_to_price"),
        industry_column=None, device=device,
    )
    risk_styles, _industry, _levels = load_daily_exposures(
        args.risk_exposures, dates, instruments,
        continuous_columns=("ln_amount", "beta"),
        industry_column=None, device=device,
    )
    fundamental_styles, _industry, _levels = load_daily_exposures(
        args.fundamental_exposures, dates, instruments,
        continuous_columns=(
            "earnings_yield", "leverage", "equity_growth_yoy",
            "revenue_growth_yoy", "profit_growth_yoy",
        ),
        industry_column=None, device=device,
    )
    exposures = {
        "size": value_styles["ln_float_market_cap"],
        "nonlinear_size": _nonlinear_size(value_styles["ln_float_market_cap"]),
        "bp": value_styles["book_to_price"],
        "growth": _growth_composite(
            fundamental_styles["equity_growth_yoy"],
            fundamental_styles["revenue_growth_yoy"],
            fundamental_styles["profit_growth_yoy"],
        ),
        "momentum": _trailing_return(close, args.momentum_window),
        "earnings_yield": fundamental_styles["earnings_yield"],
        "leverage": fundamental_styles["leverage"],
        "liquidity": risk_styles["ln_amount"],
        "beta": risk_styles["beta"],
        "volatility": trailing_volatility(close, args.vol_window),
    }

    date_array = np.asarray(dates)
    report_styles = {}
    for position, (style_key, exposure) in enumerate(exposures.items(), 1):
        label, definition = STYLE_META[style_key]
        matched_pool = pool & torch.isfinite(exposure)
        matched_raw = torch.where(matched_pool, raw, torch.full_like(raw, float("nan")))
        neutralizer = BatchedNeutralizer(
            raw.shape, continuous=(exposure,), rank_space=True, min_cross_section=30
        )
        neutral = torch.where(
            matched_pool, neutralizer(raw), torch.full_like(raw, float("nan"))
        )
        windows = {}
        for stage, _window_label, window_start, window_end in WINDOWS:
            mask = torch.as_tensor(
                (date_array >= window_start) & (date_array <= window_end), device=device
            )
            windows[stage] = {
                "coverage": _coverage(exposure, pool, mask),
                "raw": evaluate_period(matched_raw, fwd, mask, direction, 5, 30.0),
                "neutral": evaluate_period(neutral, fwd, mask, direction, 5, 30.0),
                "raw_exposure_corr": exposure_corr(matched_raw, exposure, fwd, mask),
                "neutral_exposure_corr": exposure_corr(neutral, exposure, fwd, mask),
            }
        report_styles[style_key] = {
            "label": label, "definition": definition, "windows": windows,
        }
        print(f"[style-neutral] {position}/{len(exposures)} {style_key} done", flush=True)
        del matched_raw, neutral, neutralizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "id": sidecar.name.removesuffix(".parquet.json"),
        "direction": direction,
        "expression": record["expression"],
        "method": "separate daily cross-sectional rank-space OLS residuals; matched raw universe",
        "window_labels": {key: label for key, label, _start, _end in WINDOWS},
        "styles": report_styles,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_html(report, destination)
    destination.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[style-neutral] written -> {destination}", flush=True)


if __name__ == "__main__":
    main()
