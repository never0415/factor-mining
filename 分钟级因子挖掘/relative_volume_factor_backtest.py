"""Backtest the gap-dispersion factor with causal same-minute relative volume.

The minute weight is

    sqrt(clip(volume[t, m] / median(volume[t-k, m], k=1..20), 0, 5))

where the median excludes the current day and needs at least five valid prior
observations.  The rest of the signal reproduces the v7 candidate structure:
minute OHLC CV^2 * has_gap, daily sum, trailing 20-day sum, then the standard
external-report treatment (5-day trailing mean and cross-sectional MAD5).
"""

from __future__ import annotations

import argparse
import html
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from min_gp.config import (
    ADJUSTED_CLOSE_PARQUET,
    INDEX_DAILY_PARQUET,
    MINUTE_PARQUET,
    ZZ500_PIT_PARQUET,
)
from min_gp.data import build_slice, load_pit_codes, load_pit_daily_mask, load_pit_dates
from min_gp.label import tensor_rebalance_fwd_ret
from min_gp.numeric.preprocessing import remove_outliers
from min_gp.numeric.ranking import cross_section_rank
from min_gp.operators.intraday import rolling_ohlc_dispersion
from min_gp.report_candidates import _pearson, quantile_curves, series_stats
from min_gp.spectral_data import load_daily_close_tensor


START = "2018-01-02"
END = "2026-07-31"
STAGES = (
    ("select", "样本期", "2018-01-02", "2022-12-31"),
    ("valid", "验证期", "2023-01-01", "2024-12-31"),
    ("test", "测试期", "2025-01-01", "2026-07-31"),
)
EXPRESSION = (
    "seed_ts_sum_daily(seed_day_sum_minute(seed_mul_minute("
    "rolling_ohlc_dispersion(open, high, low, close, window=5), "
    "seed_mul_minute(has_gap, sqrt(clip(volume / "
    "same_minute_median(volume, lookback=20, exclude_today=True, min_periods=5), "
    "0, 5))))), window=20)"
)


def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_safe(v) for v in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _rolling_sum(x: torch.Tensor, window: int) -> torch.Tensor:
    clean = torch.nan_to_num(x.float())
    valid = torch.isfinite(x).float()
    total = clean.cumsum(1)
    count = valid.cumsum(1)
    if window < x.shape[1]:
        total[:, window:] -= total[:, :-window].clone()
        count[:, window:] -= count[:, :-window].clone()
    return torch.where(count >= 1, total, torch.full_like(total, float("nan")))


def _rolling_mean(x: torch.Tensor, window: int) -> torch.Tensor:
    clean = torch.nan_to_num(x.float())
    valid = torch.isfinite(x).float()
    total = clean.cumsum(1)
    count = valid.cumsum(1)
    if window < x.shape[1]:
        total[:, window:] -= total[:, :-window].clone()
        count[:, window:] -= count[:, :-window].clone()
    return torch.where(
        count >= 1,
        total / count.clamp(min=1),
        torch.full_like(total, float("nan")),
    )


@torch.inference_mode()
def _compute_year(
    year: int,
    minute_path: str,
    pit_path: str,
    device: str,
    relative_lookback: int,
    relative_cap: float,
    min_history: int,
    outer_window: int,
    max_stocks: int | None,
) -> dict:
    begin = max(START, f"{year}-01-01")
    finish = min(END, f"{year}-12-31")
    instruments = load_pit_codes(pit_path, begin, finish)
    if max_stocks:
        instruments = instruments[:max_stocks]
    print(
        f"[relative-volume] {year}: loading {begin}..{finish}, "
        f"target instruments={len(instruments)}",
        flush=True,
    )
    tensors, _masks, _fwd, meta = build_slice(
        minute_path,
        begin,
        finish,
        instruments=instruments,
        device=device,
        fp=torch.bfloat16,
        extend_days=45,
    )
    open_, high, low, close, volume, has_gap = (
        tensors[name] for name in ("open", "high", "low", "close", "volume", "has_gap")
    )
    del tensors, _masks, _fwd
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    instruments_n, days, _minutes = volume.shape
    daily_event = torch.zeros(
        (instruments_n, days), dtype=torch.float32, device=device
    )
    progress_step = max(1, days // 5)
    for day in range(days):
        history_start = max(0, day - relative_lookback)
        history = volume[:, history_start:day].float()
        if history.shape[1] == 0:
            continue
        finite_history = torch.isfinite(history)
        count = finite_history.sum(1)
        baseline = torch.nanmedian(history, dim=1).values
        current = volume[:, day].float()
        dispersion = rolling_ohlc_dispersion(
            open_[:, day : day + 1],
            high[:, day : day + 1],
            low[:, day : day + 1],
            close[:, day : day + 1],
            window=5,
        )[:, 0]
        valid_relative = (
            torch.isfinite(current)
            & torch.isfinite(baseline)
            & baseline.gt(0)
            & count.ge(min_history)
        )
        relative = torch.where(
            valid_relative,
            current / baseline.clamp(min=1e-12),
            torch.full_like(current, float("nan")),
        )
        weight = torch.sqrt(relative.clamp(min=0, max=relative_cap))
        valid_term = has_gap[:, day].bool() & torch.isfinite(dispersion) & torch.isfinite(weight)
        term = torch.where(valid_term, dispersion * weight, torch.zeros_like(dispersion))
        daily_event[:, day] = term.sum(-1)
        if (day + 1) % progress_step == 0 or day + 1 == days:
            print(f"[relative-volume] {year}: minute score {day + 1}/{days}", flush=True)

    factor = _rolling_sum(daily_event, outer_window)
    date_array = np.asarray([str(value)[:10] for value in meta["dates"]])
    keep = (date_array >= begin) & (date_array <= finish)
    result = {
        "dates": date_array[keep].tolist(),
        "instruments": list(meta["instruments"]),
        "factor": factor[:, torch.as_tensor(keep, device=device)].cpu(),
        "warmup": int(meta.get("warmup", 0)),
    }
    del open_, high, low, close, volume, has_gap, daily_event, factor
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def _assemble(parts: list[dict], dates: list[str], instruments: list[str]) -> torch.Tensor:
    output = torch.full((len(instruments), len(dates)), float("nan"), dtype=torch.float32)
    instrument_index = {value: i for i, value in enumerate(instruments)}
    date_index = {str(value)[:10]: i for i, value in enumerate(dates)}
    for part in parts:
        ii = torch.as_tensor([instrument_index[x] for x in part["instruments"]])
        dd = torch.as_tensor([date_index[x] for x in part["dates"]])
        output[ii[:, None], dd[None, :]] = part["factor"]
    return output


def _index_close(index_path: str, dates: list[str], device: str) -> torch.Tensor:
    frame = pq.read_table(index_path, columns=["trade_date", "close"]).to_pandas()
    frame["trade_date"] = frame["trade_date"].astype(str).str.slice(0, 10)
    mapping = frame.drop_duplicates("trade_date", keep="last").set_index("trade_date")["close"]
    values = np.asarray([mapping.get(str(day)[:10], np.nan) for day in dates], dtype=np.float32)
    return torch.as_tensor(values, device=device).unsqueeze(0)


def _total(values: np.ndarray) -> float:
    return float(np.prod(1.0 + values) - 1.0) if values.size else float("nan")


def _annual(values: np.ndarray) -> float:
    if not values.size:
        return float("nan")
    wealth = float(np.prod(1.0 + values))
    return wealth ** (52.0 / len(values)) - 1.0 if wealth > 0 else float("nan")


def _stage_metrics(
    factor: torch.Tensor,
    fwd: torch.Tensor,
    direction: int,
    date_array: np.ndarray,
    group_returns: list[np.ndarray],
    gross: np.ndarray,
    net: np.ndarray,
    turnover: np.ndarray,
    curve_dates: np.ndarray,
    benchmark: np.ndarray,
) -> dict:
    output = {}
    rank_y = cross_section_rank(fwd)
    for key, label, begin, finish in STAGES:
        daily_mask_np = (date_array >= begin) & (date_array <= finish)
        daily_mask = torch.as_tensor(daily_mask_np, device=factor.device)
        scoped_factor = torch.where(
            daily_mask.unsqueeze(0), factor, torch.full_like(factor, float("nan"))
        )
        scoped_fwd = torch.where(
            daily_mask.unsqueeze(0), fwd, torch.full_like(fwd, float("nan"))
        )
        valid = torch.isfinite(scoped_factor) & torch.isfinite(scoped_fwd)
        ic = (_pearson(scoped_factor, scoped_fwd, valid, preprocess=False) * direction).cpu().numpy()
        rank_ic = (
            _pearson(cross_section_rank(scoped_factor), rank_y, valid, preprocess=False)
            * direction
        ).cpu().numpy()
        weekly = (curve_dates >= begin) & (curve_dates <= finish)
        groups = [values[weekly] for values in group_returns]
        gross_stage, net_stage = gross[weekly], net[weekly]
        turn_stage, benchmark_stage = turnover[weekly], benchmark[weekly]
        output[key] = {
            "label": label,
            "begin": begin,
            "finish": finish,
            "ic": series_stats(ic, 52),
            "rank_ic": series_stats(rank_ic, 52),
            "coverage": float(valid.sum() / torch.isfinite(scoped_fwd).sum().clamp(min=1)),
            "weeks": int(weekly.sum()),
            "group_total": [_total(value) for value in groups],
            "gross_total": _total(gross_stage),
            "net_total": _total(net_stage),
            "net_annual": _annual(net_stage),
            "benchmark_total": _total(benchmark_stage[np.isfinite(benchmark_stage)]),
            # One-side turnover. quantile_curves stores the L1 change of both legs.
            "turnover": float(np.mean(turn_stage) / 2.0) if turn_stage.size else float("nan"),
        }
    return output


def _pct(value: float | None, digits: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:+.{digits}%}"


def _num(value: float | None, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:+.{digits}f}"


def _nice_ticks(low: float, high: float) -> np.ndarray:
    span = max(high - low, 0.5)
    raw = span / 7
    choices = np.asarray([0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 20])
    step = float(choices[np.searchsorted(choices, raw, side="left")])
    lo = math.floor(min(low, 0) / step) * step
    hi = math.ceil(max(high, 0) / step) * step
    return np.arange(lo, hi + step * 0.5, step)


def _build_svg(report: dict) -> str:
    width, height = 1200, 620
    left, right, top, bottom = 62.0, 1070.0, 82.0, 530.0
    dates = pd.to_datetime(report["curve_dates"])
    series = report["series"]
    colors = [
        "#a63c2c", "#b95f48", "#c77a66", "#c79b7b", "#bca997",
        "#78a797", "#5a9580", "#397f68", "#147258", "#006b50",
    ]
    chart_series = []
    for idx, values in enumerate(series["groups"]):
        chart_series.append((f"D{idx + 1}", np.asarray(values), colors[idx], 1.05, ""))
    chart_series.extend(
        [
            ("毛多空", np.asarray(series["gross"]), "#536a86", 1.5, "5 4"),
            ("扣费多空", np.asarray(series["net"]), "#b47b00", 2.8, ""),
            ("中证500", np.asarray(series["benchmark"]), "#465163", 1.5, "3 3"),
        ]
    )
    finite_values = np.concatenate([value[np.isfinite(value)] for _, value, _, _, _ in chart_series])
    ticks = _nice_ticks(float(finite_values.min()), float(finite_values.max()))
    y_low, y_high = float(ticks[0]), float(ticks[-1])

    def x(position: int) -> float:
        return left + position * (right - left) / max(1, len(dates) - 1)

    def y(value: float) -> float:
        return bottom - (value - y_low) * (bottom - top) / max(1e-12, y_high - y_low)

    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        'role="img" aria-label="同分钟历史相对成交量因子十分组回测">',
        '<style>text{font-family:"Microsoft YaHei",Arial,sans-serif;fill:#283548}'
        '.grid{stroke:#e8edf3;stroke-width:1}.zero{stroke:#aeb8c5;stroke-width:1.1}'
        '.bound{stroke:#d7dfe8;stroke-width:1.2;stroke-dasharray:4 3}'
        '.small{font-size:10px}.stage{font-size:13px;font-weight:700;text-anchor:middle}'
        '.metric{font-size:10px;text-anchor:middle}.axis{font-size:10px;fill:#65748a}'
        '.end{font-size:10px;font-weight:600}.line{fill:none;stroke-linejoin:round;stroke-linecap:round}'
        '</style>',
        f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" fill="#fff"/>',
    ]
    for tick in ticks:
        yy = y(float(tick))
        cls = "grid zero" if abs(tick) < 1e-12 else "grid"
        chunks.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{right}" y2="{yy:.1f}" class="{cls}"/>')
        chunks.append(f'<text x="{left-9}" y="{yy+3.5:.1f}" class="axis" text-anchor="end">{tick:+.0%}</text>')

    boundary_positions = []
    for boundary in ("2023-01-01", "2025-01-01"):
        pos = int(np.searchsorted(dates.values, np.datetime64(boundary)))
        boundary_positions.append(pos)
        xx = x(pos)
        chunks.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{bottom}" class="bound"/>')
    first_boundary_x, second_boundary_x = (x(v) for v in boundary_positions)
    chunks.append(
        f'<rect x="{first_boundary_x:.1f}" y="{top}" width="{second_boundary_x-first_boundary_x:.1f}" '
        f'height="{bottom-top}" fill="#f6f8fb"/>'
    )
    stage_x = [(left + first_boundary_x) / 2, (first_boundary_x + second_boundary_x) / 2, (second_boundary_x + right) / 2]
    for center, (key, label, begin, finish) in zip(stage_x, STAGES):
        metric = report["stages"][key]
        date_label = f"{begin[:7]}–{finish[:7]}"
        chunks.append(f'<text x="{center:.1f}" y="18" class="stage">{label} {date_label}</text>')
        chunks.append(
            f'<text x="{center:.1f}" y="33" class="metric">RankIC {_num(metric["rank_ic"]["mean"])}　'
            f'RankICIR {_num(metric["rank_ic"]["ir_annual"],2)}</text>'
        )
        chunks.append(
            f'<text x="{center:.1f}" y="47" class="metric">IC {_num(metric["ic"]["mean"])}　'
            f'ICIR {_num(metric["ic"]["ir_annual"],2)}</text>'
        )
        chunks.append(
            f'<text x="{center:.1f}" y="61" class="metric">扣费多空 {_pct(metric["net_total"])}　'
            f'基准 {_pct(metric["benchmark_total"])}</text>'
        )
        chunks.append(
            f'<text x="{center:.1f}" y="75" class="metric">换手 {metric["turnover"]:.0%}　{metric["weeks"]}周</text>'
        )

    tick_positions = np.linspace(0, len(dates) - 1, 10).round().astype(int)
    for pos in tick_positions:
        chunks.append(f'<text x="{x(int(pos)):.1f}" y="{bottom+20}" class="axis" text-anchor="middle">{dates[int(pos)]:%Y-%m}</text>')

    endpoint_labels = []
    for name, values, color, line_width, dash in chart_series:
        points = []
        for index, value in enumerate(values):
            if np.isfinite(value):
                points.append(f"{x(index):.1f},{y(float(value)):.1f}")
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        chunks.append(
            f'<polyline points="{" ".join(points)}" class="line" stroke="{color}" '
            f'stroke-width="{line_width}"{dash_attr}/>'
        )
        final = float(values[np.flatnonzero(np.isfinite(values))[-1]])
        endpoint_labels.append([y(final), name, final, color])
    endpoint_labels.sort(key=lambda item: item[0])
    gap = 13.0
    cursor = top + 5
    for item in endpoint_labels:
        item[0] = max(float(item[0]), cursor)
        cursor = item[0] + gap
    overflow = max(0.0, endpoint_labels[-1][0] - (bottom - 2))
    for item in endpoint_labels:
        item[0] -= overflow
    for yy, name, final, color in endpoint_labels:
        label = f"{name} {_pct(final,0)}"
        chunks.append(f'<text x="{right+8}" y="{yy+3:.1f}" class="end" fill="{color}" style="fill:{color}">{label}</text>')

    legend_y = 588
    positions = np.linspace(35, 1080, len(chart_series))
    for xx, (name, values, color, _line_width, _dash) in zip(positions, chart_series):
        final = float(values[np.flatnonzero(np.isfinite(values))[-1]])
        chunks.append(f'<rect x="{xx:.1f}" y="{legend_y-8}" width="8" height="8" rx="1" fill="{color}"/>')
        chunks.append(f'<text x="{xx+12:.1f}" y="{legend_y}" class="small">{name} {_pct(final,0)}</text>')
    chunks.append("</svg>")
    return "".join(chunks)


def _build_html(report: dict, svg: str, destination: Path) -> None:
    stage_rows = []
    for key, label, _begin, _finish in STAGES:
        value = report["stages"][key]
        groups = "".join(f"<td>{_pct(x)}</td>" for x in value["group_total"])
        stage_rows.append(
            f"<tr><td>{label}</td><td>{_num(value['ic']['mean'])}</td>"
            f"<td>{_num(value['ic']['ir_annual'],2)}</td><td>{_num(value['rank_ic']['mean'])}</td>"
            f"<td>{_num(value['rank_ic']['ir_annual'],2)}</td>{groups}"
            f"<td>{_pct(value['gross_total'])}</td><td>{_pct(value['net_total'])}</td>"
            f"<td>{_pct(value['net_annual'])}</td><td>{value['turnover']:.1%}</td>"
            f"<td>{_pct(value['benchmark_total'])}</td><td>{value['weeks']}</td></tr>"
        )
    formula = html.escape(report["expression"])
    payload = json.dumps(_safe(report), ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    destination.write_text(
        f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>同分钟历史相对成交量因子回测</title>
<style>:root{{--bg:#f5f7fa;--card:#fff;--ink:#1d2939;--muted:#667085;--line:#dfe5ec}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1500px;margin:auto;padding:24px}}h1{{margin:0 0 5px}}.sub,.note{{color:var(--muted)}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:14px 0;box-shadow:0 2px 10px #1720330a}}
.chart{{padding:0;overflow:auto}}svg{{display:block;width:100%;height:auto;min-width:1050px}}code{{word-break:break-all;white-space:normal}}
.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}th,td{{padding:8px;border-bottom:1px solid #e7ebf0;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#f8fafc;position:sticky;top:0}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}.metric{{border:1px solid var(--line);border-radius:8px;padding:10px}}.metric b{{display:block;font-size:18px}}
@media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}</style></head><body><main>
<h1>同分钟历史相对成交量版：十分组回测</h1>
<div class="sub">生成于 {report['generated']} · 中证500 PIT股票池 · 周末调仓 · 样本期锁定方向 {report['direction']:+d} · 截面MAD5 · 双边30bps</div>
<section class="card chart">{svg}</section>
<section class="card"><h2>因子定义</h2><p><code>{formula}</code></p>
<p class="note">同分钟基准为本股票过去20个交易日该分钟成交量的中位数，严格排除当天；至少5个有效历史值。相对成交量截断到[0,5]后开平方。若历史基准≤0或不足5日，该分钟权重记为缺失，不使用epsilon放大。</p></section>
<section class="card"><h2>完整指标</h2><div class="scroll"><table><thead><tr><th>阶段</th><th>IC</th><th>ICIR</th><th>RankIC</th><th>RankICIR</th>
{''.join(f'<th>D{i}</th>' for i in range(1,11))}<th>毛多空</th><th>扣费多空</th><th>扣费年化</th><th>换手</th><th>中证500</th><th>周数</th></tr></thead><tbody>{''.join(stage_rows)}</tbody></table></div></section>
<section class="card"><h2>口径说明</h2><div class="grid"><div class="metric">信号方向<b>{report['direction']:+d}</b><small>只用2018–2022样本期RankIC均值决定</small></div>
<div class="metric">相对量参数<b>20日 / cap 5</b><small>同股同分钟历史中位数</small></div><div class="metric">持有与分组<b>周频 / 十分组</b><small>D10−D1，两端各10%</small></div>
<div class="metric">成本<b>双边30bps</b><small>按实际持仓变化扣费</small></div></div>
<p class="note">图中三阶段连续复利，2023与2025边界不清仓；阶段表由同一条全历史持仓序列切片，因此验证期、测试期首周换手承接上一阶段。成交量归一化、20日因子累计、5日平滑均为后向窗口。</p></section>
<script id="report-data" type="application/json">{payload}</script></main></body></html>""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--minute-parquet", default=str(MINUTE_PARQUET))
    parser.add_argument("--daily-parquet", default=str(ADJUSTED_CLOSE_PARQUET))
    parser.add_argument("--index-parquet", default=str(INDEX_DAILY_PARQUET))
    parser.add_argument("--pit", default=str(ZZ500_PIT_PARQUET))
    parser.add_argument("--relative-lookback", type=int, default=20)
    parser.add_argument("--relative-cap", type=float, default=5.0)
    parser.add_argument("--min-history", type=int, default=5)
    parser.add_argument("--outer-window", type=int, default=20)
    parser.add_argument("--signal-average-days", type=int, default=5)
    parser.add_argument("--cost-bps", type=float, default=30.0)
    parser.add_argument("--max-stocks", type=int)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = "cpu" if args.cpu else "cuda"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dates = [str(value)[:10] for value in load_pit_dates(args.pit, START, END)]
    instruments = load_pit_codes(args.pit, START, END)
    if args.max_stocks:
        instruments = instruments[: args.max_stocks]

    parts = []
    for year in range(2018, 2027):
        parts.append(
            _compute_year(
                year,
                args.minute_parquet,
                args.pit,
                device,
                args.relative_lookback,
                args.relative_cap,
                args.min_history,
                args.outer_window,
                args.max_stocks,
            )
        )
    raw_factor = _assemble(parts, dates, instruments)
    del parts
    factor = _rolling_mean(raw_factor.to(device), args.signal_average_days)
    close = load_daily_close_tensor(args.daily_parquet, dates, instruments, device=device)
    pool = load_pit_daily_mask(args.pit, dates, instruments, device=device) & torch.isfinite(close)
    factor = torch.where(pool, factor, torch.full_like(factor, float("nan")))
    factor = remove_outliers(factor, n_mad=5.0, dim=0)
    fwd = tensor_rebalance_fwd_ret(close, dates, "week_end", 1)
    fwd = torch.where(pool, fwd, torch.full_like(fwd, float("nan")))
    date_array = np.asarray(dates)

    select = torch.as_tensor(
        (date_array >= STAGES[0][2]) & (date_array <= STAGES[0][3]), device=device
    )
    raw_valid = torch.isfinite(factor) & torch.isfinite(fwd) & select.unsqueeze(0)
    raw_rank_ic = _pearson(
        cross_section_rank(factor), cross_section_rank(fwd), raw_valid, preprocess=False
    )
    sample_mean = float(torch.nanmean(raw_rank_ic).item())
    direction = 1 if sample_mean >= 0 else -1
    print(
        f"[relative-volume] sample raw RankIC={sample_mean:+.6f}; locked direction={direction:+d}",
        flush=True,
    )

    grouped, gross, net, turnover, days = quantile_curves(
        factor, fwd, direction, 10, args.cost_bps
    )
    curve_dates = date_array[np.asarray(days, dtype=np.int64)]
    index_close = _index_close(args.index_parquet, dates, device)
    index_fwd = tensor_rebalance_fwd_ret(index_close, dates, "week_end", 1)[0]
    benchmark_returns = index_fwd[torch.as_tensor(days, device=device)].cpu().numpy()
    stages = _stage_metrics(
        factor, fwd, direction, date_array, grouped, gross, net, turnover,
        curve_dates, benchmark_returns,
    )

    group_nav = [(np.cumprod(1.0 + value) - 1.0).tolist() for value in grouped]
    gross_nav = (np.cumprod(1.0 + gross) - 1.0).tolist()
    net_nav = (np.cumprod(1.0 + net) - 1.0).tolist()
    benchmark_nav = (
        np.cumprod(1.0 + np.nan_to_num(benchmark_returns, nan=0.0)) - 1.0
    ).tolist()
    report = {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "expression": EXPRESSION,
        "direction": direction,
        "sample_raw_rank_ic": sample_mean,
        "parameters": {
            "relative_lookback": args.relative_lookback,
            "relative_cap": args.relative_cap,
            "min_history": args.min_history,
            "outer_window": args.outer_window,
            "signal_average_days": args.signal_average_days,
            "outlier_mad": 5.0,
            "groups": 10,
            "cost_bps": args.cost_bps,
        },
        "universe": {"instruments": len(instruments), "trading_days": len(dates)},
        "curve_dates": curve_dates.tolist(),
        "series": {
            "groups": group_nav,
            "gross": gross_nav,
            "net": net_nav,
            "benchmark": benchmark_nav,
        },
        "stages": stages,
    }
    svg = _build_svg(report)
    svg_path = out.with_suffix(".svg")
    json_path = out.with_suffix(".json")
    factor_path = out.with_suffix(".factor.pt")
    svg_path.write_text(svg, encoding="utf-8")
    json_path.write_text(
        json.dumps(_safe(report), ensure_ascii=False, allow_nan=False, indent=2),
        encoding="utf-8",
    )
    torch.save(
        {
            "dates": dates,
            "instruments": instruments,
            "raw_factor": raw_factor,
            "expression": EXPRESSION,
            "parameters": report["parameters"],
        },
        factor_path,
    )
    _build_html(report, svg, out)
    print(f"[relative-volume] report -> {out}", flush=True)
    print(f"[relative-volume] chart  -> {svg_path}", flush=True)
    print(f"[relative-volume] data   -> {json_path}", flush=True)
    print(f"[relative-volume] factor -> {factor_path}", flush=True)


if __name__ == "__main__":
    main()
