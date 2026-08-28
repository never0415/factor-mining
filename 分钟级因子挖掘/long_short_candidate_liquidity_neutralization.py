"""Three-window liquidity-neutralisation check for one handbook GP export."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from min_gp.climb_candidate_neutralization import exposure_corr
from min_gp.config import ADJUSTED_CLOSE_PARQUET, MINUTE_PARQUET, RISK_EXPOSURES_PARQUET, ZZ500_PIT_PARQUET
from min_gp.data import load_pit_codes, load_pit_daily_mask, load_pit_dates
from min_gp.evaluation.incremental import trailing_signal_mean
from min_gp.evaluation.neutralize import BatchedNeutralizer, trailing_volatility
from min_gp.factors.handbook_skeleton import HandbookSkeletonGenome
from min_gp.label import tensor_rebalance_fwd_ret
from min_gp.long_short_battle_html_report import WINDOWS, evaluate_period
from min_gp.spectral_data import build_minute_slice, load_daily_close_tensor, load_daily_exposures


def fmt(x, pct=False, digits=4):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{digits}%}" if pct else f"{x:.{digits}f}"


def write_html(report, path):
    rows = []
    labels = report["window_labels"]
    exposure_label = report["exposure_label"]
    for key in ("select", "valid", "test"):
        w = report["windows"][key]
        for mode, title in (("raw", "原始"), ("neutral", f"{exposure_label}中性")):
            m = w[mode]
            corr = w["raw_exposure_corr"] if mode == "raw" else w["neutral_exposure_corr"]
            corr_text = " / ".join(f"{name}:{fmt(value)}" for name, value in corr.items())
            groups = "".join(f"<td>{fmt(x, True, 2)}</td>" for x in m["group_total"])
            rows.append(f"<tr class='{mode}'><td>{labels[key]}</td><td>{title}</td>"
                f"<td>{fmt(m['ic']['mean'])}</td><td>{fmt(m['ic']['ir_annual'], digits=2)}</td>"
                f"<td>{fmt(m['rank_ic']['mean'])}</td><td>{fmt(m['rank_ic']['ir_annual'], digits=2)}</td>"
                f"{groups}<td>{fmt(m['gross_ls_total'], True, 2)}</td>"
                f"<td>{fmt(m['net_ls_total'], True, 2)}</td><td>{fmt(m['turnover'], True, 1)}</td>"
                f"<td>{corr_text}</td></tr>")
    path.write_text(f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{report['id']} {exposure_label}中性化</title>
<style>body{{margin:0;background:#f3f5f8;color:#172033;font:14px/1.5 system-ui,'Microsoft YaHei'}}main{{max-width:1500px;margin:auto;padding:28px}}.card{{background:#fff;border:1px solid #dde3ec;border-radius:12px;padding:18px;margin:14px 0}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}th,td{{padding:9px;border-bottom:1px solid #e4e8ef;text-align:right}}th:first-child,td:first-child{{text-align:left}}.neutral{{background:#edf8f4}}code{{word-break:break-all;white-space:normal}}p{{color:#667085}}</style></head><body><main>
<h1>{report['id']}：{exposure_label}中性化</h1><p>每日截面排名空间 OLS：factor rank ~ 1 + {report['exposure_column']} rank；使用残差作为中性化因子。方向固定为 {report['direction']:+d}，周频调仓，5日信号均值，Q5−Q1，双边30bps成本。</p>
<section class="card"><div class="scroll"><table><thead><tr><th>阶段</th><th>处理</th><th>IC</th><th>ICIR</th><th>RankIC</th><th>RankICIR</th><th>Q1</th><th>Q2</th><th>Q3</th><th>Q4</th><th>Q5</th><th>毛多空</th><th>扣费多空</th><th>换手</th><th>与{exposure_label}秩相关</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<section class="card"><h2>表达式</h2><code>{report['expression']}</code></section></main></body></html>""", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--minute-parquet", default=str(MINUTE_PARQUET))
    ap.add_argument("--daily-parquet", default=str(ADJUSTED_CLOSE_PARQUET))
    ap.add_argument("--risk-exposures", default=str(RISK_EXPOSURES_PARQUET))
    ap.add_argument("--exposure-column", default="ln_amount")
    ap.add_argument("--vol-window", type=int, default=60)
    ap.add_argument("--pit", default=str(ZZ500_PIT_PARQUET))
    ap.add_argument("--chunk-rows", type=int, default=4096)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    device = "cpu" if args.cpu else "cuda"
    sidecar = Path(args.candidate)
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    genome = HandbookSkeletonGenome.from_dict(record["genome"])
    direction = int(record["fitness"]["direction"])
    start, end = WINDOWS[0][2], WINDOWS[-1][3]
    instruments = load_pit_codes(args.pit, start, end)
    dates = load_pit_dates(args.pit, start, end)
    date_index = pd.Index([str(x) for x in dates], name="trade_date")
    instrument_index = pd.Index([str(x) for x in instruments], name="instrument")

    print("[liq-neutral] load cached 2018-2024 factor", flush=True)
    cached = pd.read_parquet(sidecar.with_suffix(""), columns=["instrument", "trade_date", "factor"])
    cached["trade_date"] = cached["trade_date"].astype(str)
    frame = cached.pivot(index="instrument", columns="trade_date", values="factor")
    frame = frame.reindex(index=instrument_index, columns=date_index)
    raw = torch.as_tensor(frame.to_numpy(copy=True), device=device, dtype=torch.float32)
    raw = trailing_signal_mean(raw, 5)

    warmup_dates = [str(x) for x in dates if str(x) >= "2024-01-02"]
    print("[liq-neutral] rebuild 2025-2026 holdout factor", flush=True)
    context, meta = build_minute_slice(args.minute_parquet, "2024-01-02", end,
        fields=("open", "high", "low", "close", "volume"), instruments=instruments,
        dates=warmup_dates, device=device)
    holdout = trailing_signal_mean(genome.evaluate(context, chunk_rows=args.chunk_rows), 5)
    holdout_dates = np.asarray(meta["dates"])
    use = np.flatnonzero(holdout_dates >= WINDOWS[2][2])
    full_pos = np.array([date_index.get_loc(str(holdout_dates[i])) for i in use])
    raw[:, full_pos] = holdout[:, use]

    close = load_daily_close_tensor(args.daily_parquet, dates, instruments, device=device)
    pool = load_pit_daily_mask(args.pit, dates, instruments, device=device) & torch.isfinite(close)
    fwd = tensor_rebalance_fwd_ret(close, dates, "week_end", 1)
    fwd = torch.where(pool, fwd, torch.full_like(fwd, float("nan")))
    raw = torch.where(pool, raw, torch.full_like(raw, float("nan")))
    exposure_columns = tuple(value.strip() for value in args.exposure_column.split(",") if value.strip())
    external_columns = tuple(column for column in exposure_columns if column != "trailing_volatility")
    styles = {}
    if external_columns:
        styles, _industries, _levels = load_daily_exposures(args.risk_exposures, dates, instruments,
            continuous_columns=external_columns, industry_column=None, device=device)
    if "trailing_volatility" in exposure_columns:
        styles["trailing_volatility"] = trailing_volatility(close, args.vol_window)
    exposures = tuple(styles[column] for column in exposure_columns)
    neutralizer = BatchedNeutralizer(raw.shape, continuous=exposures,
        rank_space=True, min_cross_section=30)
    neutral = torch.where(pool, neutralizer(raw), torch.full_like(raw, float("nan")))

    date_array = np.asarray(dates)
    windows = {}
    for key, _label, wstart, wend in WINDOWS:
        mask = torch.as_tensor((date_array >= wstart) & (date_array <= wend), device=device)
        windows[key] = {
            "raw": evaluate_period(raw, fwd, mask, direction, 5, 30.0),
            "neutral": evaluate_period(neutral, fwd, mask, direction, 5, 30.0),
            "raw_exposure_corr": {column: exposure_corr(raw, styles[column], fwd, mask) for column in exposure_columns},
            "neutral_exposure_corr": {column: exposure_corr(neutral, styles[column], fwd, mask) for column in exposure_columns},
        }
    report = {"generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "id": sidecar.name.removesuffix(".parquet.json"), "direction": direction,
        "expression": record["expression"], "exposure_column": ",".join(exposure_columns),
        "exposure_label": "流动性＋市场Beta" if set(exposure_columns) == {"ln_amount", "beta"} else ("市场Beta" if exposure_columns == ("beta",) else ("流通市值" if exposure_columns == ("ln_float_market_cap",) else (f"{args.vol_window}日波动率" if exposure_columns == ("trailing_volatility",) else "流动性"))),
        "definition": {"exposures": exposure_columns, "method": "joint daily cross-sectional rank-space OLS residual",
            "trailing_volatility": f"{args.vol_window}-day standard deviation of daily close returns ending at signal date"},
        "window_labels": {k: label for k, label, _s, _e in WINDOWS}, "windows": windows}
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_html(report, destination)
    destination.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    print(f"[liq-neutral] written -> {destination}", flush=True)


if __name__ == "__main__":
    main()
