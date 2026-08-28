# -*- coding: utf-8 -*-
"""Read the exported factor back and reproduce the report's headline numbers.

An export that cannot be replayed is worse than none, so this loads the
parquet through the project's own reader, rebuilds the weekly labels from
scratch, and checks RankIC / IC / turnover / net long-short against the
backtest. It touches no minute data - which is the point of exporting.
"""
import json
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

from min_gp.config import ADJUSTED_CLOSE_PARQUET, ZZ500_PIT_PARQUET
from min_gp.data import load_pit_codes, load_pit_daily_mask
from min_gp.factor_export import load_factor_parquet
from min_gp.label import tensor_rebalance_fwd_ret
from min_gp.numeric.ranking import cross_section_rank
from min_gp.spectral_data import load_daily_close_tensor
import min_gp.unified_top5_html_report as R

SIGNAL = pathlib.Path(
    "分钟级因子挖掘/output/factors/f5_ts60_vol60_20260828_signal.parquet")
REPORT = json.loads(pathlib.Path(
    "分钟级因子挖掘/output/reports/f5_n60_volsweep_20260828.json"
).read_text(encoding="utf-8"))
ARM, G, COST = "vol60", 10, 30.0 * 1e-4
DEV = "cpu"

print(f"replaying {SIGNAL.name}\n", flush=True)
series = {f: [] for f in ("gross", "net", "turnover", "ic", "rank_ic")}
bounds, previous, prev_codes = {}, None, None
total_dates = 0

import pyarrow.parquet as pq

# Dates come from the export itself, so the replay cannot silently use a
# different calendar than the file was written on.
EXPORT_DATES = sorted({
    str(d)[:10] for d in
    pq.read_table(SIGNAL, columns=["trade_date"]).column("trade_date").to_pylist()
})

for key, label, begin, finish in R.WINDOWS:
    codes = load_pit_codes(str(ZZ500_PIT_PARQUET), begin, finish)
    dates = [d for d in EXPORT_DATES if begin <= d <= finish]
    insts = sorted(codes)
    signal = load_factor_parquet(SIGNAL, insts, dates, device=DEV)
    close = load_daily_close_tensor(str(ADJUSTED_CLOSE_PARQUET), dates, insts,
                                    device=DEV)
    pool = load_pit_daily_mask(str(ZZ500_PIT_PARQUET), dates, insts, device=DEV)
    pool &= torch.isfinite(close)
    fwd = tensor_rebalance_fwd_ret(close, dates, "week_end", 1)
    fwd = torch.where(pool, fwd, torch.full_like(fwd, float("nan")))

    start_index = total_dates
    if prev_codes is not None and previous is not None:
        where = {c: i for i, c in enumerate(insts)}
        carried = torch.zeros(len(insts))
        for code, w in zip(prev_codes, previous.tolist()):
            if w != 0.0 and code in where:
                carried[where[code]] = w
        book = carried
    else:
        book = torch.zeros(len(insts))

    for day in range(len(dates)):
        score, future = signal[:, day], fwd[:, day]
        live = torch.isfinite(score) & torch.isfinite(future)
        if int(live.sum()) < max(30, G):
            continue
        index = torch.nonzero(live, as_tuple=False).squeeze(1)
        ordered = index[torch.argsort(score[index])]
        chunks = list(torch.tensor_split(ordered, G))
        w = torch.zeros(len(insts))
        w[chunks[-1]] = 1.0 / len(chunks[-1])
        w[chunks[0]] = -1.0 / len(chunks[0])
        gross = float((w * torch.nan_to_num(future)).sum())
        turn = float(0.5 * (w - book).abs().sum())
        book = w
        series["gross"].append(gross)
        series["net"].append(gross - turn * COST)
        series["turnover"].append(turn)
        ls, lf = torch.where(live, score, torch.nan), torch.where(live, future, torch.nan)

        def corr(a, b):
            m = torch.isfinite(a) & torch.isfinite(b)
            x, y = a[m].double(), b[m].double()
            x, y = x - x.mean(), y - y.mean()
            d = torch.sqrt((x * x).sum() * (y * y).sum())
            return float((x * y).sum() / d) if float(d) > 0 else float("nan")

        series["ic"].append(corr(ls, lf))
        series["rank_ic"].append(corr(cross_section_rank(ls),
                                      cross_section_rank(lf)))
        total_dates += 1
    previous, prev_codes = book, insts
    bounds[key] = (start_index, total_dates)
    print(f"  [{key}] {total_dates - start_index} 个调仓周", flush=True)


def total(v):
    return float(np.prod(1.0 + np.asarray(v, dtype=np.float64)) - 1.0)


def moments(v):
    a = np.asarray([x for x in v if np.isfinite(x)], dtype=np.float64)
    sd = a.std(ddof=1)
    return float(a.mean()), float(a.mean() / sd * np.sqrt(52))


print(f"\n{'指标':<12}{'回放文件':>12}{'回测报告':>12}{'差':>11}")
ic_m, ic_ir = moments(series["ic"])
ric_m, ric_ir = moments(series["rank_ic"])
mine = {"weeks": len(series["net"]), "rank_ic": ric_m, "rank_icir": ric_ir,
        "ic": ic_m, "icir": ic_ir, "gross": total(series["gross"]),
        "net": total(series["net"]),
        "turnover": float(np.mean(series["turnover"]))}
ref = REPORT["stats"][ARM]["full"]
worst, rows = 0.0, [
    ("调仓周数", "weeks", 0, "{:.0f}"), ("RankIC", "rank_ic", 1e-4, "{:+.4f}"),
    ("RankICIR", "rank_icir", 1e-2, "{:+.2f}"), ("IC", "ic", 1e-4, "{:+.4f}"),
    ("ICIR", "icir", 1e-2, "{:+.2f}"),
    ("毛多空", "gross", 2e-2, "{:+.3f}"), ("扣费多空", "net", 2e-2, "{:+.3f}"),
    ("换手", "turnover", 2e-3, "{:.4f}")]
bad = []
for name, key, tol, fmt in rows:
    a, b = mine[key], ref[key]
    gap = abs(a - b)
    flag = "" if gap <= tol else "   <-- 超出容差"
    if gap > tol:
        bad.append(name)
    print(f"{name:<11}{fmt.format(a):>12}{fmt.format(b):>12}"
          f"{gap:>11.2e}{flag}")

if bad:
    raise SystemExit(f"\n导出文件无法复现回测：{', '.join(bad)}")
print("\n导出文件完全复现回测（未触碰任何分钟数据）")
