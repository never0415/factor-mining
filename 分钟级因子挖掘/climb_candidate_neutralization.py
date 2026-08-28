"""Compare one exported Climb-Mountain factor before/after liquidity-vol neutralisation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from min_gp.climb_mountain_html_report import WINDOWS, evaluate_period
from min_gp.config import (
    ADJUSTED_CLOSE_PARQUET, MINUTE_PARQUET, RISK_EXPOSURES_PARQUET,
    ZZ500_PIT_PARQUET,
)
from min_gp.data import load_pit_codes, load_pit_daily_mask, load_pit_dates
from min_gp.evaluation.incremental import trailing_signal_mean
from min_gp.evaluation.neutralize import BatchedNeutralizer, trailing_volatility
from min_gp.factors.climb_skeleton import ClimbMountainSkeletonGenome
from min_gp.label import tensor_rebalance_fwd_ret
from min_gp.numeric.ranking import cross_section_rank
from min_gp.report_candidates import _pearson
from min_gp.spectral_data import (
    build_minute_slice, load_daily_close_tensor, load_daily_exposures,
)


def exposure_corr(factor, exposure, fwd, date_mask):
    a, b = cross_section_rank(factor), cross_section_rank(exposure)
    valid = (
        torch.isfinite(a) & torch.isfinite(b) & torch.isfinite(fwd)
        & date_mask.unsqueeze(0)
    )
    values = _pearson(a, b, valid).cpu().numpy()
    return float(np.nanmedian(values)) if np.isfinite(values).any() else float("nan")


def fmt(x, pct=False, digits=4):
    if not np.isfinite(x):
        return "—"
    return f"{x:.{digits}%}" if pct else f"{x:.{digits}f}"


def write_html(report, path):
    rows = []
    labels = {key: label for key, label, _s, _e in WINDOWS}
    for key, _label, _s, _e in WINDOWS:
        raw, neutral = report["windows"][key]["raw"], report["windows"][key]["neutral"]
        rows.append(
            f"<tr><td>{labels[key]}</td><td>原始</td><td>{fmt(raw['ic']['mean'])}</td>"
            f"<td>{fmt(raw['ic']['ir_annual'], digits=2)}</td><td>{fmt(raw['rank_ic']['mean'])}</td>"
            f"<td>{fmt(raw['rank_ic']['ir_annual'], digits=2)}</td><td>{fmt(raw['gross_ls_total'], True, 2)}</td>"
            f"<td>{fmt(raw['net_ls_total'], True, 2)}</td><td>{fmt(raw['turnover'], True, 1)}</td>"
            f"<td>{fmt(report['windows'][key]['raw_liq_corr'])}</td><td>{fmt(report['windows'][key]['raw_vol_corr'])}</td></tr>"
        )
        rows.append(
            f"<tr class='n'><td>{labels[key]}</td><td>流动性+波动中性</td><td>{fmt(neutral['ic']['mean'])}</td>"
            f"<td>{fmt(neutral['ic']['ir_annual'], digits=2)}</td><td>{fmt(neutral['rank_ic']['mean'])}</td>"
            f"<td>{fmt(neutral['rank_ic']['ir_annual'], digits=2)}</td><td>{fmt(neutral['gross_ls_total'], True, 2)}</td>"
            f"<td>{fmt(neutral['net_ls_total'], True, 2)}</td><td>{fmt(neutral['turnover'], True, 1)}</td>"
            f"<td>{fmt(report['windows'][key]['neutral_liq_corr'])}</td><td>{fmt(report['windows'][key]['neutral_vol_corr'])}</td></tr>"
        )
    payload = json.dumps(report, ensure_ascii=False).replace("</", "<\\/")
    path.write_text(f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>candidate_0000 中性化对比</title>
<style>body{{font:14px/1.55 system-ui,'Microsoft YaHei';background:#f3f5f8;color:#172033;margin:0}}main{{max-width:1280px;margin:auto;padding:28px}}.card{{background:white;border:1px solid #dfe4ec;border-radius:12px;padding:18px;margin:14px 0}}table{{width:100%;border-collapse:collapse}}th,td{{padding:9px;border-bottom:1px solid #e4e8ef;text-align:right}}th:first-child,td:first-child{{text-align:left}}.n{{background:#edf8f4}}.good{{color:#087f5b}}.bad{{color:#c33}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.box{{border:1px solid #dfe4ec;border-radius:9px;padding:12px}}small,p{{color:#667085}}code{{word-break:break-all}}@media(max-width:850px){{.grid{{grid-template-columns:1fr}}}}</style></head>
<body><main><h1>candidate_0000_rank0 流动性与波动性中性化</h1><p>日截面秩空间 OLS：因子秩 ~ 常数 + ln_amount秩 + 60日波动率秩；组合方向固定为 {report['direction']}，周频，Q5−Q1，成本30bps。</p>
<section class='card'><h2>核心对比</h2><table><thead><tr><th>阶段</th><th>处理</th><th>IC</th><th>ICIR</th><th>RankIC</th><th>RankICIR</th><th>毛多空</th><th>扣费多空</th><th>换手</th><th>流动性暴露</th><th>波动暴露</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
<section class='card'><h2>五分组累计收益</h2><div id='groups'></div></section>
<section class='card'><h2>解读</h2><div class='grid' id='takeaways'></div><p><code>{report['expression']}</code></p></section>
<script id='d' type='application/json'>{payload}</script><script>const R=JSON.parse(document.getElementById('d').textContent),L={{select:'样本期',valid:'验证期',test:'测试期'}};function p(x){{return (x*100).toFixed(2)+'%'}}let h='<table><tr><th>阶段</th><th>处理</th><th>Q1</th><th>Q2</th><th>Q3</th><th>Q4</th><th>Q5</th></tr>';for(const [k,v] of Object.entries(R.windows))for(const mode of ['raw','neutral'])h+=`<tr class='${{mode==='neutral'?'n':''}}'><td>${{L[k]}}</td><td>${{mode==='raw'?'原始':'中性化'}}</td>${{v[mode].group_total.map(x=>`<td>${{p(x)}}</td>`).join('')}}</tr>`;document.getElementById('groups').innerHTML=h+'</table>';let t='';for(const k of ['select','valid','test']){{const a=R.windows[k].raw.net_ls_total,b=R.windows[k].neutral.net_ls_total,d=b-a;t+=`<div class='box'><b>${{L[k]}}</b><h2 class='${{d>=0?'good':'bad'}}'>${{d>=0?'+':''}}${{p(d)}}</h2><small>扣费多空变化：${{p(a)}} → ${{p(b)}}</small></div>`}}document.getElementById('takeaways').innerHTML=t;</script></main></body></html>""", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--minute-parquet", default=str(MINUTE_PARQUET))
    ap.add_argument("--daily-parquet", default=str(ADJUSTED_CLOSE_PARQUET))
    ap.add_argument("--risk-exposures", default=str(RISK_EXPOSURES_PARQUET))
    ap.add_argument("--pit", default=str(ZZ500_PIT_PARQUET))
    ap.add_argument("--vol-window", type=int, default=60)
    ap.add_argument("--chunk-rows", type=int, default=4096)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    record = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    genome = ClimbMountainSkeletonGenome.from_dict(record["genome"])
    direction = int(record["fitness"].get("direction", 1))
    start, end = WINDOWS[0][2], WINDOWS[-1][3]
    device = "cpu" if args.cpu else "cuda"
    instruments, dates = load_pit_codes(args.pit, start, end), load_pit_dates(args.pit, start, end)
    context, meta = build_minute_slice(args.minute_parquet, start, end,
        fields=("open", "high", "low", "close"), instruments=instruments, dates=dates, device=device)
    close = load_daily_close_tensor(args.daily_parquet, meta["dates"], meta["instruments"], device=device)
    pool = load_pit_daily_mask(args.pit, meta["dates"], meta["instruments"], device=device) & torch.isfinite(close)
    fwd = tensor_rebalance_fwd_ret(close, meta["dates"], "week_end", 1)
    fwd = torch.where(pool, fwd, torch.full_like(fwd, float("nan")))
    vol = trailing_volatility(close, args.vol_window)
    styles, _industry, _levels = load_daily_exposures(args.risk_exposures, meta["dates"], meta["instruments"],
        continuous_columns=("ln_amount",), industry_column=None, device=device)
    liquidity = styles["ln_amount"]
    neutralizer = BatchedNeutralizer(fwd.shape, continuous=(liquidity, vol),
        rank_space=True, min_cross_section=30)
    raw = trailing_signal_mean(genome.evaluate(context, chunk_rows=args.chunk_rows), 5)
    raw = torch.where(pool, raw, torch.full_like(raw, float("nan")))
    neutral = torch.where(pool, neutralizer(raw), torch.full_like(raw, float("nan")))
    date_array = np.asarray(meta["dates"])
    windows = {}
    for key, _label, wstart, wend in WINDOWS:
        mask = torch.as_tensor((date_array >= wstart) & (date_array <= wend), device=device)
        windows[key] = {
            "raw": evaluate_period(raw, fwd, mask, direction, 5, 30.0),
            "neutral": evaluate_period(neutral, fwd, mask, direction, 5, 30.0),
            "raw_liq_corr": exposure_corr(raw, liquidity, fwd, mask),
            "raw_vol_corr": exposure_corr(raw, vol, fwd, mask),
            "neutral_liq_corr": exposure_corr(neutral, liquidity, fwd, mask),
            "neutral_vol_corr": exposure_corr(neutral, vol, fwd, mask),
        }
    report = {"generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "direction": direction, "expression": record["expression"],
        "definitions": {"liquidity": "PIT risk exposure ln_amount", "volatility": "60-day trailing daily-return volatility", "method": "daily cross-sectional rank-space OLS residual"},
        "windows": windows}
    destination = Path(args.out); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    write_html(report, destination)
    print(f"[climb-neutral] written -> {destination}")


if __name__ == "__main__":
    main()
