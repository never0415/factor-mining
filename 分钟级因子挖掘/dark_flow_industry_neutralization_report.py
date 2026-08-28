"""Industry-neutral performance report for Dark-Flow Pareto factors."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from min_gp.climb_mountain_html_report import evaluate_period
from min_gp.config import (
    ADJUSTED_CLOSE_PARQUET,
    INDUSTRY_VALUE_EXPOSURES_PARQUET,
    MINUTE_PARQUET,
    ZZ500_PIT_PARQUET,
)
from min_gp.data import load_pit_codes, load_pit_daily_mask, load_pit_dates
from min_gp.evaluation.incremental import trailing_signal_mean
from min_gp.evaluation.neutralize import BatchedNeutralizer
from min_gp.label import tensor_rebalance_fwd_ret
from min_gp.long_short_battle_html_report import WINDOWS, load_candidates
from min_gp.numeric.ranking import cross_section_rank
from min_gp.spectral_data import (
    build_minute_slice,
    load_daily_close_tensor,
    load_daily_exposures,
)


STAGE_LABELS = {"select": "样本期", "valid": "验证期", "test": "测试期"}


def _finite(value) -> bool:
    return value is not None and np.isfinite(value)


def _fmt(value, digits=4) -> str:
    return "—" if not _finite(value) else f"{value:.{digits}f}"


def _pct(value, digits=2) -> str:
    return "—" if not _finite(value) else f"{value:.{digits}%}"


def _industry_coverage(industry, pool, date_mask) -> float:
    eligible = pool & date_mask.unsqueeze(0)
    denominator = eligible.sum().clamp(min=1)
    return float((eligible & industry.ge(0)).sum() / denominator)


def _industry_rank_r2(ranked, industry, fwd, date_mask, levels) -> float:
    """Median daily R2 of tradable factor ranks on industry dummies."""
    valid = (
        torch.isfinite(ranked)
        & torch.isfinite(fwd)
        & industry.ge(0)
        & date_mask.unsqueeze(0)
    )
    target = torch.where(valid, ranked, torch.zeros_like(ranked))
    count = valid.sum(dim=0)
    mean = target.sum(dim=0) / count.clamp(min=1)
    centered = torch.where(valid, ranked - mean, torch.zeros_like(ranked))
    total_ss = centered.square().sum(dim=0)
    between_ss = torch.zeros_like(total_ss)
    for level in levels:
        member = valid & industry.eq(level)
        n_group = member.sum(dim=0)
        group_sum = torch.where(member, centered, torch.zeros_like(centered)).sum(dim=0)
        between_ss += group_sum.square() / n_group.clamp(min=1)
    r2 = between_ss / total_ss.clamp(min=1e-12)
    good = (count >= 30) & (total_ss > 1e-12) & date_mask
    values = r2[good]
    return float(values.median()) if values.numel() else float("nan")


def _build_html(report: dict, destination: Path) -> None:
    all_rows = []
    test_rows = []
    for candidate in report["candidates"]:
        for stage in ("select", "valid", "test"):
            window = candidate["windows"][stage]
            raw, neutral = window["raw"], window["neutral"]
            delta = neutral["net_ls_total"] - raw["net_ls_total"]
            all_rows.append(
                f"<tr><td>{candidate['id']}</td><td>{STAGE_LABELS[stage]}</td>"
                f"<td>{_pct(window['industry_coverage'], 1)}</td>"
                f"<td>{_fmt(raw['ic']['mean'])} → {_fmt(neutral['ic']['mean'])}</td>"
                f"<td>{_fmt(raw['ic']['ir_annual'], 2)} → {_fmt(neutral['ic']['ir_annual'], 2)}</td>"
                f"<td>{_fmt(raw['rank_ic']['mean'])} → {_fmt(neutral['rank_ic']['mean'])}</td>"
                f"<td>{_fmt(raw['rank_ic']['ir_annual'], 2)} → {_fmt(neutral['rank_ic']['ir_annual'], 2)}</td>"
                f"<td>{_pct(raw['net_ls_total'])} → {_pct(neutral['net_ls_total'])}</td>"
                f"<td class={'good' if delta >= 0 else 'bad'}>{_pct(delta)}</td>"
                f"<td>{_pct(raw['turnover'], 1)} → {_pct(neutral['turnover'], 1)}</td>"
                f"<td>{_pct(window['raw_industry_rank_r2'], 2)} → {_pct(window['neutral_industry_rank_r2'], 2)}</td></tr>"
            )
        test = candidate["windows"]["test"]
        valid = candidate["windows"]["valid"]
        delta_test = test["neutral"]["net_ls_total"] - test["raw"]["net_ls_total"]
        delta_valid = valid["neutral"]["net_ls_total"] - valid["raw"]["net_ls_total"]
        tag = "稳健改善" if candidate["robust_improvement"] else ("测试期改善" if delta_test > 0 else "未改善")
        test_rows.append(
            f"<tr><td>{candidate['id']}</td><td class={'good' if delta_test > 0 else 'bad'}>{tag}</td>"
            f"<td>{_pct(test['raw']['net_ls_total'])}</td><td>{_pct(test['neutral']['net_ls_total'])}</td>"
            f"<td>{_pct(delta_test)}</td><td>{_pct(delta_valid)}</td>"
            f"<td>{_fmt(test['neutral']['rank_ic']['mean'])}</td>"
            f"<td>{_fmt(test['neutral']['rank_ic']['ir_annual'], 2)}</td>"
            f"<td>{_pct(test['neutral_industry_rank_r2'], 2)}</td></tr>"
        )
    payload = json.dumps(report, ensure_ascii=False).replace("</", "<\\/")
    destination.write_text(
        f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Dark Flow 10因子行业中性化</title>
<style>
:root{{--bg:#f3f5f8;--card:#fff;--ink:#172033;--muted:#687187;--line:#dde3ec;--blue:#315efb}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1720px;margin:auto;padding:28px}}h1{{margin:0 0 6px}}h2{{font-size:19px}}.sub,.note{{color:var(--muted)}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin:15px 0;box-shadow:0 2px 12px #1720330a}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.metric{{border:1px solid var(--line);border-radius:9px;padding:12px}}.metric b{{display:block;font-size:18px}}
.scroll{{overflow:auto;max-height:680px}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}th,td{{padding:8px 9px;border-bottom:1px solid var(--line);text-align:right}}th{{position:sticky;top:0;background:#f8f9fc;z-index:1}}th:first-child,td:first-child{{text-align:left}}
.good{{color:#087f5b;font-weight:600}}.bad{{color:#c33}}select{{padding:8px 12px;border:1px solid var(--line);border-radius:7px;background:#fff}}canvas{{width:100%;height:330px;border:1px solid var(--line);border-radius:8px}}code{{word-break:break-all;white-space:normal}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>Dark Flow：10个 Pareto 因子行业中性化</h1>
<div class="sub">生成时间 {report['generated']} · 逐日申万一级点时行业 · 截面排名对行业哑变量回归取残差 · 周频调仓 · 5日信号均值 · Q5−Q1 · 双边30bps成本</div>
<section class="card"><h2>口径</h2><div class="grid"><div class="metric">样本期<b>2018-01-02～2022-03-07</b></div><div class="metric">验证期<b>2022-03-08～2024-12-31</b></div><div class="metric">测试期<b>2025-01-02～2026-07-31</b></div><div class="metric">行业分类<b>{report['industry_level_count']} 个申万一级桶</b></div></div>
<p class="note">原始与中性化指标都限制在行业有效的完全相同样本上。方向沿用训练期并锁定。行业暴露用每日“信号截面排名 ~ 行业哑变量”的 R² 中位数衡量。稳健改善定义：验证期和测试期扣费多空都优于匹配原始值，且中性化后测试期 RankIC 与扣费多空均为正。</p></section>
<section class="card"><h2>测试期结论</h2><div class="scroll"><table><thead><tr><th>因子</th><th>判断</th><th>原始净多空</th><th>行业中性净多空</th><th>测试期变化</th><th>验证期变化</th><th>中性RankIC</th><th>中性RankICIR</th><th>剩余行业R²</th></tr></thead><tbody>{''.join(test_rows)}</tbody></table></div></section>
<section class="card"><h2>三阶段完整对比</h2><div class="scroll"><table><thead><tr><th>因子</th><th>阶段</th><th>行业覆盖</th><th>IC 原→中</th><th>ICIR 原→中</th><th>RankIC 原→中</th><th>RankICIR 原→中</th><th>扣费多空 原→中</th><th>净收益变化</th><th>换手 原→中</th><th>行业R² 原→中</th></tr></thead><tbody>{''.join(all_rows)}</tbody></table></div></section>
<section class="card"><h2>单因子明细与曲线</h2><label>选择因子：<select id="factor"></select></label><div id="detail"></div><h3>行业中性化后：三阶段五分组收益曲线（同一张图）</h3><canvas id="groups" width="1500" height="330"></canvas><h3>原始与行业中性化：三阶段扣费多空曲线</h3><canvas id="ls" width="1500" height="330"></canvas><p><code id="expr"></code></p></section>
<script id="data" type="application/json">{payload}</script><script>
const R=JSON.parse(document.getElementById('data').textContent),S=document.getElementById('factor');R.candidates.forEach((c,i)=>S.add(new Option(c.id,i)));
const pct=x=>Number.isFinite(x)?(x*100).toFixed(2)+'%':'—',num=(x,d=4)=>Number.isFinite(x)?x.toFixed(d):'—';
function draw(id,series){{const c=document.getElementById(id),g=c.getContext('2d'),W=c.width,H=c.height,p=45,all=series.flatMap(s=>s.v).filter(Number.isFinite);g.clearRect(0,0,W,H);if(!all.length)return;let lo=Math.min(...all),hi=Math.max(...all);if(hi===lo)hi=lo+1;g.strokeStyle='#dfe4ec';g.beginPath();g.moveTo(p,p);g.lineTo(p,H-p);g.lineTo(W-p,H-p);g.stroke();const cs=['#315efb','#08a36a','#e28a11','#8b5cf6','#e04b4b'];series.forEach((s,j)=>{{g.strokeStyle=cs[s.color%cs.length];g.setLineDash(s.dash||[]);g.lineWidth=1.8;g.beginPath();s.v.forEach((v,i)=>{{const x=p+i*(W-2*p)/Math.max(1,s.v.length-1),y=H-p-(v-lo)*(H-2*p)/(hi-lo);i?g.lineTo(x,y):g.moveTo(x,y)}});g.stroke();g.setLineDash([]);g.fillStyle=g.strokeStyle;g.fillText(s.n,p+(j%5)*220,16+Math.floor(j/5)*15)}});g.fillStyle='#687187';g.fillText(hi.toFixed(2),3,p);g.fillText(lo.toFixed(2),3,H-p)}}
function render(){{const c=R.candidates[+S.value];let h='<table><thead><tr><th>阶段</th><th>中性IC</th><th>中性ICIR</th><th>中性RankIC</th><th>中性RankICIR</th><th>Q1</th><th>Q2</th><th>Q3</th><th>Q4</th><th>Q5</th><th>毛多空</th><th>净多空</th><th>换手</th></tr></thead><tbody>';['select','valid','test'].forEach(k=>{{const m=c.windows[k].neutral;h+=`<tr><td>${{R.window_labels[k]}}</td><td>${{num(m.ic.mean)}}</td><td>${{num(m.ic.ir_annual,2)}}</td><td>${{num(m.rank_ic.mean)}}</td><td>${{num(m.rank_ic.ir_annual,2)}}</td>${{m.group_total.map(x=>`<td>${{pct(x)}}</td>`).join('')}}<td>${{pct(m.gross_ls_total)}}</td><td>${{pct(m.net_ls_total)}}</td><td>${{pct(m.turnover)}}</td></tr>`}});document.getElementById('detail').innerHTML=h+'</tbody></table>';document.getElementById('expr').textContent=c.expression;const dash=[[],[8,4],[2,4]],gs=[];['select','valid','test'].forEach((k,si)=>c.windows[k].neutral.group_nav.forEach((v,qi)=>gs.push({{n:R.window_labels[k]+' Q'+(qi+1),v:[1,...v],color:qi,dash:dash[si]}})));draw('groups',gs);const ls=[];['select','valid','test'].forEach((k,si)=>{{ls.push({{n:R.window_labels[k]+' 原始',v:[1,...c.windows[k].raw.net_nav],color:si,dash:[7,4]}});ls.push({{n:R.window_labels[k]+' 行业中性',v:[1,...c.windows[k].neutral.net_nav],color:si,dash:[]}})}});draw('ls',ls)}}S.onchange=render;render();
</script></main></body></html>""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--minute-parquet", default=str(MINUTE_PARQUET))
    parser.add_argument("--daily-parquet", default=str(ADJUSTED_CLOSE_PARQUET))
    parser.add_argument("--pit", default=str(ZZ500_PIT_PARQUET))
    parser.add_argument("--industry-exposures", default=str(INDUSTRY_VALUE_EXPOSURES_PARQUET))
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--cost-bps", type=float, default=30.0)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = "cpu" if args.cpu else "cuda"
    candidates = load_candidates(Path(args.factor_dir))
    start, end = WINDOWS[0][2], WINDOWS[-1][3]
    instruments = load_pit_codes(args.pit, start, end)
    dates = load_pit_dates(args.pit, start, end)
    date_array = np.asarray(dates)
    date_index = pd.Index([str(value) for value in dates], name="trade_date")
    instrument_index = pd.Index([str(value) for value in instruments], name="instrument")

    print("[industry-neutral] loading daily data and point-in-time industry", flush=True)
    close = load_daily_close_tensor(args.daily_parquet, dates, instruments, device=device)
    pool = load_pit_daily_mask(args.pit, dates, instruments, device=device) & torch.isfinite(close)
    fwd = tensor_rebalance_fwd_ret(close, dates, "week_end", 1)
    fwd = torch.where(pool, fwd, torch.full_like(fwd, float("nan")))
    _styles, industry, industry_names = load_daily_exposures(
        args.industry_exposures,
        dates,
        instruments,
        continuous_columns=(),
        industry_column="sw_level1",
        device=device,
    )
    levels = torch.unique(industry[industry.ge(0)])
    matched_pool = pool & industry.ge(0)
    neutralizer = BatchedNeutralizer(
        pool.shape,
        industry=industry,
        rank_space=True,
        min_cross_section=30,
    )

    print("[industry-neutral] loading 2024-2026 minute context once", flush=True)
    warmup_dates = [str(value) for value in dates if str(value) >= "2024-01-02"]
    context, context_meta = build_minute_slice(
        args.minute_parquet,
        "2024-01-02",
        end,
        fields=("open", "high", "low", "close", "volume"),
        instruments=instruments,
        dates=warmup_dates,
        device=device,
    )
    holdout_dates = np.asarray(context_meta["dates"])
    holdout_use = np.flatnonzero(holdout_dates >= WINDOWS[2][2])
    holdout_positions = np.array(
        [date_index.get_loc(str(holdout_dates[index])) for index in holdout_use]
    )

    destination = Path(args.out)
    partial = destination.with_suffix(".partial.json")
    output = json.loads(partial.read_text(encoding="utf-8")) if partial.exists() else []
    completed = {row["id"] for row in output}
    for position, candidate in enumerate(candidates, 1):
        if candidate["id"] in completed:
            print(f"[industry-neutral] {position}/{len(candidates)} {candidate['id']} cached", flush=True)
            continue
        print(f"[industry-neutral] {position}/{len(candidates)} {candidate['id']}", flush=True)
        cached = pd.read_parquet(
            candidate["parquet"], columns=["instrument", "trade_date", "factor"]
        )
        cached["trade_date"] = cached["trade_date"].astype(str)
        frame = cached.pivot(index="instrument", columns="trade_date", values="factor")
        frame = frame.reindex(index=instrument_index, columns=date_index)
        raw = torch.as_tensor(frame.to_numpy(copy=True), device=device, dtype=torch.float32)
        raw = trailing_signal_mean(raw, 5)
        holdout = trailing_signal_mean(
            candidate["genome"].evaluate(context, chunk_rows=args.chunk_rows), 5
        )
        raw[:, holdout_positions] = holdout[:, holdout_use]
        raw = torch.where(matched_pool, raw, torch.full_like(raw, float("nan")))
        neutral = neutralizer(raw)
        neutral = torch.where(matched_pool, neutral, torch.full_like(neutral, float("nan")))
        raw_rank = cross_section_rank(raw.float())
        neutral_rank = cross_section_rank(neutral.float())

        windows = {}
        for stage, _label, window_start, window_end in WINDOWS:
            date_mask = torch.as_tensor(
                (date_array >= window_start) & (date_array <= window_end), device=device
            )
            windows[stage] = {
                "industry_coverage": _industry_coverage(industry, pool, date_mask),
                "raw_industry_rank_r2": _industry_rank_r2(
                    raw_rank, industry, fwd, date_mask, levels
                ),
                "neutral_industry_rank_r2": _industry_rank_r2(
                    neutral_rank, industry, fwd, date_mask, levels
                ),
                "raw": evaluate_period(
                    raw, fwd, date_mask, candidate["direction"], args.groups, args.cost_bps
                ),
                "neutral": evaluate_period(
                    neutral, fwd, date_mask, candidate["direction"], args.groups, args.cost_bps
                ),
            }
        delta_valid = (
            windows["valid"]["neutral"]["net_ls_total"]
            - windows["valid"]["raw"]["net_ls_total"]
        )
        delta_test = (
            windows["test"]["neutral"]["net_ls_total"]
            - windows["test"]["raw"]["net_ls_total"]
        )
        robust = bool(
            delta_valid > 0
            and delta_test > 0
            and windows["test"]["neutral"]["rank_ic"]["mean"] > 0
            and windows["test"]["neutral"]["net_ls_total"] > 0
        )
        output.append(
            {
                "id": candidate["id"],
                "direction": candidate["direction"],
                "expression": candidate["expression"],
                "robust_improvement": robust,
                "windows": windows,
            }
        )
        output.sort(key=lambda row: row["id"])
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
        del cached, frame, raw, holdout, neutral, raw_rank, neutral_rank
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "method": "daily rank-space OLS residual on point-in-time SW level-1 dummies; matched raw universe",
        "window_labels": STAGE_LABELS,
        "industry_level_count": len(industry_names),
        "industry_levels": industry_names,
        "candidates": sorted(output, key=lambda row: row["id"]),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    _build_html(report, destination)
    destination.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False), encoding="utf-8"
    )
    partial.unlink(missing_ok=True)
    print(f"[industry-neutral] written -> {destination}", flush=True)


if __name__ == "__main__":
    main()
