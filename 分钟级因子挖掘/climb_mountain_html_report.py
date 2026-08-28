"""Build a standalone three-window HTML report for exported Climb-Mountain factors."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import torch

from min_gp.config import ADJUSTED_CLOSE_PARQUET, MINUTE_PARQUET, ZZ500_PIT_PARQUET
from min_gp.data import load_pit_codes, load_pit_daily_mask, load_pit_dates
from min_gp.evaluation.incremental import trailing_signal_mean
from min_gp.factors.climb_skeleton import ClimbMountainSkeletonGenome
from min_gp.label import tensor_rebalance_fwd_ret
from min_gp.numeric.ranking import cross_section_rank
from min_gp.report_candidates import _pearson, quantile_curves, series_stats
from min_gp.spectral_data import build_minute_slice, load_daily_close_tensor


WINDOWS = (
    ("select", "样本期", "2018-01-02", "2022-03-07"),
    ("valid", "验证期", "2022-03-08", "2024-12-31"),
    ("test", "测试期", "2025-01-02", "2026-07-31"),
)


def load_candidates(directory: Path):
    rows = []
    for sidecar in sorted(directory.glob("*.parquet.json")):
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        rows.append({
            "id": sidecar.name.removesuffix(".parquet.json"),
            "direction": int(record["fitness"].get("direction", 1)),
            "fitness": record["fitness"],
            "expression": record["expression"],
            "genome": ClimbMountainSkeletonGenome.from_dict(record["genome"]),
        })
    if not rows:
        raise SystemExit(f"no *.parquet.json candidates in {directory}")
    return rows


def total_return(values):
    return float(np.prod(1.0 + values) - 1.0) if values.size else float("nan")


def annualized_return(values, periods=52):
    if not values.size:
        return float("nan")
    wealth = float(np.prod(1.0 + values))
    return wealth ** (periods / len(values)) - 1.0 if wealth > 0 else float("nan")


def evaluate_period(factor, fwd, date_mask, direction, groups, cost_bps):
    scoped_fwd = torch.where(date_mask.unsqueeze(0), fwd, torch.full_like(fwd, float("nan")))
    scoped_factor = torch.where(
        date_mask.unsqueeze(0), factor, torch.full_like(factor, float("nan"))
    )
    valid = torch.isfinite(scoped_factor) & torch.isfinite(scoped_fwd)
    rank_y = cross_section_rank(scoped_fwd)
    ic = (_pearson(scoped_factor, scoped_fwd, valid) * direction).cpu().numpy()
    rank_ic = (
        _pearson(cross_section_rank(scoped_factor), rank_y, valid) * direction
    ).cpu().numpy()
    grouped, gross, net, turnover, days = quantile_curves(
        scoped_factor, scoped_fwd, direction, groups, cost_bps
    )
    ic_stats = series_stats(ic, 52)
    rank_stats = series_stats(rank_ic, 52)
    group_total = [total_return(values) for values in grouped]
    return {
        "ic": ic_stats,
        "rank_ic": rank_stats,
        "coverage": float(valid.sum() / torch.isfinite(scoped_fwd).sum().clamp(min=1)),
        "weeks": len(days),
        "group_total": group_total,
        "group_nav": [np.cumprod(1.0 + x).tolist() for x in grouped],
        "gross_ls_total": total_return(gross),
        "gross_ls_annual": annualized_return(gross),
        "net_ls_total": total_return(net),
        "net_ls_annual": annualized_return(net),
        "net_ls_mean": float(net.mean()) if net.size else float("nan"),
        "turnover": float(turnover.mean()) if turnover.size else float("nan"),
        "gross_nav": np.cumprod(1.0 + gross).tolist(),
        "net_nav": np.cumprod(1.0 + net).tolist(),
    }


def fmt(value, percent=False, digits=4):
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{digits}%}" if percent else f"{value:.{digits}f}"


def build_html(report, destination: Path):
    labels = {key: label for key, label, _start, _end in WINDOWS}
    ranges = {key: f"{start} ~ {end}" for key, _label, start, end in WINDOWS}
    summary_rows = []
    for candidate in report["candidates"]:
        for key, _label, _start, _end in WINDOWS:
            m = candidate["windows"][key]
            summary_rows.append(
                f"<tr><td>{candidate['id']}</td><td>{labels[key]}</td>"
                f"<td>{fmt(m['ic']['mean'])}</td><td>{fmt(m['ic']['ir_annual'], digits=2)}</td>"
                f"<td>{fmt(m['rank_ic']['mean'])}</td><td>{fmt(m['rank_ic']['ir_annual'], digits=2)}</td>"
                f"<td>{fmt(m['gross_ls_total'], True, 2)}</td>"
                f"<td>{fmt(m['net_ls_total'], True, 2)}</td>"
                f"<td>{fmt(m['net_ls_annual'], True, 2)}</td>"
                f"<td>{fmt(m['turnover'], True, 1)}</td><td>{m['weeks']}</td></tr>"
            )
    payload = json.dumps(report, ensure_ascii=False).replace("</", "<\\/")
    destination.write_text(f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Climb Mountain 13因子分期表现</title>
<style>
:root{{--bg:#f4f6f9;--card:#fff;--ink:#172033;--muted:#687187;--line:#dfe4ec;--blue:#315efb;--green:#079669;--red:#d44}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1480px;margin:auto;padding:28px}} h1{{margin:0 0 6px;font-size:28px}} h2{{margin:0 0 14px;font-size:19px}}
.sub{{color:var(--muted);margin-bottom:22px}} .card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin:14px 0;box-shadow:0 2px 10px #1720330a}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}} .metric{{padding:12px;border:1px solid var(--line);border-radius:9px}} .metric b{{display:block;font-size:20px}}
.scroll{{overflow:auto;max-height:620px}} table{{border-collapse:collapse;width:100%;white-space:nowrap}} th,td{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right}} th{{position:sticky;top:0;background:#f8f9fc;z-index:1}} th:first-child,td:first-child{{text-align:left}}
select{{padding:8px 12px;border:1px solid var(--line);border-radius:7px;background:white}} .charts{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} canvas{{width:100%;height:280px;border:1px solid var(--line);border-radius:8px}}
.note{{font-size:13px;color:var(--muted)}} code{{white-space:normal;word-break:break-all}} @media(max-width:900px){{.grid,.charts{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Climb Mountain：13个日频因子分期表现</h1>
<div class="sub">生成时间 {report['generated']} · 周频调仓 · 5日信号均值 · Q5−Q1 · 双边30bps成本口径</div>
<section class="card"><h2>切分口径</h2><div class="grid">
<div class="metric"><span>样本期</span><b>{ranges['select']}</b><small>GP walk-forward选择涉及区间</small></div>
<div class="metric"><span>验证期</span><b>{ranges['valid']}</b><small>未进入GP目标的时间块</small></div>
<div class="metric"><span>测试期</span><b>{ranges['test']}</b><small>独立样本外，方向沿用训练期</small></div>
</div><p class="note">IC 是截面 Pearson IC（你原文的“OC”暂按 IC 处理）；RankIC 是截面 Spearman 秩相关。ICIR/RankICIR 为周序列 mean/std×√52。所有指标都按训练期锁定方向后报告。</p></section>
<section class="card"><h2>全部指标</h2><div class="scroll"><table><thead><tr><th>因子</th><th>阶段</th><th>IC</th><th>ICIR</th><th>RankIC</th><th>RankICIR</th><th>毛多空</th><th>扣费多空</th><th>扣费年化</th><th>换手</th><th>周数</th></tr></thead><tbody>{''.join(summary_rows)}</tbody></table></div></section>
<section class="card"><h2>单因子详情</h2><label>选择因子：<select id="factor"></select></label><div id="detail"></div><div class="charts"><div><h3>五分组累计净值</h3><canvas id="groups" width="680" height="280"></canvas></div><div><h3>多空累计净值</h3><canvas id="ls" width="680" height="280"></canvas></div></div><p class="note"><code id="expr"></code></p></section>
<script id="data" type="application/json">{payload}</script>
<script>
const R=JSON.parse(document.getElementById('data').textContent), sel=document.getElementById('factor');
R.candidates.forEach((c,i)=>sel.add(new Option(c.id,i)));
function pct(x){{return Number.isFinite(x)?(x*100).toFixed(2)+'%':'—'}} function num(x,d=4){{return Number.isFinite(x)?x.toFixed(d):'—'}}
function draw(id,series,colors){{const c=document.getElementById(id),x=c.getContext('2d'),W=c.width,H=c.height,p=34;x.clearRect(0,0,W,H);const all=series.flatMap(s=>s.v).filter(Number.isFinite);if(!all.length)return;let lo=Math.min(...all),hi=Math.max(...all);if(hi===lo)hi=lo+1; x.strokeStyle='#dfe4ec';x.beginPath();x.moveTo(p,H-p);x.lineTo(W-p,H-p);x.moveTo(p,p);x.lineTo(p,H-p);x.stroke(); series.forEach((s,j)=>{{x.strokeStyle=colors[j%colors.length];x.lineWidth=2;x.beginPath();s.v.forEach((v,i)=>{{const xx=p+i*(W-2*p)/Math.max(1,s.v.length-1),yy=H-p-(v-lo)*(H-2*p)/(hi-lo);i?x.lineTo(xx,yy):x.moveTo(xx,yy)}});x.stroke();x.fillStyle=x.strokeStyle;x.fillText(s.n,p+90*j,16)}});x.fillStyle='#687187';x.fillText(hi.toFixed(2),2,p);x.fillText(lo.toFixed(2),2,H-p)}}
function render(){{const c=R.candidates[+sel.value];let h='<table><thead><tr><th>阶段</th><th>IC</th><th>ICIR</th><th>RankIC</th><th>RankICIR</th><th>Q1</th><th>Q2</th><th>Q3</th><th>Q4</th><th>Q5</th><th>毛多空</th><th>扣费多空</th></tr></thead><tbody>';Object.entries(c.windows).forEach(([k,m])=>{{h+=`<tr><td>${{R.window_labels[k]}}</td><td>${{num(m.ic.mean)}}</td><td>${{num(m.ic.ir_annual,2)}}</td><td>${{num(m.rank_ic.mean)}}</td><td>${{num(m.rank_ic.ir_annual,2)}}</td>${{m.group_total.map(v=>`<td>${{pct(v)}}</td>`).join('')}}<td>${{pct(m.gross_ls_total)}}</td><td>${{pct(m.net_ls_total)}}</td></tr>`}});document.getElementById('detail').innerHTML=h+'</tbody></table>';document.getElementById('expr').textContent=c.expression;const keys=['select','valid','test'],cols=['#315efb','#079669','#d97706','#8b5cf6','#d44'];let gs=[];keys.forEach((k,wi)=>c.windows[k].group_nav.forEach((v,qi)=>gs.push({{n:R.window_labels[k]+' Q'+(qi+1),v}})));draw('groups',gs,cols);draw('ls',keys.flatMap(k=>[{{n:R.window_labels[k]+'毛',v:c.windows[k].gross_nav}},{{n:R.window_labels[k]+'净',v:c.windows[k].net_nav}}]),['#315efb','#8ba2ff','#079669','#6fd3b0','#d97706','#f3b65f'])}}
sel.onchange=render;render();
</script></main></body></html>""", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--minute-parquet", default=str(MINUTE_PARQUET))
    parser.add_argument("--daily-parquet", default=str(ADJUSTED_CLOSE_PARQUET))
    parser.add_argument("--pit", default=str(ZZ500_PIT_PARQUET))
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
    context, meta = build_minute_slice(
        args.minute_parquet, start, end, fields=("open", "high", "low", "close"),
        instruments=instruments, dates=dates, device=device,
    )
    close = load_daily_close_tensor(args.daily_parquet, meta["dates"], meta["instruments"], device=device)
    pool = load_pit_daily_mask(args.pit, meta["dates"], meta["instruments"], device=device) & torch.isfinite(close)
    fwd = tensor_rebalance_fwd_ret(close, meta["dates"], "week_end", 1)
    fwd = torch.where(pool, fwd, torch.full_like(fwd, float("nan")))
    date_array = np.asarray(meta["dates"])
    output = []
    for pos, candidate in enumerate(candidates, 1):
        print(f"[climb-report] {pos}/{len(candidates)} {candidate['id']}", flush=True)
        factor = candidate["genome"].evaluate(context, chunk_rows=args.chunk_rows)
        factor = trailing_signal_mean(factor, 5)
        factor = torch.where(pool, factor, torch.full_like(factor, float("nan")))
        windows = {}
        for key, _label, wstart, wend in WINDOWS:
            mask = torch.as_tensor((date_array >= wstart) & (date_array <= wend), device=device)
            windows[key] = evaluate_period(
                factor, fwd, mask, candidate["direction"], args.groups, args.cost_bps
            )
        output.append({
            "id": candidate["id"], "direction": candidate["direction"],
            "expression": candidate["expression"], "train_fitness": candidate["fitness"],
            "windows": windows,
        })
        del factor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    from datetime import datetime
    report = {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "window_labels": {key: label for key, label, _s, _e in WINDOWS},
        "candidates": output,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    build_html(report, destination)
    destination.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[climb-report] written -> {destination}")


if __name__ == "__main__":
    main()
