"""Build a standalone three-window report for Long-Short-Battle GP exports."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from min_gp.climb_mountain_html_report import evaluate_period
from min_gp.config import ADJUSTED_CLOSE_PARQUET, MINUTE_PARQUET, ZZ500_PIT_PARQUET
from min_gp.data import load_pit_codes, load_pit_daily_mask, load_pit_dates
from min_gp.evaluation.incremental import trailing_signal_mean
from min_gp.factors.handbook_skeleton import HandbookSkeletonGenome
from min_gp.label import tensor_rebalance_fwd_ret
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
            "genome": HandbookSkeletonGenome.from_dict(record["genome"]),
            "parquet": sidecar.with_suffix(""),
        })
    if not rows:
        raise SystemExit(f"no *.parquet.json candidates in {directory}")
    return rows


def number(value, digits=4):
    return "—" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def percent(value, digits=2):
    return "—" if value is None or not np.isfinite(value) else f"{value:.{digits}%}"


def build_html(report, destination: Path):
    labels = report["window_labels"]
    rows = []
    for candidate in report["candidates"]:
        for key, _label, _start, _end in WINDOWS:
            m = candidate["windows"][key]
            groups = "".join(f"<td>{percent(x)}</td>" for x in m["group_total"])
            rows.append(
                f"<tr><td>{candidate['id']}</td><td>{labels[key]}</td>"
                f"<td>{number(m['ic']['mean'])}</td><td>{number(m['ic']['ir_annual'], 2)}</td>"
                f"<td>{number(m['rank_ic']['mean'])}</td><td>{number(m['rank_ic']['ir_annual'], 2)}</td>"
                f"{groups}<td>{percent(m['gross_ls_total'])}</td><td>{percent(m['net_ls_total'])}</td>"
                f"<td>{percent(m['net_ls_annual'])}</td><td>{percent(m['turnover'], 1)}</td>"
                f"<td>{m['weeks']}</td></tr>"
            )
    payload = json.dumps(report, ensure_ascii=False).replace("</", "<\\/")
    destination.write_text(f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Long-Short Battle 12因子分期报告</title>
<style>
:root{{--bg:#f3f5f8;--card:#fff;--ink:#182033;--muted:#697287;--line:#dde3ec;--blue:#315efb}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1540px;margin:auto;padding:28px}}h1{{margin:0 0 5px}}h2{{font-size:19px}}.sub,.note{{color:var(--muted)}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin:15px 0;box-shadow:0 2px 12px #1720330a}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.metric{{border:1px solid var(--line);border-radius:9px;padding:12px}}.metric b{{display:block;font-size:18px}}
.scroll{{overflow:auto;max-height:650px}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}th,td{{padding:8px 9px;border-bottom:1px solid var(--line);text-align:right}}th{{position:sticky;top:0;background:#f8f9fc}}th:first-child,td:first-child{{text-align:left}}
select{{padding:8px;border:1px solid var(--line);border-radius:7px}}canvas{{width:100%;height:300px;border:1px solid var(--line);border-radius:8px}}code{{word-break:break-all;white-space:normal}}@media(max-width:850px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>Long-Short Battle：12个 Pareto 日频因子</h1>
<div class="sub">生成时间 {report['generated']} · 周频调仓 · 5日信号均值 · 五分组 · Q5−Q1 · 双边30bps成本</div>
<section class="card"><h2>数据切分与统计口径</h2><div class="grid">
<div class="metric">样本期<b>2018-01-02 ~ 2022-03-07</b></div><div class="metric">验证期<b>2022-03-08 ~ 2024-12-31</b></div><div class="metric">测试期<b>2025-01-02 ~ 2026-07-31</b></div></div>
<p class="note">IC 为周度截面 Pearson IC，RankIC 为周度截面 Spearman 相关；ICIR/RankICIR = mean/std×√52。方向沿用训练期锁定值，不在样本外重新选择。</p></section>
<section class="card"><h2>全部指标</h2><div class="scroll"><table><thead><tr><th>因子</th><th>阶段</th><th>IC</th><th>ICIR</th><th>RankIC</th><th>RankICIR</th><th>Q1</th><th>Q2</th><th>Q3</th><th>Q4</th><th>Q5</th><th>毛多空</th><th>扣费多空</th><th>扣费年化</th><th>换手</th><th>周数</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<section class="card"><h2>单因子详情</h2><select id="factor"></select><div id="detail"></div><h3>分组与多空累计净值</h3><canvas id="chart" width="1300" height="300"></canvas><p><code id="expr"></code></p></section>
<script id="data" type="application/json">{payload}</script><script>
const R=JSON.parse(document.getElementById('data').textContent),S=document.getElementById('factor');R.candidates.forEach((c,i)=>S.add(new Option(c.id,i)));
const pct=x=>Number.isFinite(x)?(x*100).toFixed(2)+'%':'—',num=(x,d=4)=>Number.isFinite(x)?x.toFixed(d):'—';
function draw(series){{const c=document.getElementById('chart'),g=c.getContext('2d'),W=c.width,H=c.height,p=40,all=series.flatMap(s=>s.v).filter(Number.isFinite);g.clearRect(0,0,W,H);if(!all.length)return;let lo=Math.min(...all),hi=Math.max(...all);if(hi===lo)hi=lo+1;g.strokeStyle='#dfe4ec';g.beginPath();g.moveTo(p,p);g.lineTo(p,H-p);g.lineTo(W-p,H-p);g.stroke();const cs=['#315efb','#08a36a','#e28a11','#8b5cf6','#e04b4b','#111827','#7c8aa5'];series.forEach((s,j)=>{{g.strokeStyle=cs[j%cs.length];g.lineWidth=j>4?2.7:1.5;g.beginPath();s.v.forEach((v,i)=>{{let x=p+i*(W-2*p)/Math.max(1,s.v.length-1),y=H-p-(v-lo)*(H-2*p)/(hi-lo);i?g.lineTo(x,y):g.moveTo(x,y)}});g.stroke();g.fillStyle=g.strokeStyle;g.fillText(s.n,p+j*120,17)}});g.fillStyle='#697287';g.fillText(hi.toFixed(2),3,p);g.fillText(lo.toFixed(2),3,H-p)}}
function render(){{const c=R.candidates[+S.value];let h='<table><thead><tr><th>阶段</th><th>IC</th><th>ICIR</th><th>RankIC</th><th>RankICIR</th><th>Q1</th><th>Q2</th><th>Q3</th><th>Q4</th><th>Q5</th><th>毛多空</th><th>扣费多空</th></tr></thead><tbody>';for(const [k,m] of Object.entries(c.windows))h+=`<tr><td>${{R.window_labels[k]}}</td><td>${{num(m.ic.mean)}}</td><td>${{num(m.ic.ir_annual,2)}}</td><td>${{num(m.rank_ic.mean)}}</td><td>${{num(m.rank_ic.ir_annual,2)}}</td>${{m.group_total.map(x=>`<td>${{pct(x)}}</td>`).join('')}}<td>${{pct(m.gross_ls_total)}}</td><td>${{pct(m.net_ls_total)}}</td></tr>`;document.getElementById('detail').innerHTML=h+'</tbody></table>';document.getElementById('expr').textContent=c.expression;const m=c.windows.test;draw(m.group_nav.map((v,i)=>({{n:'测试期 Q'+(i+1),v}})).concat([{{n:'测试期毛多空',v:m.gross_nav}},{{n:'测试期净多空',v:m.net_nav}}]))}}S.onchange=render;render();
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
    # GP exports already contain 2018-2024.  Only rebuild the genuinely unseen
    # test window, with a full year of warm-up for nested temporal operators.
    warmup_dates = [str(value) for value in dates if str(value) >= "2024-01-02"]
    test_context, test_meta = build_minute_slice(args.minute_parquet, "2024-01-02", end,
        fields=("open", "high", "low", "close", "volume"), instruments=instruments,
        dates=warmup_dates, device=device)
    close = load_daily_close_tensor(args.daily_parquet, dates, instruments, device=device)
    pool = load_pit_daily_mask(args.pit, dates, instruments, device=device) & torch.isfinite(close)
    fwd = tensor_rebalance_fwd_ret(close, dates, "week_end", 1)
    fwd = torch.where(pool, fwd, torch.full_like(fwd, float("nan")))
    date_array = np.asarray(dates)
    destination = Path(args.out)
    partial_path = destination.with_suffix(".partial.json")
    output = json.loads(partial_path.read_text(encoding="utf-8")) if partial_path.exists() else []
    completed_ids = {row["id"] for row in output}
    date_index = pd.Index([str(x) for x in dates], name="trade_date")
    instrument_index = pd.Index([str(x) for x in instruments], name="instrument")
    test_positions = np.array([date_index.get_loc(str(x)) for x in test_meta["dates"]])
    test_pool, test_fwd = pool[:, test_positions], fwd[:, test_positions]
    test_dates = np.asarray(test_meta["dates"])
    for pos, candidate in enumerate(candidates, 1):
        if candidate["id"] in completed_ids:
            print(f"[lsb-report] {pos}/{len(candidates)} {candidate['id']} cached", flush=True)
            continue
        print(f"[lsb-report] {pos}/{len(candidates)} {candidate['id']}", flush=True)
        cached = pd.read_parquet(candidate["parquet"], columns=["instrument", "trade_date", "factor"])
        cached["trade_date"] = cached["trade_date"].astype(str)
        frame = cached.pivot(index="instrument", columns="trade_date", values="factor")
        frame = frame.reindex(index=instrument_index, columns=date_index)
        cached_factor = torch.as_tensor(frame.to_numpy(copy=True), device=device, dtype=torch.float32)
        cached_factor = trailing_signal_mean(cached_factor, 5)
        cached_factor = torch.where(pool, cached_factor, torch.full_like(cached_factor, float("nan")))
        test_factor = trailing_signal_mean(
            candidate["genome"].evaluate(test_context, chunk_rows=args.chunk_rows), 5
        )
        test_factor = torch.where(test_pool, test_factor, torch.full_like(test_factor, float("nan")))
        metrics = {}
        for key, _label, wstart, wend in WINDOWS[:2]:
            mask = torch.as_tensor((date_array >= wstart) & (date_array <= wend), device=device)
            metrics[key] = evaluate_period(cached_factor, fwd, mask, candidate["direction"], args.groups, args.cost_bps)
        test_mask = torch.as_tensor((test_dates >= WINDOWS[2][2]) & (test_dates <= WINDOWS[2][3]), device=device)
        metrics["test"] = evaluate_period(test_factor, test_fwd, test_mask,
            candidate["direction"], args.groups, args.cost_bps)
        output.append({"id": candidate["id"], "direction": candidate["direction"],
            "expression": candidate["expression"], "train_fitness": candidate["fitness"], "windows": metrics})
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
        del cached, frame, cached_factor, test_factor
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    report = {"generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "window_labels": {k: label for k, label, _s, _e in WINDOWS}, "candidates": output}
    output.sort(key=lambda row: row["id"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    build_html(report, destination)
    destination.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    partial_path.unlink(missing_ok=True)
    print(f"[lsb-report] written -> {destination}", flush=True)


if __name__ == "__main__":
    main()
