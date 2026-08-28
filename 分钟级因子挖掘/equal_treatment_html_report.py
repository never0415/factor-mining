"""Five-window performance report for Equal-Treatment Pareto exports.

The GP parquet exports end at 2024-12-31.  This report restores each genome
from its sidecar and evaluates it on genuinely unseen 2025-2026 minute data,
using 2024 only as operator warm-up.  Direction is locked by the GP fitness
record; it is never re-selected on validation or holdout data.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from min_gp.climb_mountain_html_report import evaluate_period
from min_gp.config import ADJUSTED_CLOSE_PARQUET, MINUTE_PARQUET, ZZ500_PIT_PARQUET
from min_gp.data import load_pit_codes, load_pit_daily_mask, load_pit_dates
from min_gp.evaluation.incremental import trailing_signal_mean
from min_gp.label import tensor_rebalance_fwd_ret
from min_gp.long_short_battle_html_report import load_candidates
from min_gp.spectral_data import build_minute_slice, load_daily_close_tensor


WINDOWS = (
    ("select", "样本期", "2018-01-02", "2022-03-07"),
    ("valid", "验证期", "2022-03-08", "2024-12-31"),
    ("test", "测试期（2025—2026）", "2025-01-02", "2026-07-31"),
    ("y2025", "2025年", "2025-01-02", "2025-12-31"),
    ("y2026", "2026年截至7月", "2026-01-02", "2026-07-31"),
)


def _finite(value):
    return value is not None and np.isfinite(value)


def _number(value, digits=4):
    return "—" if not _finite(value) else f"{value:.{digits}f}"


def _percent(value, digits=2):
    return "—" if not _finite(value) else f"{value:.{digits}%}"


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _aggregate(candidates, key):
    metrics = [candidate["windows"][key] for candidate in candidates]
    net = np.asarray([item["net_ls_total"] for item in metrics], dtype=float)
    ic = np.asarray([item["ic"]["mean"] for item in metrics], dtype=float)
    rank_ic = np.asarray([item["rank_ic"]["mean"] for item in metrics], dtype=float)
    finite_net = net[np.isfinite(net)]
    return {
        "positive_ic": int(np.sum(ic > 0)),
        "positive_rank_ic": int(np.sum(rank_ic > 0)),
        "positive_net": int(np.sum(net > 0)),
        "median_net": float(np.median(finite_net)) if finite_net.size else float("nan"),
    }


def _best(candidates, key, path):
    def metric(candidate):
        value = candidate["windows"][key]
        for part in path:
            value = value[part]
        return value if _finite(value) else -float("inf")

    winner = max(candidates, key=metric)
    value = winner["windows"][key]
    for part in path:
        value = value[part]
    return {"id": winner["id"], "value": value}


def build_html(report, destination: Path):
    labels = report["window_labels"]
    summary_rows = []
    for candidate in report["candidates"]:
        for key, _label, _start, _end in WINDOWS:
            item = candidate["windows"][key]
            groups = "".join(f"<td>{_percent(value)}</td>" for value in item["group_total"])
            summary_rows.append(
                f"<tr><td>{candidate['id']}</td><td>{labels[key]}</td>"
                f"<td>{candidate['direction']:+d}</td>"
                f"<td>{_number(item['ic']['mean'])}</td><td>{_number(item['ic']['ir_annual'], 2)}</td>"
                f"<td>{_number(item['rank_ic']['mean'])}</td><td>{_number(item['rank_ic']['ir_annual'], 2)}</td>"
                f"{groups}<td>{_percent(item['gross_ls_total'])}</td>"
                f"<td>{_percent(item['net_ls_total'])}</td><td>{_percent(item['net_ls_annual'])}</td>"
                f"<td>{_percent(item['turnover'], 1)}</td><td>{item['weeks']}</td></tr>"
            )

    year_rows = []
    for candidate in report["candidates"]:
        y25, y26, test = (
            candidate["windows"]["y2025"], candidate["windows"]["y2026"],
            candidate["windows"]["test"],
        )
        year_rows.append(
            f"<tr><td>{candidate['id']}</td>"
            f"<td>{_number(y25['ic']['mean'])}</td><td>{_number(y25['rank_ic']['mean'])}</td>"
            f"<td>{_percent(y25['net_ls_total'])}</td>"
            f"<td>{_number(y26['ic']['mean'])}</td><td>{_number(y26['rank_ic']['mean'])}</td>"
            f"<td>{_percent(y26['net_ls_total'])}</td><td>{_percent(test['net_ls_total'])}</td></tr>"
        )

    test = report["analysis"]["test"]
    best_net = report["best"]["test_net"]
    best_ic = report["best"]["test_ic"]
    best_rank = report["best"]["test_rank_ic"]
    duplicate_note = "；".join("、".join(group) for group in report["duplicate_groups"])
    if not duplicate_note:
        duplicate_note = "无"
    payload = json.dumps(_json_safe(report), ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    destination.write_text(f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Equal Treatment 10个Pareto因子绩效</title>
<style>
:root{{--bg:#f3f5f8;--card:#fff;--ink:#172033;--muted:#697287;--line:#dde3ec;--blue:#315efb;--green:#079669;--orange:#d97706}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1580px;margin:auto;padding:28px}}h1{{margin:0 0 5px}}h2{{font-size:19px;margin:0 0 13px}}h3{{font-size:16px}}
.sub,.note{{color:var(--muted)}}.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin:15px 0;box-shadow:0 2px 12px #1720330a}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.metric{{border:1px solid var(--line);border-radius:9px;padding:12px}}.metric b{{display:block;font-size:20px}}
.scroll{{overflow:auto;max-height:680px}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}th,td{{padding:8px 9px;border-bottom:1px solid var(--line);text-align:right}}th{{position:sticky;top:0;background:#f8f9fc;z-index:1}}th:first-child,td:first-child{{text-align:left}}
select{{padding:8px;border:1px solid var(--line);border-radius:7px;background:white;margin-right:10px}}canvas{{width:100%;height:310px;border:1px solid var(--line);border-radius:8px}}code{{word-break:break-all;white-space:normal}}.good{{color:var(--green)}}.warn{{color:var(--orange)}}
@media(max-width:900px){{.grid{{grid-template-columns:1fr 1fr}}}}@media(max-width:600px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>Equal Treatment：10个 Pareto 因子绩效分析</h1>
<div class="sub">生成时间 {report['generated']} · 周频调仓 · 5日信号均值 · 五分组 · Q5−Q1 · 双边30bps成本</div>
<section class="card"><h2>独立测试期概览（2025-01-02—2026-07-31）</h2><div class="grid">
<div class="metric">测试期IC为正<b>{test['positive_ic']}/10</b></div>
<div class="metric">测试期RankIC为正<b>{test['positive_rank_ic']}/10</b></div>
<div class="metric">扣费多空为正<b>{test['positive_net']}/10</b></div>
<div class="metric">扣费多空中位数<b>{_percent(test['median_net'])}</b></div>
<div class="metric">最高测试期IC<b>{best_ic['id']}</b><span>{_number(best_ic['value'])}</span></div>
<div class="metric">最高测试期RankIC<b>{best_rank['id']}</b><span>{_number(best_rank['value'])}</span></div>
<div class="metric">最高扣费多空<b>{best_net['id']}</b><span>{_percent(best_net['value'])}</span></div>
<div class="metric">唯一公式<b>{report['unique_expressions']}/10</b></div></div>
<p class="note">样本期锁定方向，验证期、2025年和2026年均不重新选方向。重复公式组：{duplicate_note}。</p></section>
<section class="card"><h2>阶段口径</h2><div class="grid">
<div class="metric">样本期<b>2018-01-02—2022-03-07</b></div><div class="metric">验证期<b>2022-03-08—2024-12-31</b></div>
<div class="metric">测试期<b>2025-01-02—2026-07-31</b></div><div class="metric">年度拆分<b>2025全年 / 2026截至7月</b></div></div>
<p class="note">IC为周度截面Pearson相关，RankIC为Spearman秩相关；ICIR/RankICIR=mean/std×√52。因子在周末形成，使用未来一周收益；最后一个无完整未来收益的周自动剔除。</p></section>
<section class="card"><h2>2025与2026年度对比</h2><div class="scroll"><table><thead><tr><th>因子</th><th>2025 IC</th><th>2025 RankIC</th><th>2025扣费多空</th><th>2026 IC</th><th>2026 RankIC</th><th>2026扣费多空</th><th>25—26合计扣费多空</th></tr></thead><tbody>{''.join(year_rows)}</tbody></table></div></section>
<section class="card"><h2>完整指标</h2><div class="scroll"><table><thead><tr><th>因子</th><th>阶段</th><th>方向</th><th>IC</th><th>ICIR</th><th>RankIC</th><th>RankICIR</th><th>Q1</th><th>Q2</th><th>Q3</th><th>Q4</th><th>Q5</th><th>毛多空</th><th>扣费多空</th><th>扣费年化</th><th>换手</th><th>周数</th></tr></thead><tbody>{''.join(summary_rows)}</tbody></table></div></section>
<section class="card"><h2>单因子详情</h2><label>因子 <select id="factor"></select></label><label>阶段 <select id="window"></select></label><div id="detail"></div><h3>五分组及多空累计净值</h3><canvas id="chart" width="1350" height="310"></canvas><p><code id="expr"></code></p></section>
<script id="data" type="application/json">{payload}</script><script>
const R=JSON.parse(document.getElementById('data').textContent),F=document.getElementById('factor'),W=document.getElementById('window');
R.candidates.forEach((c,i)=>F.add(new Option(c.id,i)));Object.entries(R.window_labels).forEach(([k,v])=>W.add(new Option(v,k)));
const pct=x=>Number.isFinite(x)?(x*100).toFixed(2)+'%':'—',num=(x,d=4)=>Number.isFinite(x)?x.toFixed(d):'—';
function draw(series){{const c=document.getElementById('chart'),g=c.getContext('2d'),X=c.width,Y=c.height,p=42,all=series.flatMap(s=>s.v).filter(Number.isFinite);g.clearRect(0,0,X,Y);if(!all.length)return;let lo=Math.min(...all),hi=Math.max(...all);if(hi===lo)hi=lo+1;g.strokeStyle='#dfe4ec';g.beginPath();g.moveTo(p,p);g.lineTo(p,Y-p);g.lineTo(X-p,Y-p);g.stroke();const cs=['#315efb','#079669','#d97706','#8b5cf6','#d44','#111827','#78a0ff'];series.forEach((s,j)=>{{g.strokeStyle=cs[j%cs.length];g.lineWidth=j>=5?2.7:1.5;g.beginPath();s.v.forEach((v,i)=>{{const x=p+i*(X-2*p)/Math.max(1,s.v.length-1),y=Y-p-(v-lo)*(Y-2*p)/(hi-lo);i?g.lineTo(x,y):g.moveTo(x,y)}});g.stroke();g.fillStyle=g.strokeStyle;g.fillText(s.n,p+j*145,18)}});g.fillStyle='#697287';g.fillText(hi.toFixed(2),3,p);g.fillText(lo.toFixed(2),3,Y-p)}}
function render(){{const c=R.candidates[+F.value],m=c.windows[W.value];document.getElementById('detail').innerHTML=`<table><tr><th>方向</th><th>IC</th><th>ICIR</th><th>RankIC</th><th>RankICIR</th><th>Q1</th><th>Q2</th><th>Q3</th><th>Q4</th><th>Q5</th><th>毛多空</th><th>扣费多空</th><th>扣费年化</th><th>换手</th><th>周数</th></tr><tr><td>${{c.direction>0?'+1':'-1'}}</td><td>${{num(m.ic.mean)}}</td><td>${{num(m.ic.ir_annual,2)}}</td><td>${{num(m.rank_ic.mean)}}</td><td>${{num(m.rank_ic.ir_annual,2)}}</td>${{m.group_total.map(p=>`<td>${{pct(p)}}</td>`).join('')}}<td>${{pct(m.gross_ls_total)}}</td><td>${{pct(m.net_ls_total)}}</td><td>${{pct(m.net_ls_annual)}}</td><td>${{pct(m.turnover,1)}}</td><td>${{m.weeks}}</td></tr></table>`;document.getElementById('expr').textContent=c.expression;draw(m.group_nav.map((v,i)=>({{n:'Q'+(i+1),v}})).concat([{{n:'毛多空',v:m.gross_nav}},{{n:'净多空',v:m.net_nav}}]))}}F.onchange=render;W.onchange=render;W.value='test';render();
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
    start, end = WINDOWS[0][2], WINDOWS[2][3]
    instruments = load_pit_codes(args.pit, start, end)
    dates = load_pit_dates(args.pit, start, end)
    date_index = pd.Index([str(value) for value in dates], name="trade_date")
    instrument_index = pd.Index([str(value) for value in instruments], name="instrument")

    fields = tuple(sorted({field for candidate in candidates for field in candidate["genome"].required_fields}))
    warmup_dates = [str(value) for value in dates if str(value) >= "2024-01-02"]
    print(f"[equal-report] rebuild holdout fields={fields}", flush=True)
    context, meta = build_minute_slice(
        args.minute_parquet, "2024-01-02", end, fields=fields,
        instruments=instruments, dates=warmup_dates, device=device,
    )
    close = load_daily_close_tensor(args.daily_parquet, dates, instruments, device=device)
    pool = load_pit_daily_mask(args.pit, dates, instruments, device=device) & torch.isfinite(close)
    fwd = tensor_rebalance_fwd_ret(close, dates, "week_end", 1)
    fwd = torch.where(pool, fwd, torch.full_like(fwd, float("nan")))
    date_array = np.asarray(dates)
    holdout_dates = np.asarray(meta["dates"])
    use = np.flatnonzero(holdout_dates >= WINDOWS[2][2])
    full_positions = np.asarray([date_index.get_loc(str(holdout_dates[index])) for index in use])

    destination = Path(args.out)
    partial_path = destination.with_suffix(".partial.json")
    output = json.loads(partial_path.read_text(encoding="utf-8")) if partial_path.exists() else []
    completed = {row["id"] for row in output}
    for position, candidate in enumerate(candidates, 1):
        if candidate["id"] in completed:
            print(f"[equal-report] {position}/{len(candidates)} {candidate['id']} cached", flush=True)
            continue
        print(f"[equal-report] {position}/{len(candidates)} {candidate['id']}", flush=True)
        cached = pd.read_parquet(candidate["parquet"], columns=["instrument", "trade_date", "factor"])
        cached["trade_date"] = cached["trade_date"].astype(str)
        frame = cached.pivot(index="instrument", columns="trade_date", values="factor")
        frame = frame.reindex(index=instrument_index, columns=date_index)
        factor = torch.as_tensor(frame.to_numpy(copy=True), device=device, dtype=torch.float32)
        factor = trailing_signal_mean(factor, 5)
        holdout = trailing_signal_mean(
            candidate["genome"].evaluate(context, chunk_rows=args.chunk_rows), 5
        )
        factor[:, full_positions] = holdout[:, use]
        factor = torch.where(pool, factor, torch.full_like(factor, float("nan")))
        metrics = {}
        for key, _label, window_start, window_end in WINDOWS:
            mask = torch.as_tensor(
                (date_array >= window_start) & (date_array <= window_end), device=device
            )
            metrics[key] = evaluate_period(
                factor, fwd, mask, candidate["direction"], args.groups, args.cost_bps
            )
        output.append({
            "id": candidate["id"], "direction": candidate["direction"],
            "expression": candidate["expression"], "train_fitness": candidate["fitness"],
            "windows": metrics,
        })
        output.sort(key=lambda row: row["id"])
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path.write_text(
            json.dumps(_json_safe(output), ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
        del cached, frame, factor, holdout
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    expressions = defaultdict(list)
    for candidate in output:
        expressions[candidate["expression"]].append(candidate["id"])
    report = {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "method": {
            "rebalance": "week_end", "signal_average_days": 5, "groups": args.groups,
            "spread": "Q5-Q1 after locked direction", "cost_bps": args.cost_bps,
            "holdout_rebuilt_from_minutes": True, "holdout_warmup_start": "2024-01-02",
        },
        "window_labels": {key: label for key, label, _start, _end in WINDOWS},
        "window_ranges": {key: [start, end] for key, _label, start, end in WINDOWS},
        "unique_expressions": len(expressions),
        "duplicate_groups": [ids for ids in expressions.values() if len(ids) > 1],
        "candidates": output,
    }
    report["analysis"] = {key: _aggregate(output, key) for key, _label, _start, _end in WINDOWS}
    report["best"] = {
        "test_ic": _best(output, "test", ("ic", "mean")),
        "test_rank_ic": _best(output, "test", ("rank_ic", "mean")),
        "test_net": _best(output, "test", ("net_ls_total",)),
        "y2025_net": _best(output, "y2025", ("net_ls_total",)),
        "y2026_net": _best(output, "y2026", ("net_ls_total",)),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    build_html(report, destination)
    destination.with_suffix(".json").write_text(
        json.dumps(_json_safe(report), ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    partial_path.unlink(missing_ok=True)
    print(f"[equal-report] written -> {destination}", flush=True)


if __name__ == "__main__":
    main()
