"""Three-window performance report for Dark-Flow Pareto factors.

The GP exports stop at 2024-12-31.  Holdout values are rebuilt from minute
data with a 2024 operator warm-up.  Each factor keeps the direction selected
by the GP; validation and test data never re-select the sign.
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
    ("test", "测试期", "2025-01-02", "2026-07-31"),
)


def finite(value):
    return value is not None and np.isfinite(value)


def number(value, digits=4):
    return "—" if not finite(value) else f"{value:.{digits}f}"


def percent(value, digits=2):
    return "—" if not finite(value) else f"{value:.{digits}%}"


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def aggregate(candidates, key):
    metrics = [candidate["windows"][key] for candidate in candidates]
    ic = np.asarray([item["ic"]["mean"] for item in metrics], dtype=float)
    rank_ic = np.asarray([item["rank_ic"]["mean"] for item in metrics], dtype=float)
    net = np.asarray([item["net_ls_total"] for item in metrics], dtype=float)
    return {
        "positive_ic": int(np.sum(ic > 0)),
        "positive_rank_ic": int(np.sum(rank_ic > 0)),
        "positive_net": int(np.sum(net > 0)),
        "median_net": float(np.nanmedian(net)),
    }


def best(candidates, key, path):
    def metric(candidate):
        value = candidate["windows"][key]
        for part in path:
            value = value[part]
        return value if finite(value) else -float("inf")

    winner = max(candidates, key=metric)
    value = winner["windows"][key]
    for part in path:
        value = value[part]
    return {"id": winner["id"], "value": value}


def build_html(report, destination):
    labels = report["window_labels"]
    rows = []
    for candidate in report["candidates"]:
        for key, _label, _start, _end in WINDOWS:
            item = candidate["windows"][key]
            groups = "".join(f"<td>{percent(value)}</td>" for value in item["group_total"])
            rows.append(
                f"<tr><td>{candidate['id']}</td><td>{labels[key]}</td>"
                f"<td>{candidate['direction']:+d}</td>"
                f"<td>{number(item['ic']['mean'])}</td><td>{number(item['ic']['ir_annual'], 2)}</td>"
                f"<td>{number(item['rank_ic']['mean'])}</td><td>{number(item['rank_ic']['ir_annual'], 2)}</td>"
                f"{groups}<td>{percent(item['gross_ls_total'])}</td>"
                f"<td>{percent(item['net_ls_total'])}</td><td>{percent(item['net_ls_annual'])}</td>"
                f"<td>{percent(item['turnover'], 1)}</td><td>{item['weeks']}</td></tr>"
            )

    test = report["analysis"]["test"]
    duplicate_note = "；".join("、".join(group) for group in report["duplicate_groups"]) or "无"
    payload = json.dumps(json_safe(report), ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    destination.write_text(f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Dark Flow 10个Pareto因子三阶段绩效</title>
<style>
:root{{--bg:#f3f5f8;--card:#fff;--ink:#172033;--muted:#697287;--line:#dde3ec;--green:#079669}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1600px;margin:auto;padding:28px}}h1{{margin:0 0 5px}}h2{{font-size:19px;margin:0 0 13px}}h3{{font-size:16px}}
.sub,.note{{color:var(--muted)}}.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin:15px 0;box-shadow:0 2px 12px #1720330a}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.periods{{grid-template-columns:repeat(3,1fr)}}.metric{{border:1px solid var(--line);border-radius:9px;padding:12px}}.metric b{{display:block;font-size:19px}}
.scroll{{overflow:auto;max-height:680px}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}th,td{{padding:8px 9px;border-bottom:1px solid var(--line);text-align:right}}th{{position:sticky;top:0;background:#f8f9fc;z-index:1}}th:first-child,td:first-child{{text-align:left}}
select,button{{padding:8px 12px;border:1px solid var(--line);border-radius:7px;background:white;margin-right:8px}}button{{cursor:pointer}}canvas{{width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:white}}code{{word-break:break-all;white-space:normal}}.good{{color:var(--green)}}
@media(max-width:900px){{.grid,.periods{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>Dark Flow：10个 Pareto 因子三阶段绩效</h1>
<div class="sub">生成时间 {report['generated']} · 周频调仓 · 5日信号均值 · 五分组 · Q5−Q1 · 双边30bps成本</div>
<section class="card"><h2>阶段口径</h2><div class="grid periods">
<div class="metric">样本期<b>2018-01-02—2022-03-07</b></div>
<div class="metric">验证期<b>2022-03-08—2024-12-31</b></div>
<div class="metric">测试期<b>2025-01-02—2026-07-31</b></div></div>
<p class="note">测试期不拆分年度。IC为周度截面Pearson相关，RankIC为Spearman秩相关；ICIR/RankICIR=mean/std×√52。因子方向使用GP训练期锁定值，验证期和测试期均不重新选方向。</p></section>
<section class="card"><h2>独立测试期概览</h2><div class="grid">
<div class="metric">IC为正<b>{test['positive_ic']}/10</b></div><div class="metric">RankIC为正<b>{test['positive_rank_ic']}/10</b></div>
<div class="metric">扣费多空为正<b>{test['positive_net']}/10</b></div><div class="metric">扣费多空中位数<b>{percent(test['median_net'])}</b></div>
<div class="metric">最高测试期IC<b>{report['best']['test_ic']['id']}</b><span>{number(report['best']['test_ic']['value'])}</span></div>
<div class="metric">最高测试期RankIC<b>{report['best']['test_rank_ic']['id']}</b><span>{number(report['best']['test_rank_ic']['value'])}</span></div>
<div class="metric">最高扣费多空<b>{report['best']['test_net']['id']}</b><span>{percent(report['best']['test_net']['value'])}</span></div>
<div class="metric">唯一公式<b>{report['unique_expressions']}/10</b></div></div><p class="note">重复公式组：{duplicate_note}。</p></section>
<section class="card"><h2>全部指标</h2><div class="scroll"><table><thead><tr><th>因子</th><th>阶段</th><th>方向</th><th>IC</th><th>ICIR</th><th>RankIC</th><th>RankICIR</th><th>Q1</th><th>Q2</th><th>Q3</th><th>Q4</th><th>Q5</th><th>毛多空</th><th>扣费多空</th><th>扣费年化</th><th>换手</th><th>周数</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<section class="card"><h2>单因子三阶段五分组曲线</h2><label>选择因子：<select id="factor"></select></label><button id="save">下载当前图 PNG</button><div id="detail"></div>
<h3>样本期、验证期、测试期五分组累计净值（一个画布）</h3><canvas id="groups" width="1500" height="790"></canvas><p class="note">三个子区分别使用自己的纵轴范围；Q1—Q5颜色在三个时期保持一致。每个阶段均从净值1重新起算。</p><p><code id="expr"></code></p></section>
<script id="data" type="application/json">{payload}</script><script>
const R=JSON.parse(document.getElementById('data').textContent),S=document.getElementById('factor'),K=['select','valid','test'];R.candidates.forEach((c,i)=>S.add(new Option(c.id,i)));
const pct=x=>Number.isFinite(x)?(x*100).toFixed(2)+'%':'—',num=(x,d=4)=>Number.isFinite(x)?x.toFixed(d):'—',C=['#315efb','#079669','#d97706','#8b5cf6','#df4d4d'];
function panel(g,m,title,range,x0,y0,w,h,legend){{const pL=55,pR=18,pT=33,pB=30,all=m.group_nav.flat().filter(Number.isFinite);let lo=Math.min(1,...all),hi=Math.max(1,...all);if(hi===lo)hi=lo+1;const X=x0+pL,Y=y0+pT,W=w-pL-pR,H=h-pT-pB;g.strokeStyle='#dfe4ec';g.lineWidth=1;for(let z=0;z<=4;z++){{const yy=Y+H*z/4;g.beginPath();g.moveTo(X,yy);g.lineTo(X+W,yy);g.stroke();const val=hi-(hi-lo)*z/4;g.fillStyle='#697287';g.fillText(val.toFixed(2),x0+4,yy+4)}}g.strokeStyle='#9aa4b5';g.beginPath();g.moveTo(X,Y);g.lineTo(X,Y+H);g.lineTo(X+W,Y+H);g.stroke();g.fillStyle='#172033';g.font='bold 16px system-ui';g.fillText(title+'  '+range,X,y0+20);g.font='12px system-ui';m.group_nav.forEach((v,j)=>{{g.strokeStyle=C[j];g.lineWidth=2;g.beginPath();v.forEach((q,i)=>{{const xx=X+i*W/Math.max(1,v.length-1),yy=Y+H-(q-lo)*H/(hi-lo);i?g.lineTo(xx,yy):g.moveTo(xx,yy)}});g.stroke()}});g.fillStyle='#697287';g.fillText('阶段起点',X,Y+H+20);g.fillText('阶段终点',X+W-48,Y+H+20);if(legend)m.group_nav.forEach((_,j)=>{{const xx=X+j*115;g.fillStyle=C[j];g.fillRect(xx,Y-23,18,3);g.fillText('Q'+(j+1),xx+24,Y-17)}})}}
function draw(c){{const el=document.getElementById('groups'),g=el.getContext('2d');g.clearRect(0,0,el.width,el.height);K.forEach((k,i)=>panel(g,c.windows[k],R.window_labels[k],R.window_ranges[k].join('—'),12,10+i*255,1476,240,i===0))}}
function render(){{const c=R.candidates[+S.value];let h='<table><thead><tr><th>阶段</th><th>方向</th><th>IC</th><th>ICIR</th><th>RankIC</th><th>RankICIR</th><th>Q1</th><th>Q2</th><th>Q3</th><th>Q4</th><th>Q5</th><th>毛多空</th><th>扣费多空</th><th>扣费年化</th><th>换手</th><th>周数</th></tr></thead><tbody>';K.forEach(k=>{{const m=c.windows[k];h+=`<tr><td>${{R.window_labels[k]}}</td><td>${{c.direction>0?'+1':'-1'}}</td><td>${{num(m.ic.mean)}}</td><td>${{num(m.ic.ir_annual,2)}}</td><td>${{num(m.rank_ic.mean)}}</td><td>${{num(m.rank_ic.ir_annual,2)}}</td>${{m.group_total.map(x=>`<td>${{pct(x)}}</td>`).join('')}}<td>${{pct(m.gross_ls_total)}}</td><td>${{pct(m.net_ls_total)}}</td><td>${{pct(m.net_ls_annual)}}</td><td>${{pct(m.turnover,1)}}</td><td>${{m.weeks}}</td></tr>`}});document.getElementById('detail').innerHTML=h+'</tbody></table>';document.getElementById('expr').textContent=c.expression;draw(c)}}S.onchange=render;document.getElementById('save').onclick=()=>{{const a=document.createElement('a');a.download=R.candidates[+S.value].id+'_three_period_groups.png';a.href=document.getElementById('groups').toDataURL('image/png');a.click()}};render();
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
    date_index = pd.Index([str(value) for value in dates], name="trade_date")
    instrument_index = pd.Index([str(value) for value in instruments], name="instrument")
    fields = tuple(sorted({field for candidate in candidates for field in candidate["genome"].required_fields}))

    warmup_dates = [str(value) for value in dates if str(value) >= "2024-01-02"]
    print(f"[dark-flow-report] rebuild holdout fields={fields}", flush=True)
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
    use = np.flatnonzero(holdout_dates >= WINDOWS[-1][2])
    full_positions = np.asarray([date_index.get_loc(str(holdout_dates[index])) for index in use])

    destination = Path(args.out)
    partial = destination.with_suffix(".partial.json")
    output = json.loads(partial.read_text(encoding="utf-8")) if partial.exists() else []
    completed = {row["id"] for row in output}
    for position, candidate in enumerate(candidates, 1):
        if candidate["id"] in completed:
            print(f"[dark-flow-report] {position}/{len(candidates)} {candidate['id']} cached", flush=True)
            continue
        print(f"[dark-flow-report] {position}/{len(candidates)} {candidate['id']}", flush=True)
        cached = pd.read_parquet(candidate["parquet"], columns=["instrument", "trade_date", "factor"])
        cached["trade_date"] = cached["trade_date"].astype(str)
        frame = cached.pivot(index="instrument", columns="trade_date", values="factor")
        frame = frame.reindex(index=instrument_index, columns=date_index)
        factor = torch.as_tensor(frame.to_numpy(copy=True), device=device, dtype=torch.float32)
        factor = trailing_signal_mean(factor, 5)
        holdout = trailing_signal_mean(candidate["genome"].evaluate(context, chunk_rows=args.chunk_rows), 5)
        factor[:, full_positions] = holdout[:, use]
        factor = torch.where(pool, factor, torch.full_like(factor, float("nan")))
        metrics = {}
        for key, _label, window_start, window_end in WINDOWS:
            mask = torch.as_tensor((date_array >= window_start) & (date_array <= window_end), device=device)
            metrics[key] = evaluate_period(factor, fwd, mask, candidate["direction"], args.groups, args.cost_bps)
        output.append({
            "id": candidate["id"], "direction": candidate["direction"],
            "expression": candidate["expression"], "train_fitness": candidate["fitness"],
            "windows": metrics,
        })
        output.sort(key=lambda row: row["id"])
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_text(json.dumps(json_safe(output), ensure_ascii=False, allow_nan=False), encoding="utf-8")
        del cached, frame, factor, holdout
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    expressions = defaultdict(list)
    for candidate in output:
        expressions[candidate["expression"]].append(candidate["id"])
    report = {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "method": {"rebalance": "week_end", "signal_average_days": 5, "groups": args.groups,
                   "spread": "Q5-Q1 after locked direction", "cost_bps": args.cost_bps,
                   "holdout_rebuilt_from_minutes": True, "holdout_warmup_start": "2024-01-02"},
        "window_labels": {key: label for key, label, _start, _end in WINDOWS},
        "window_ranges": {key: [begin, finish] for key, _label, begin, finish in WINDOWS},
        "unique_expressions": len(expressions),
        "duplicate_groups": [ids for ids in expressions.values() if len(ids) > 1],
        "candidates": output,
    }
    report["analysis"] = {key: aggregate(output, key) for key, _label, _start, _end in WINDOWS}
    report["best"] = {
        "test_ic": best(output, "test", ("ic", "mean")),
        "test_rank_ic": best(output, "test", ("rank_ic", "mean")),
        "test_net": best(output, "test", ("net_ls_total",)),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    build_html(report, destination)
    destination.with_suffix(".json").write_text(
        json.dumps(json_safe(report), ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    partial.unlink(missing_ok=True)
    print(f"[dark-flow-report] written -> {destination}", flush=True)


if __name__ == "__main__":
    main()
