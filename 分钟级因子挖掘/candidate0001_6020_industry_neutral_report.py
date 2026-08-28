"""Compare candidate_0001_rank0 20/5, 60/20, and 60/20 industry neutral."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from min_gp.climb_mountain_html_report import evaluate_period
from min_gp.dark_flow_industry_neutralization_report import (
    _industry_coverage,
    _industry_rank_r2,
)
from min_gp.data import load_pit_codes, load_pit_daily_mask, load_pit_dates
from min_gp.evaluation.incremental import trailing_signal_mean
from min_gp.evaluation.neutralize import BatchedNeutralizer
from min_gp.label import tensor_rebalance_fwd_ret
from min_gp.numeric.ranking import cross_section_rank
from min_gp.operators.temporal import equal_blend, mean_std_blend
from min_gp.spectral_data import load_daily_close_tensor, load_daily_exposures


WINDOWS = (
    ("select", "样本期", "2018-01-02", "2022-03-07"),
    ("valid", "验证期", "2022-03-08", "2024-12-31"),
    ("test", "测试期", "2025-01-02", "2026-07-31"),
)


def _safe(value):
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _fmt(value, digits=4):
    return "—" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def _pct(value, digits=2):
    return "—" if value is None or not np.isfinite(value) else f"{value:.{digits}%}"


def _factor(raw_a, raw_b, mean_window, signal_window):
    daily = equal_blend(
        mean_std_blend(raw_a, mean_window),
        mean_std_blend(raw_b, mean_window),
    )
    return trailing_signal_mean(daily, signal_window)


def _build_html(report, destination):
    rows = []
    group_rows = []
    labels = report["window_labels"]
    for key, _label, _begin, _finish in WINDOWS:
        window = report["windows"][key]
        base, raw, neutral = window["baseline_20_5"], window["raw_60_20"], window["industry_neutral_60_20"]
        rows.append(
            f"<tr><td>{labels[key]}</td><td>{_pct(window['industry_coverage'], 1)}</td>"
            f"<td>{_fmt(base['ic']['mean'])}</td><td>{_fmt(raw['ic']['mean'])}</td><td>{_fmt(neutral['ic']['mean'])}</td>"
            f"<td>{_fmt(base['rank_ic']['mean'])}</td><td>{_fmt(raw['rank_ic']['mean'])}</td><td>{_fmt(neutral['rank_ic']['mean'])}</td>"
            f"<td>{_pct(base['net_ls_total'])}</td><td>{_pct(raw['net_ls_total'])}</td><td>{_pct(neutral['net_ls_total'])}</td>"
            f"<td>{_pct(neutral['net_ls_total']-raw['net_ls_total'])}</td>"
            f"<td>{_pct(base['turnover'], 1)}</td><td>{_pct(raw['turnover'], 1)}</td><td>{_pct(neutral['turnover'], 1)}</td>"
            f"<td>{_pct(window['raw_60_20_industry_rank_r2'], 2)} → {_pct(window['neutral_60_20_industry_rank_r2'], 2)}</td></tr>"
        )
        groups = "".join(f"<td>{_pct(value)}</td>" for value in neutral["group_total"])
        group_rows.append(
            f"<tr><td>{labels[key]}</td>{groups}<td>{_pct(neutral['gross_ls_total'])}</td>"
            f"<td>{_pct(neutral['net_ls_total'])}</td><td>{_pct(neutral['net_ls_annual'])}</td>"
            f"<td>{_pct(neutral['turnover'], 1)}</td><td>{neutral['weeks']}</td></tr>"
        )

    test = report["windows"]["test"]
    base, raw, neutral = test["baseline_20_5"], test["raw_60_20"], test["industry_neutral_60_20"]
    payload = json.dumps(_safe(report), ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    destination.write_text(f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>candidate_0001 60/20 + 行业中性化</title>
<style>
:root{{--bg:#f3f5f8;--card:#fff;--ink:#172033;--muted:#687187;--line:#dde3ec;--blue:#315efb;--green:#087f5b;--red:#c33}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1650px;margin:auto;padding:28px}}
h1{{margin:0 0 5px}}h2{{font-size:19px;margin:0 0 12px}}h3{{font-size:16px}}.sub,.note{{color:var(--muted)}}.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin:15px 0;box-shadow:0 2px 12px #1720330a}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.metric{{border:1px solid var(--line);border-radius:9px;padding:12px}}.metric span{{display:block;color:var(--muted)}}.metric b{{display:block;font-size:22px;margin-top:3px}}
.pos{{color:var(--green)}}.neg{{color:var(--red)}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}th,td{{padding:8px 9px;border-bottom:1px solid var(--line);text-align:right}}th{{background:#f8f9fc;position:sticky;top:0}}th:first-child,td:first-child{{text-align:left}}
canvas{{width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:white}}code{{word-break:break-all;white-space:normal}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>candidate_0001_rank0：60/20 + 行业中性化</h1>
<div class="sub">每日秩空间行业哑变量 OLS 残差 · 点时申万一级 · 方向固定−1 · 周频调仓 · 五分组 · Q5−Q1 · 双边30bps成本</div>
<section class="card"><h2>独立测试期结论</h2><div class="grid">
<div class="metric"><span>原始20/5净多空</span><b class="{'pos' if base['net_ls_total'] >= 0 else 'neg'}">{_pct(base['net_ls_total'])}</b></div>
<div class="metric"><span>未中性化60/20</span><b class="{'pos' if raw['net_ls_total'] >= 0 else 'neg'}">{_pct(raw['net_ls_total'])}</b></div>
<div class="metric"><span>60/20行业中性</span><b class="{'pos' if neutral['net_ls_total'] >= 0 else 'neg'}">{_pct(neutral['net_ls_total'])}</b></div>
<div class="metric"><span>行业中性带来的变化</span><b class="{'pos' if neutral['net_ls_total']-raw['net_ls_total'] >= 0 else 'neg'}">{_pct(neutral['net_ls_total']-raw['net_ls_total'])}</b></div>
<div class="metric"><span>中性后测试IC</span><b>{_fmt(neutral['ic']['mean'])}</b></div><div class="metric"><span>中性后测试RankIC</span><b>{_fmt(neutral['rank_ic']['mean'])}</b></div>
<div class="metric"><span>中性后测试RankICIR</span><b>{_fmt(neutral['rank_ic']['ir_annual'], 2)}</b></div><div class="metric"><span>中性后测试换手</span><b>{_pct(neutral['turnover'], 1)}</b></div></div></section>
<section class="card"><h2>三阶段三版本对比</h2><div class="scroll"><table><thead><tr><th>阶段</th><th>行业覆盖</th><th>IC 20/5</th><th>IC 60/20</th><th>IC 60/20中性</th><th>RankIC 20/5</th><th>RankIC 60/20</th><th>RankIC 60/20中性</th><th>净多空 20/5</th><th>净多空 60/20</th><th>净多空 60/20中性</th><th>中性化变化</th><th>换手 20/5</th><th>换手 60/20</th><th>换手 60/20中性</th><th>行业R² 60/20原→中</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<section class="card"><h2>60/20行业中性化后五分组</h2><div class="scroll"><table><thead><tr><th>阶段</th><th>Q1</th><th>Q2</th><th>Q3</th><th>Q4</th><th>Q5</th><th>毛多空</th><th>净多空</th><th>净多空年化</th><th>换手</th><th>周数</th></tr></thead><tbody>{''.join(group_rows)}</tbody></table></div></section>
<section class="card"><h2>行业中性化后：三阶段五分组曲线</h2><canvas id="groups" width="1500" height="780"></canvas></section>
<section class="card"><h2>三版本扣费多空曲线</h2><p class="note">每个阶段独立从净值1开始；灰色虚线为20/5，蓝色虚线为未中性化60/20，绿色实线为60/20行业中性。</p><canvas id="ls" width="1500" height="600"></canvas></section>
<section class="card"><h2>构造</h2><code>{report['expression']}</code><p class="note">行业中性化顺序：先得到60/20平滑信号，再做每日截面秩空间行业回归，不使用未来行业信息。</p></section>
<script id="data" type="application/json">{payload}</script><script>
const R=JSON.parse(document.getElementById('data').textContent),K=['select','valid','test'],C=['#315efb','#079669','#d97706','#8b5cf6','#df4d4d'];
function panel(g,series,title,x0,y0,w,h,legend){{const all=series.flatMap(s=>s.v).filter(Number.isFinite);if(!all.length)return;let lo=Math.min(1,...all),hi=Math.max(1,...all);if(hi===lo)hi=lo+1;const L=58,T=38,B=30,X=x0+L,Y=y0+T,W=w-L-20,H=h-T-B;g.strokeStyle='#e1e5ec';for(let z=0;z<=4;z++){{const yy=Y+H*z/4;g.beginPath();g.moveTo(X,yy);g.lineTo(X+W,yy);g.stroke();g.fillStyle='#687187';g.fillText((hi-(hi-lo)*z/4).toFixed(2),x0+5,yy+4)}}g.fillStyle='#172033';g.font='bold 15px system-ui';g.fillText(title,X,y0+20);g.font='12px system-ui';series.forEach((s,j)=>{{g.strokeStyle=s.color||C[j%C.length];g.lineWidth=s.width||2;g.setLineDash(s.dash||[]);g.beginPath();s.v.forEach((v,i)=>{{const xx=X+i*W/Math.max(1,s.v.length-1),yy=Y+H-(v-lo)*H/(hi-lo);i?g.lineTo(xx,yy):g.moveTo(xx,yy)}});g.stroke();g.setLineDash([]);if(legend){{g.fillStyle=g.strokeStyle;g.fillText(s.n,X+j*180,Y-9)}}}})}}
let c=document.getElementById('groups'),g=c.getContext('2d');K.forEach((k,i)=>{{const m=R.windows[k].industry_neutral_60_20;panel(g,m.group_nav.map((v,q)=>({{n:'Q'+(q+1),v:[1,...v]}})),R.window_labels[k],10,8+i*255,1480,240,i===0)}});
c=document.getElementById('ls');g=c.getContext('2d');K.forEach((k,i)=>{{const w=R.windows[k];panel(g,[{{n:'原始20/5',v:[1,...w.baseline_20_5.net_nav],color:'#7b8495',dash:[4,4]}},{{n:'未中性60/20',v:[1,...w.raw_60_20.net_nav],color:'#315efb',dash:[8,4]}},{{n:'60/20行业中性',v:[1,...w.industry_neutral_60_20.net_nav],color:'#079669',width:2.5}}],R.window_labels[k],10,8+i*195,1480,182,i===0)}});
</script></main></body></html>""", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-cache", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--daily-parquet", required=True)
    parser.add_argument("--pit", required=True)
    parser.add_argument("--industry-exposures", required=True)
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--cost-bps", type=float, default=30.0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = "cpu" if args.cpu else "cuda"
    sidecar = json.loads(Path(args.sidecar).read_text(encoding="utf-8"))
    direction = int(sidecar["fitness"]["direction"])
    dates = load_pit_dates(args.pit, WINDOWS[0][2], WINDOWS[-1][3])
    instruments = load_pit_codes(args.pit, WINDOWS[0][2], WINDOWS[-1][3])
    date_array = np.asarray([str(value) for value in dates])
    cached = torch.load(args.raw_cache, map_location="cpu", weights_only=False)
    if cached["dates"] != [str(value) for value in dates]:
        raise ValueError("raw component dates do not align")
    if cached["instruments"] != [str(value) for value in instruments]:
        raise ValueError("raw component instruments do not align")
    raw_a, raw_b = (value.to(device) for value in cached["components"])

    print("[6020-industry] loading daily returns and point-in-time industry", flush=True)
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

    print("[6020-industry] building matched 20/5 and 60/20 signals", flush=True)
    baseline = torch.where(
        matched_pool, _factor(raw_a, raw_b, 20, 5), torch.full_like(raw_a, float("nan"))
    )
    raw_60_20 = torch.where(
        matched_pool, _factor(raw_a, raw_b, 60, 20), torch.full_like(raw_a, float("nan"))
    )
    neutralizer = BatchedNeutralizer(
        pool.shape,
        industry=industry,
        rank_space=True,
        min_cross_section=30,
    )
    print("[6020-industry] daily rank-space industry OLS", flush=True)
    neutral = neutralizer(raw_60_20)
    neutral = torch.where(matched_pool, neutral, torch.full_like(neutral, float("nan")))
    raw_rank = cross_section_rank(raw_60_20.float())
    neutral_rank = cross_section_rank(neutral.float())

    windows = {}
    for key, label, begin, finish in WINDOWS:
        print(f"[6020-industry] evaluate {label}", flush=True)
        date_mask = torch.as_tensor(
            (date_array >= begin) & (date_array <= finish), device=device
        )
        windows[key] = {
            "industry_coverage": _industry_coverage(industry, pool, date_mask),
            "raw_60_20_industry_rank_r2": _industry_rank_r2(
                raw_rank, industry, fwd, date_mask, levels
            ),
            "neutral_60_20_industry_rank_r2": _industry_rank_r2(
                neutral_rank, industry, fwd, date_mask, levels
            ),
            "baseline_20_5": evaluate_period(
                baseline, fwd, date_mask, direction, args.groups, args.cost_bps
            ),
            "raw_60_20": evaluate_period(
                raw_60_20, fwd, date_mask, direction, args.groups, args.cost_bps
            ),
            "industry_neutral_60_20": evaluate_period(
                neutral, fwd, date_mask, direction, args.groups, args.cost_bps
            ),
        }

    report = {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "candidate_id": "candidate_0001_rank0",
        "direction": direction,
        "expression": "industry_neutral_rank_ols(trailing_signal_mean(equal_blend(mean_std_blend(component_1,60),mean_std_blend(component_2,60)),20), SW_level1_PIT)",
        "method": {
            "order": ["mean_std_blend_60", "equal_blend", "signal_mean_20", "industry_rank_ols_residual"],
            "industry": "point-in-time SW level-1",
            "rank_space": True,
            "min_cross_section": 30,
            "matched_universe": True,
            "rebalance": "week_end",
            "groups": args.groups,
            "spread": "Q5-Q1 after locked direction",
            "cost_bps": args.cost_bps,
        },
        "industry_level_count": len(industry_names),
        "industry_levels": industry_names,
        "window_labels": {key: label for key, label, _begin, _finish in WINDOWS},
        "window_ranges": {key: [begin, finish] for key, _label, begin, finish in WINDOWS},
        "windows": windows,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _build_html(report, destination)
    destination.with_suffix(".json").write_text(
        json.dumps(_safe(report), ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    print(f"[6020-industry] written -> {destination}", flush=True)


if __name__ == "__main__":
    main()
