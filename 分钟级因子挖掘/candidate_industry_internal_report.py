"""Within-industry IC and long-short report for one handbook GP factor."""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

from min_gp.config import (
    ADJUSTED_CLOSE_PARQUET,
    INDUSTRY_VALUE_EXPOSURES_PARQUET,
    MINUTE_PARQUET,
    ZZ500_PIT_PARQUET,
)
from min_gp.data import load_pit_codes, load_pit_daily_mask, load_pit_dates
from min_gp.evaluation.incremental import trailing_signal_mean
from min_gp.evaluation.neutralize import BatchedNeutralizer
from min_gp.factors.handbook_skeleton import HandbookSkeletonGenome
from min_gp.label import tensor_rebalance_fwd_ret
from min_gp.long_short_battle_html_report import WINDOWS
from min_gp.spectral_data import (
    build_minute_slice,
    load_daily_close_tensor,
    load_daily_exposures,
)


STAGE_LABELS = {"select": "样本期", "valid": "验证期", "test": "测试期"}

# Current categories come from CNINFO's Shenyin-Wanguo category endpoint.
# Codes no longer returned by that endpoint are retained with their historical
# level-1 names because the point-in-time exposure file spans taxonomy changes.
INDUSTRY_NAMES = {
    "11": "农林牧渔", "21": "采掘（历史分类）", "22": "基础化工",
    "23": "钢铁", "24": "有色金属", "25": "建筑建材（历史分类）",
    "26": "机械设备（历史分类）", "27": "电子", "28": "汽车",
    "31": "交运设备（历史分类）", "32": "信息设备（历史分类）",
    "33": "家用电器", "34": "食品饮料", "35": "纺织服饰",
    "36": "轻工制造", "37": "医药生物", "41": "公用事业",
    "42": "交通运输", "43": "房地产", "44": "金融服务（历史分类）",
    "45": "商贸零售", "46": "社会服务", "47": "信息服务（历史分类）",
    "48": "银行", "49": "非银金融", "51": "综合",
    "61": "建筑材料", "62": "建筑装饰", "63": "电力设备",
    "64": "机械设备", "65": "国防军工", "71": "计算机",
    "72": "传媒", "73": "通信", "74": "煤炭",
    "75": "石油石化", "76": "环保", "77": "美容护理",
}


def _stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size < 2:
        return {"mean": None, "ir": None, "ir_annual": None,
                "positive": None, "n": int(array.size)}
    mean, sigma = float(array.mean()), float(array.std())
    ir = mean / sigma if sigma > 0 else None
    return {
        "mean": mean,
        "ir": ir,
        "ir_annual": None if ir is None else ir * float(np.sqrt(52)),
        "positive": float((array > 0).mean()),
        "n": int(array.size),
    }


def _total_return(values: list[float]) -> float | None:
    return float(np.prod(1.0 + np.asarray(values)) - 1.0) if values else None


def _annualized(values: list[float]) -> float | None:
    if not values:
        return None
    wealth = float(np.prod(1.0 + np.asarray(values)))
    return wealth ** (52.0 / len(values)) - 1.0 if wealth > 0 else None


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _evaluate_industry(
    signal: np.ndarray,
    fwd: np.ndarray,
    industry: np.ndarray,
    code: int,
    date_positions: np.ndarray,
    direction: int,
    groups: int,
    cost_bps: float,
    min_ic_names: int,
    min_ls_names: int,
) -> dict:
    ic_values, rank_ic_values, name_counts, ic_dates = [], [], [], []
    grouped = [[] for _ in range(groups)]
    gross, net, turnovers, ls_dates = [], [], [], []
    previous = None
    for day in date_positions:
        valid = (
            np.isfinite(signal[:, day])
            & np.isfinite(fwd[:, day])
            & (industry[:, day] == code)
        )
        names = np.flatnonzero(valid)
        count = len(names)
        if count >= min_ic_names:
            x, y = signal[names, day], fwd[names, day]
            ic_values.append(_corr(x, y) * direction)
            rank_ic_values.append(
                _corr(rankdata(x, method="average"), rankdata(y, method="average"))
                * direction
            )
            name_counts.append(count)
            ic_dates.append(int(day))
        if count < min_ls_names:
            continue
        scores = signal[names, day] * direction
        order = np.argsort(scores, kind="stable")
        chunks = np.array_split(names[order], groups)
        if any(len(chunk) == 0 for chunk in chunks):
            continue
        means = [float(np.mean(fwd[chunk, day])) for chunk in chunks]
        for index, value in enumerate(means):
            grouped[index].append(value)
        spread = means[-1] - means[0]
        weights = np.zeros(signal.shape[0], dtype=np.float32)
        weights[chunks[-1]] = 1.0 / len(chunks[-1])
        weights[chunks[0]] = -1.0 / len(chunks[0])
        turnover = (
            float(np.abs(weights).sum())
            if previous is None else float(np.abs(weights - previous).sum())
        )
        previous = weights
        gross.append(spread)
        net.append(spread - 0.5 * turnover * cost_bps * 1e-4)
        turnovers.append(turnover)
        ls_dates.append(int(day))
    return {
        "ic": _stats(ic_values),
        "rank_ic": _stats(rank_ic_values),
        "average_names": float(np.mean(name_counts)) if name_counts else None,
        "min_names": int(np.min(name_counts)) if name_counts else None,
        "max_names": int(np.max(name_counts)) if name_counts else None,
        "ic_dates": ic_dates,
        "group_total": [_total_return(values) for values in grouped],
        "group_nav": [np.cumprod(1.0 + np.asarray(values)).tolist() for values in grouped],
        "gross_ls_total": _total_return(gross),
        "net_ls_total": _total_return(net),
        "net_ls_annual": _annualized(net),
        "turnover": float(np.mean(turnovers)) if turnovers else None,
        "gross_nav": np.cumprod(1.0 + np.asarray(gross)).tolist(),
        "net_nav": np.cumprod(1.0 + np.asarray(net)).tolist(),
        "ls_dates": ls_dates,
        "ls_weeks": len(ls_dates),
    }


def _fmt(value, digits=4) -> str:
    return "—" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def _pct(value, digits=2) -> str:
    return "—" if value is None or not np.isfinite(value) else f"{value:.{digits}%}"


def _clean(value):
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _write_html(report: dict, destination: Path) -> None:
    table_rows = []
    for industry_row in report["industries"]:
        for stage in ("select", "valid", "test"):
            metric = industry_row["windows"][stage]
            reliable = metric["ic"]["n"] >= 30 and metric["ls_weeks"] >= 30
            groups = "".join(f"<td>{_pct(value)}</td>" for value in metric["group_total"])
            table_rows.append(
                f"<tr data-stage='{stage}'><td>{industry_row['code']}</td>"
                f"<td>{industry_row['name']}</td><td>{STAGE_LABELS[stage]}</td>"
                f"<td>{_fmt(metric['average_names'], 1)}</td>"
                f"<td>{metric['ic']['n']}</td><td>{metric['ls_weeks']}</td>"
                f"<td>{_fmt(metric['ic']['mean'])}</td><td>{_fmt(metric['ic']['ir_annual'], 2)}</td>"
                f"<td>{_fmt(metric['rank_ic']['mean'])}</td><td>{_fmt(metric['rank_ic']['ir_annual'], 2)}</td>"
                f"{groups}<td>{_pct(metric['gross_ls_total'])}</td>"
                f"<td class='{'good' if metric['net_ls_total'] is not None and metric['net_ls_total'] > 0 else 'bad'}'>{_pct(metric['net_ls_total'])}</td>"
                f"<td>{_pct(metric['net_ls_annual'])}</td><td>{_pct(metric['turnover'], 1)}</td>"
                f"<td>{'较可靠' if reliable else '样本偏少'}</td></tr>"
            )

    test_reliable = [
        row for row in report["industries"]
        if row["windows"]["test"]["ic"]["n"] >= 30
        and row["windows"]["test"]["ls_weeks"] >= 30
        and row["windows"]["test"]["net_ls_total"] is not None
    ]
    best_ls = sorted(
        test_reliable, key=lambda row: row["windows"]["test"]["net_ls_total"], reverse=True
    )[:5]
    best_ic = sorted(
        test_reliable,
        key=lambda row: row["windows"]["test"]["rank_ic"]["mean"] or -999,
        reverse=True,
    )[:5]
    ls_cards = "".join(
        f"<div class='metric'><b>{row['name']}</b><span>{_pct(row['windows']['test']['net_ls_total'])}</span><small>RankIC {_fmt(row['windows']['test']['rank_ic']['mean'])}</small></div>"
        for row in best_ls
    )
    ic_cards = "".join(
        f"<div class='metric'><b>{row['name']}</b><span>{_fmt(row['windows']['test']['rank_ic']['mean'])}</span><small>净多空 {_pct(row['windows']['test']['net_ls_total'])}</small></div>"
        for row in best_ic
    )
    payload = json.dumps(_clean(report), ensure_ascii=False).replace("</", "<\\/")
    destination.write_text(
        f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{report['candidate_id']} 行业内绩效</title><style>
:root{{--bg:#f3f5f8;--card:#fff;--ink:#172033;--muted:#687187;--line:#dde3ec;--blue:#315efb;--green:#087f5b;--red:#c33}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1800px;margin:auto;padding:28px}}h1{{margin:0 0 5px}}h2{{font-size:19px}}.sub,.note,small{{color:var(--muted)}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin:15px 0;box-shadow:0 2px 12px #1720330a}}.grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}}.metric{{border:1px solid var(--line);border-radius:8px;padding:10px}}.metric b,.metric span,.metric small{{display:block}}.metric span{{font-size:20px;font-weight:650}}
.scroll{{overflow:auto;max-height:720px}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}th,td{{padding:8px;border-bottom:1px solid var(--line);text-align:right}}th{{position:sticky;top:0;background:#f8f9fc;z-index:1}}th:nth-child(2),td:nth-child(2){{text-align:left}}.good{{color:var(--green);font-weight:600}}.bad{{color:var(--red)}}select{{padding:8px 12px;border:1px solid var(--line);border-radius:7px;background:#fff}}canvas{{width:100%;height:350px;border:1px solid var(--line);border-radius:8px}}code{{word-break:break-all;white-space:normal}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>{report['candidate_id']}：逐行业内部 IC 与多空收益</h1>
<div class="sub">行业中性化后信号 · 点时申万一级（含历史版分类）· 行业内截面排序 · 周频调仓 · 5日信号均值 · 行业内Q5−Q1 · 双边30bps成本</div>
<section class="card"><h2>统计口径</h2><p>IC/RankIC 每个有效周至少 {report['min_ic_names']} 只股票；五分组多空每个有效周至少 {report['min_ls_names']} 只股票。表中的“较可靠”要求该阶段 IC 和多空都至少 30 周。方向固定为 {report['direction']:+d}，没有在行业或样本外重新选方向。</p><p class="note">行业代码来自历史点时数据。申万分类曾调整，已经停止使用的旧一级代码单独标记为“历史分类”，不会强行映射到当前新行业。</p></section>
<section class="card"><h2>测试期扣费多空领先行业</h2><div class="grid">{ls_cards}</div></section>
<section class="card"><h2>测试期 RankIC 领先行业</h2><div class="grid">{ic_cards}</div></section>
<section class="card"><h2>全部行业三阶段指标</h2><label>阶段筛选：<select id="stage"><option value="all">全部</option><option value="select">样本期</option><option value="valid">验证期</option><option value="test" selected>测试期</option></select></label><div class="scroll"><table id="metrics"><thead><tr><th>代码</th><th>行业</th><th>阶段</th><th>平均股票数</th><th>IC周数</th><th>多空周数</th><th>IC</th><th>ICIR</th><th>RankIC</th><th>RankICIR</th><th>Q1</th><th>Q2</th><th>Q3</th><th>Q4</th><th>Q5</th><th>毛多空</th><th>扣费多空</th><th>扣费年化</th><th>换手</th><th>可靠性</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div></section>
<section class="card"><h2>测试期行业横向比较</h2><p class="note">仅展示 IC 和多空均不少于30周的行业。</p><canvas id="bars" width="1600" height="350"></canvas></section>
<section class="card"><h2>单行业三阶段曲线</h2><label>选择行业：<select id="industry"></select></label><div id="detail"></div><canvas id="curves" width="1600" height="350"></canvas></section>
<section class="card"><h2>因子表达式</h2><code>{html.escape(report['expression'])}</code></section>
<script id="data" type="application/json">{payload}</script><script>
const R=JSON.parse(document.getElementById('data').textContent),stage=document.getElementById('stage');function filter(){{document.querySelectorAll('#metrics tbody tr').forEach(r=>r.style.display=stage.value==='all'||r.dataset.stage===stage.value?'':'none')}}stage.onchange=filter;filter();
function line(id,series){{const c=document.getElementById(id),g=c.getContext('2d'),W=c.width,H=c.height,p=55,all=series.flatMap(s=>s.v).filter(Number.isFinite);g.clearRect(0,0,W,H);if(!all.length)return;let lo=Math.min(...all),hi=Math.max(...all);if(hi===lo)hi=lo+1;g.strokeStyle='#dfe4ec';g.beginPath();g.moveTo(p,p);g.lineTo(p,H-p);g.lineTo(W-p,H-p);g.stroke();const cs=['#315efb','#08a36a','#e28a11'];series.forEach((s,j)=>{{g.strokeStyle=cs[j%3];g.lineWidth=2;g.beginPath();s.v.forEach((v,i)=>{{const x=p+i*(W-2*p)/Math.max(1,s.v.length-1),y=H-p-(v-lo)*(H-2*p)/(hi-lo);i?g.lineTo(x,y):g.moveTo(x,y)}});g.stroke();g.fillStyle=g.strokeStyle;g.fillText(s.n,p+j*210,18)}});g.fillStyle='#687187';g.fillText(hi.toFixed(2),3,p);g.fillText(lo.toFixed(2),3,H-p)}}
const reliable=R.industries.filter(x=>{{const m=x.windows.test;return m.ic.n>=30&&m.ls_weeks>=30&&Number.isFinite(m.net_ls_total)}}).sort((a,b)=>b.windows.test.net_ls_total-a.windows.test.net_ls_total);(function(){{const c=document.getElementById('bars'),g=c.getContext('2d'),W=c.width,H=c.height,p=65,n=reliable.length;if(!n)return;const vals=reliable.map(x=>x.windows.test.net_ls_total),lo=Math.min(0,...vals),hi=Math.max(0,...vals),span=hi-lo||1,bw=(W-2*p)/n;g.clearRect(0,0,W,H);reliable.forEach((x,i)=>{{const v=x.windows.test.net_ls_total,y0=H-p-(0-lo)*(H-2*p)/span,y=H-p-(v-lo)*(H-2*p)/span;g.fillStyle=v>=0?'#087f5b':'#d44';g.fillRect(p+i*bw+2,Math.min(y,y0),Math.max(2,bw-4),Math.abs(y-y0));g.save();g.translate(p+i*bw+bw/2,H-p+5);g.rotate(-Math.PI/3);g.fillStyle='#687187';g.fillText(x.name,0,0);g.restore()}})}})();
const sel=document.getElementById('industry');R.industries.forEach((x,i)=>sel.add(new Option(x.code+' '+x.name,i)));const pct=x=>Number.isFinite(x)?(x*100).toFixed(2)+'%':'—',num=(x,d=4)=>Number.isFinite(x)?x.toFixed(d):'—';function render(){{const x=R.industries[+sel.value],series=[];let h='<table><thead><tr><th>阶段</th><th>平均股票</th><th>IC</th><th>ICIR</th><th>RankIC</th><th>RankICIR</th><th>扣费多空</th><th>年化</th><th>周数</th></tr></thead><tbody>';['select','valid','test'].forEach((k,i)=>{{const m=x.windows[k];h+=`<tr><td>${{R.stage_labels[k]}}</td><td>${{num(m.average_names,1)}}</td><td>${{num(m.ic.mean)}}</td><td>${{num(m.ic.ir_annual,2)}}</td><td>${{num(m.rank_ic.mean)}}</td><td>${{num(m.rank_ic.ir_annual,2)}}</td><td>${{pct(m.net_ls_total)}}</td><td>${{pct(m.net_ls_annual)}}</td><td>${{m.ls_weeks}}</td></tr>`;series.push({{n:R.stage_labels[k]+'扣费多空',v:[1,...m.net_nav]}})}});document.getElementById('detail').innerHTML=h+'</tbody></table>';line('curves',series)}}sel.onchange=render;render();
</script></main></body></html>""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--minute-parquet", default=str(MINUTE_PARQUET))
    parser.add_argument("--daily-parquet", default=str(ADJUSTED_CLOSE_PARQUET))
    parser.add_argument("--pit", default=str(ZZ500_PIT_PARQUET))
    parser.add_argument("--industry-exposures", default=str(INDUSTRY_VALUE_EXPOSURES_PARQUET))
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--cost-bps", type=float, default=30.0)
    parser.add_argument("--min-ic-names", type=int, default=8)
    parser.add_argument("--min-ls-names", type=int, default=10)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    sidecar = Path(args.candidate)
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    genome = HandbookSkeletonGenome.from_dict(record["genome"])
    direction = int(record["fitness"]["direction"])
    candidate_id = sidecar.name.removesuffix(".parquet.json")
    start, end = WINDOWS[0][2], WINDOWS[-1][3]
    device = "cpu" if args.cpu else "cuda"
    instruments = load_pit_codes(args.pit, start, end)
    dates = load_pit_dates(args.pit, start, end)
    date_index = pd.Index([str(value) for value in dates], name="trade_date")
    instrument_index = pd.Index([str(value) for value in instruments], name="instrument")

    print("[industry-internal] load daily data and industry", flush=True)
    close = load_daily_close_tensor(args.daily_parquet, dates, instruments, device=device)
    pool = load_pit_daily_mask(args.pit, dates, instruments, device=device) & torch.isfinite(close)
    fwd = tensor_rebalance_fwd_ret(close, dates, "week_end", 1)
    fwd = torch.where(pool, fwd, torch.full_like(fwd, float("nan")))
    _styles, industry, levels = load_daily_exposures(
        args.industry_exposures, dates, instruments, continuous_columns=(),
        industry_column="sw_level1", device=device,
    )
    matched_pool = pool & industry.ge(0)

    print("[industry-internal] rebuild full signal", flush=True)
    cached = pd.read_parquet(sidecar.with_suffix(""), columns=["instrument", "trade_date", "factor"])
    cached["trade_date"] = cached["trade_date"].astype(str)
    frame = cached.pivot(index="instrument", columns="trade_date", values="factor")
    frame = frame.reindex(index=instrument_index, columns=date_index)
    raw = torch.as_tensor(frame.to_numpy(copy=True), device=device, dtype=torch.float32)
    raw = trailing_signal_mean(raw, 5)
    warmup_dates = [str(value) for value in dates if str(value) >= "2024-01-02"]
    context, context_meta = build_minute_slice(
        args.minute_parquet, "2024-01-02", end,
        fields=("open", "high", "low", "close", "volume"),
        instruments=instruments, dates=warmup_dates, device=device,
    )
    holdout_dates = np.asarray(context_meta["dates"])
    use = np.flatnonzero(holdout_dates >= WINDOWS[2][2])
    positions = np.array([date_index.get_loc(str(holdout_dates[index])) for index in use])
    holdout = trailing_signal_mean(genome.evaluate(context, chunk_rows=args.chunk_rows), 5)
    raw[:, positions] = holdout[:, use]
    raw = torch.where(matched_pool, raw, torch.full_like(raw, float("nan")))
    neutral = BatchedNeutralizer(
        raw.shape, industry=industry, rank_space=True, min_cross_section=30
    )(raw)
    neutral = torch.where(matched_pool, neutral, torch.full_like(neutral, float("nan")))

    signal_np = neutral.cpu().numpy()
    fwd_np = fwd.cpu().numpy()
    industry_np = industry.cpu().numpy()
    date_array = np.asarray(dates)
    rebalance = np.flatnonzero(np.isfinite(fwd_np).any(axis=0))
    industry_rows = []
    for industry_id, code in enumerate(levels):
        windows = {}
        for stage, _label, window_start, window_end in WINDOWS:
            stage_dates = rebalance[
                (date_array[rebalance] >= window_start) & (date_array[rebalance] <= window_end)
            ]
            windows[stage] = _evaluate_industry(
                signal_np, fwd_np, industry_np, industry_id, stage_dates, direction,
                args.groups, args.cost_bps, args.min_ic_names, args.min_ls_names,
            )
        code_text = str(code)
        industry_rows.append({
            "code": code_text,
            "name": INDUSTRY_NAMES.get(code_text, f"行业代码{code_text}"),
            "windows": windows,
        })
        print(f"[industry-internal] {industry_id + 1}/{len(levels)} {code_text}", flush=True)

    report = _clean({
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "candidate_id": candidate_id,
        "direction": direction,
        "expression": record["expression"],
        "method": "industry-neutralized signal; weekly within-industry IC and Q5-Q1",
        "min_ic_names": args.min_ic_names,
        "min_ls_names": args.min_ls_names,
        "groups": args.groups,
        "cost_bps": args.cost_bps,
        "stage_labels": STAGE_LABELS,
        "industries": industry_rows,
    })
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_html(report, destination)
    destination.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[industry-internal] written -> {destination}", flush=True)


if __name__ == "__main__":
    main()
