"""Three-stage HTML diagnostics for the five best distinct unified-GP factors."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from min_gp.config import (
    ADJUSTED_CLOSE_PARQUET, INDEX_DAILY_PARQUET,
    INDUSTRY_VALUE_EXPOSURES_PARQUET, MINUTE_PARQUET, ZZ500_PIT_PARQUET,
)
from min_gp.data import build_slice, load_pit_codes, load_pit_daily_mask
from min_gp.evaluation.incremental import trailing_signal_mean
from min_gp.factor_leaf_factory import (
    LeafFactoryConfig, build_external_factor_leaves,
)
from min_gp.factors.catalog import genome_from_export
from min_gp.factors.seed_tree import SeedTreeGenome
from min_gp.gp.organs import EXTERNAL_FACTOR_NAMES
from min_gp.label import tensor_rebalance_fwd_ret
from min_gp.numeric.preprocessing import align_signal, neutralize, remove_outliers
from min_gp.numeric.ranking import cross_section_rank
from min_gp.operators import build_operator_registry
from min_gp.spectral_data import load_daily_close_tensor, load_daily_exposures


WINDOWS = (
    ("select", "样本期", "2018-01-02", "2022-12-31"),
    ("valid", "验证期", "2023-01-01", "2024-12-31"),
    ("test", "测试期", "2025-01-02", "2026-07-31"),
)


def _top_distinct(path: Path, count: int):
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    front = sorted(
        (
            record for record in records
            if record.get("pareto_rank") == 0
            and record.get("fitness", {}).get("valid", False)
        ),
        key=lambda record: record["fitness"]["robust_ic"],
        reverse=True,
    )
    output, seen = [], set()
    for record in front:
        expression = record.get("expression", "")
        if expression in seen:
            continue
        seen.add(expression)
        output.append(record)
        if len(output) == count:
            break
    if len(output) < count:
        raise SystemExit(f"only {len(output)} distinct Pareto factors, need {count}")
    return output


def _evaluate(genome, context, registry, chunk_rows):
    if isinstance(genome, SeedTreeGenome):
        return genome.evaluate(context, registry, chunk_rows)
    return genome.evaluate(context, chunk_rows=chunk_rows)


def _corr(left: torch.Tensor, right: torch.Tensor, minimum=30):
    valid = torch.isfinite(left) & torch.isfinite(right)
    count = valid.sum(0)
    weight = valid.float()
    n = count.clamp(min=1).float()
    x, y = torch.nan_to_num(left.float()), torch.nan_to_num(right.float())
    mx = (x * weight).sum(0) / n
    my = (y * weight).sum(0) / n
    dx = (x - mx) * weight
    dy = (y - my) * weight
    covariance = (dx * dy).sum(0) / n
    scale = torch.sqrt((dx.square()).sum(0) / n) * torch.sqrt(
        (dy.square()).sum(0) / n
    )
    result = covariance / scale.clamp(min=1e-12)
    return torch.where(
        count >= minimum, result, torch.full_like(result, float("nan"))
    )


def _stats(values):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return {"mean": None, "ir": None, "ir_annual": None, "n": int(finite.size)}
    mean = float(finite.mean())
    std = float(finite.std(ddof=1))
    ir = mean / std if std > 0 else None
    return {
        "mean": mean,
        "ir": ir,
        "ir_annual": None if ir is None else ir * float(np.sqrt(52.0)),
        "n": int(finite.size),
    }


def _annualized(total, periods):
    if periods <= 0 or not np.isfinite(total) or 1.0 + total <= 0:
        return None
    return float((1.0 + total) ** (52.0 / periods) - 1.0)


def _portfolio(signal, returns, groups=5, cost_bps=30.0, minimum=30):
    eligible = torch.nonzero(
        torch.isfinite(returns).any(0), as_tuple=False
    ).squeeze(1)
    group_returns = [[] for _ in range(groups)]
    gross, net, turnovers, kept = [], [], [], []
    previous = torch.zeros(signal.shape[0], dtype=torch.float32)
    for day in eligible.tolist():
        score, future = signal[:, day].float(), returns[:, day].float()
        valid = torch.isfinite(score) & torch.isfinite(future)
        index = torch.nonzero(valid, as_tuple=False).squeeze(1)
        if index.numel() < max(minimum, groups):
            continue
        ordered = index[torch.argsort(score[index])]
        partitions = torch.tensor_split(ordered, groups)
        values = [float(future[part].mean().item()) for part in partitions]
        for slot, value in enumerate(values):
            group_returns[slot].append(value)
        weights = torch.zeros_like(previous)
        weights[partitions[-1]] = 1.0 / len(partitions[-1])
        weights[partitions[0]] = -1.0 / len(partitions[0])
        spread = float((weights * torch.nan_to_num(future)).sum().item())
        turnover = float((0.5 * (weights - previous).abs().sum()).item())
        gross.append(spread)
        net.append(spread - turnover * cost_bps * 1e-4)
        turnovers.append(turnover)
        kept.append(day)
        previous = weights
    group_array = [np.asarray(item, dtype=np.float64) for item in group_returns]
    gross_array = np.asarray(gross, dtype=np.float64)
    net_array = np.asarray(net, dtype=np.float64)

    def nav(values):
        return np.r_[1.0, np.cumprod(1.0 + values)].tolist()

    group_total = [
        float(np.prod(1.0 + values) - 1.0) if values.size else None
        for values in group_array
    ]
    gross_total = float(np.prod(1.0 + gross_array) - 1.0) if gross_array.size else None
    net_total = float(np.prod(1.0 + net_array) - 1.0) if net_array.size else None
    return {
        "weeks": int(net_array.size),
        "day_indices": kept,
        "group_total": group_total,
        "group_nav": [nav(values) for values in group_array],
        "gross_ls_total": gross_total,
        "net_ls_total": net_total,
        "net_ls_annual": _annualized(net_total, net_array.size),
        "turnover": float(np.mean(turnovers)) if turnovers else None,
        "gross_nav": nav(gross_array),
        "net_nav": nav(net_array),
    }


def _metrics(factor, returns, direction, cost_bps):
    oriented = factor * direction
    ic = _corr(oriented, returns)
    rank_ic = _corr(cross_section_rank(oriented), cross_section_rank(returns))
    portfolio = _portfolio(oriented, returns, cost_bps=cost_bps)
    indices = portfolio.pop("day_indices")
    return {
        "ic": _stats(ic[indices].numpy()),
        "rank_ic": _stats(rank_ic[indices].numpy()),
        **portfolio,
    }


def _stage(
    key, begin, finish, genomes, registry, minute_path, daily_path, pit,
    device, chunk_rows, signal_days, align, neutralization, exposures_path,
):
    instruments = load_pit_codes(pit, begin, finish)
    tensors, masks, _label, meta = build_slice(
        minute_path, begin, finish, instruments=instruments,
        device=device, extend_days=45,
    )
    # build_slice returns the session masks on the CPU whatever device the
    # minute panels land on, so they have to be moved explicitly - seed_tree_gp
    # does the same. Without this a factor that selects a session mask dies in
    # _mask_mul with a cuda/cpu mismatch, which stayed hidden only while no
    # mined factor happened to use one.
    context = {
        **tensors,
        **{name: value.to(device) for name, value in masks.items()},
    }
    dates = [str(value) for value in meta["dates"]]
    instruments = [str(value) for value in meta["instruments"]]
    close = load_daily_close_tensor(daily_path, dates, instruments, device=device)
    pool = load_pit_daily_mask(pit, dates, instruments, device=device)
    pool &= torch.isfinite(close)
    # seed_tree_gp derives eleven further terminals (daily_close, volume_share,
    # price_state, ...) and installs them into its own context unless
    # --no-external-organs is passed, and that flag defaults to off. Factors
    # mined against those terminals cannot be rebuilt from build_slice alone,
    # so this report has to derive them the same way or it fails on more than
    # half the population with "missing leaves".
    leaf_report = build_external_factor_leaves(
        context, close, pool, dates, instruments, EXTERNAL_FACTOR_NAMES,
        LeafFactoryConfig(
            market_parquet=str(INDEX_DAILY_PARQUET),
            exposures_parquet=exposures_path,
            cache_directory=str(Path("output") / "leaf_cache"),
        ),
    )
    built = sorted(leaf_report.get("built", {}))
    print(
        f"[top5-report] {key} derived leaves: {len(built)} "
        f"({', '.join(built) if built else 'none'})",
        flush=True,
    )
    fwd = tensor_rebalance_fwd_ret(close, dates, "week_end", 1)
    fwd = torch.where(pool, fwd, torch.full_like(fwd, float("nan")))
    target = torch.as_tensor(
        (np.asarray(dates) >= begin) & (np.asarray(dates) <= finish),
        device=device,
    )
    industry, levels, styles = None, None, ()
    if neutralization != "none":
        continuous, industry, industry_levels = load_daily_exposures(
            exposures_path, dates, instruments, device=device,
        )
        levels = len(industry_levels)
        styles = tuple(continuous.values())
        if neutralization == "industry":
            styles = ()
        elif neutralization == "market_cap":
            industry, levels = None, None
        covered = float((industry >= 0).float().mean().item()) if industry is not None else float("nan")
        print(
            f"[top5-report] {key} exposures: {levels or 0} industries, "
            f"{len(styles)} style column(s), industry coverage {covered:.3f}",
            flush=True,
        )
    results = []
    for index, genome in enumerate(genomes, 1):
        required = set(genome.required_fields)
        missing = sorted(required - set(context))
        if missing:
            raise RuntimeError(f"factor {index} missing leaves: {missing}")
        factor = _evaluate(genome, context, registry, chunk_rows).float()
        factor = trailing_signal_mean(factor, signal_days)
        # Delay before every cross-sectional step: on the day the signal is
        # actually traded it must be screened, winsorized and neutralised
        # against that day's universe and exposures, not the day it was formed.
        factor = align_signal(factor, align)
        factor = torch.where(pool, factor, torch.full_like(factor, float("nan")))
        factor = remove_outliers(factor, n_mad=5.0, dim=0)
        if neutralization != "none":
            factor = neutralize(
                factor, industry=industry, levels=levels, continuous=styles,
            )
        results.append(factor[:, target].cpu())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[top5-report] {key} factor {index}/{len(genomes)}", flush=True)
    scoped_dates = [day for day in dates if begin <= day <= finish]
    scoped_fwd = fwd[:, target].cpu()
    del tensors, masks, context, close, pool, fwd
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Instruments travel with the panels: the PIT universe differs between
    # windows (892 names in 2018-2022 against 733 in 2023-2024), so anything
    # carrying a book across a window boundary has to realign it by code
    # rather than by row position.
    return {
        "dates": scoped_dates, "factors": results, "returns": scoped_fwd,
        "instruments": instruments,
    }


def _safe(value):
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _fmt(value, digits=4):
    return "—" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def _pct(value, digits=2):
    return "—" if value is None or not np.isfinite(value) else f"{value:.{digits}%}"


def _html(report, path: Path):
    neutral_label = {
        "both": "行业 + 对数市值", "industry": "仅行业",
        "market_cap": "仅对数市值", "none": "未中性化",
    }[report["neutralization"]]
    align_label = (
        "收盘后移一日" if report["align"] == "close" else "不后移（当日收盘成交）"
    )
    align_note = (
        "因子在 t 日收盘算出后整体后移一日，以 t+1 起的收益计分，"
        "确保收盘形成的信号不会用当根收盘价成交。"
        if report["align"] == "close" else
        "因子在形成当日即计分，隐含尾盘以收盘价成交的假设。"
    )
    neutral_note = (
        "" if report["neutralization"] == "none" else
        "去极值后按点时申万一级行业哑变量与对数流通市值做逐日截面 OLS"
        "（pinv 解正规方程），以残差替换原因子；行业或市值缺失的个股当日剔除。"
    )
    rows = []
    for factor in report["factors"]:
        for key, label, _begin, _finish in WINDOWS:
            metric = factor["windows"][key]
            groups = "".join(f"<td>{_pct(value)}</td>" for value in metric["group_total"])
            rows.append(
                f"<tr><td>{factor['id']}</td><td>{label}</td>"
                f"<td>{factor['direction']:+d}</td>"
                f"<td>{_fmt(metric['ic']['mean'])}</td>"
                f"<td>{_fmt(metric['ic']['ir_annual'], 2)}</td>"
                f"<td>{_fmt(metric['rank_ic']['mean'])}</td>"
                f"<td>{_fmt(metric['rank_ic']['ir_annual'], 2)}</td>"
                f"{groups}<td>{_pct(metric['gross_ls_total'])}</td>"
                f"<td>{_pct(metric['net_ls_total'])}</td>"
                f"<td>{_pct(metric['net_ls_annual'])}</td>"
                f"<td>{_pct(metric['turnover'], 1)}</td><td>{metric['weeks']}</td></tr>"
            )
    payload = json.dumps(
        _safe(report), ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).replace("</", "<\\/")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>统一GP前五因子三阶段绩效</title>
<style>
:root{{--bg:#f3f5f8;--card:#fff;--ink:#182033;--muted:#687287;--line:#dce3ed;--blue:#315efb}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1580px;margin:auto;padding:28px}}h1{{margin:0}}h2{{font-size:19px;margin:0 0 12px}}h3{{font-size:16px}}.sub,.note{{color:var(--muted)}}.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin:15px 0;box-shadow:0 2px 12px #1520380a}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.metric{{border:1px solid var(--line);border-radius:9px;padding:12px}}.metric b{{display:block;font-size:20px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}th,td{{padding:8px 9px;border-bottom:1px solid var(--line);text-align:right}}th{{background:#f8f9fc;position:sticky;top:0}}th:first-child,td:first-child{{text-align:left}}select,button{{padding:8px 12px;border:1px solid var(--line);border-radius:7px;background:#fff;margin-right:8px}}canvas{{width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:#fff}}code{{display:block;white-space:pre-wrap;word-break:break-all;background:#f7f8fb;padding:12px;border-radius:8px}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>统一GP：前五个不同 Pareto 因子三阶段绩效</h1>
<div class="sub">MAD5 + {neutral_label}中性化 · {align_label} · 周频调仓 · 5日信号均值 · 五分组 · Q5−Q1 · 30bps成本</div>
<section class="card"><h2>评估口径</h2><div class="grid"><div class="metric">样本期<b>2018–2022</b></div><div class="metric">验证期<b>2023–2024</b></div><div class="metric">测试期<b>2025–2026-07</b></div><div class="metric">去极值<b>中位数 ±5×MAD</b></div><div class="metric">中性化<b>{neutral_label}</b></div><div class="metric">信号对齐<b>{align_label}</b></div><div class="metric">回归解法<b>pinv 正规方程</b></div><div class="metric">行业口径<b>点时申万一级</b></div></div><p class="note">前五名只按2018–2022走步稳健RankIC选择；每个因子的方向沿用训练搜索结果，验证期与测试期不重新选方向。普通IC为Pearson，RankIC为Spearman；ICIR按周频年化，即 mean/std×√52。仅因子值预处理，未来收益不处理。{align_note}{neutral_note}</p></section>
<section class="card"><h2>全部指标</h2><div class="scroll"><table><thead><tr><th>因子</th><th>阶段</th><th>方向</th><th>IC</th><th>ICIR</th><th>RankIC</th><th>RankICIR</th><th>Q1</th><th>Q2</th><th>Q3</th><th>Q4</th><th>Q5</th><th>毛多空</th><th>扣费多空</th><th>扣费年化</th><th>换手</th><th>周数</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<section class="card"><h2>单因子详情与收益曲线</h2><select id="factor"></select><button id="save">下载曲线PNG</button><div id="detail"></div><h3>三个时期的五分组累计净值</h3><canvas id="groups" width="1480" height="780"></canvas><h3>三个时期的扣费多空累计净值</h3><canvas id="ls" width="1480" height="380"></canvas><code id="expr"></code></section>
<script id="payload" type="application/json">{payload}</script><script>
const R=JSON.parse(document.getElementById('payload').textContent),K=['select','valid','test'],S=document.getElementById('factor'),COL=['#315efb','#079669','#d97706','#8b5cf6','#df4d4d'];R.factors.forEach((f,i)=>S.add(new Option(f.id+'  训练稳健RankIC '+f.train_fitness.robust_ic.toFixed(4),i)));const num=(x,d=4)=>Number.isFinite(x)?x.toFixed(d):'—',pct=x=>Number.isFinite(x)?(100*x).toFixed(2)+'%':'—';
function chart(canvas,series,title){{const g=canvas.getContext('2d'),W=canvas.width,H=canvas.height,p={{l:65,r:25,t:42,b:38}};g.clearRect(0,0,W,H);let all=series.flatMap(s=>s.v).filter(Number.isFinite),lo=Math.min(1,...all),hi=Math.max(1,...all);if(lo===hi)hi=lo+.01;for(let j=0;j<=5;j++){{let y=p.t+(H-p.t-p.b)*j/5;g.strokeStyle='#e0e5ed';g.beginPath();g.moveTo(p.l,y);g.lineTo(W-p.r,y);g.stroke();g.fillStyle='#697287';g.fillText((hi-(hi-lo)*j/5).toFixed(2),8,y+4)}}g.fillStyle='#182033';g.font='bold 16px system-ui';g.fillText(title,p.l,24);series.forEach((s,j)=>{{g.strokeStyle=s.c;g.setLineDash(s.d||[]);g.lineWidth=2;g.beginPath();s.v.forEach((v,i)=>{{let x=p.l+i*(W-p.l-p.r)/Math.max(1,s.v.length-1),y=p.t+(hi-v)*(H-p.t-p.b)/(hi-lo);i?g.lineTo(x,y):g.moveTo(x,y)}});g.stroke();g.setLineDash([])}});let x=p.l;series.forEach(s=>{{g.fillStyle=s.c;g.fillRect(x,H-20,16,3);g.fillStyle='#465168';g.fillText(s.n,x+21,H-15);x+=Math.max(115,s.n.length*13)}})}}
function render(){{const f=R.factors[+S.value];let h='<div class="scroll"><table><tr><th>阶段</th><th>IC</th><th>ICIR</th><th>RankIC</th><th>RankICIR</th><th>扣费多空</th><th>年化</th><th>换手</th></tr>';K.forEach(k=>{{const m=f.windows[k];h+=`<tr><td>${{R.window_labels[k]}}</td><td>${{num(m.ic.mean)}}</td><td>${{num(m.ic.ir_annual,2)}}</td><td>${{num(m.rank_ic.mean)}}</td><td>${{num(m.rank_ic.ir_annual,2)}}</td><td>${{pct(m.net_ls_total)}}</td><td>${{pct(m.net_ls_annual)}}</td><td>${{pct(m.turnover)}}</td></tr>`}});document.getElementById('detail').innerHTML=h+'</table></div>';document.getElementById('expr').textContent=f.expression;let gs=[];K.forEach((k,ki)=>f.windows[k].group_nav.forEach((v,qi)=>gs.push({{n:R.window_labels[k]+' Q'+(qi+1),v,c:COL[qi],d:ki===0?[]:ki===1?[8,4]:[2,4]}})));chart(document.getElementById('groups'),gs,'五分组累计净值（实线样本期、长虚线验证期、点线测试期）');chart(document.getElementById('ls'),K.map((k,i)=>({{n:R.window_labels[k],v:f.windows[k].net_nav,c:COL[i],d:[]}})),'扣费多空累计净值（各阶段从1重新起算）')}}S.onchange=render;document.getElementById('save').onclick=()=>{{const a=document.createElement('a');a.download=R.factors[+S.value].id+'_performance.png';a.href=document.getElementById('groups').toDataURL('image/png');a.click()}};render();
</script></main></body></html>""", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--minute-parquet", default=str(MINUTE_PARQUET))
    parser.add_argument("--daily-parquet", default=str(ADJUSTED_CLOSE_PARQUET))
    parser.add_argument("--pit", default=str(ZZ500_PIT_PARQUET))
    parser.add_argument("--exposures", default=str(INDUSTRY_VALUE_EXPOSURES_PARQUET),
                        help="point-in-time sw_level1 + ln_float_market_cap grid")
    parser.add_argument(
        "--neutralization", choices=("both", "industry", "market_cap", "none"),
        default="both",
        help="daily cross-sectional OLS residual on the chosen exposures",
    )
    parser.add_argument(
        "--align", choices=("close", "none"), default="close",
        help="close: delay the signal one day before it is scored",
    )
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--signal-days", type=int, default=5)
    parser.add_argument("--cost-bps", type=float, default=30.0)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)
    device = "cpu" if args.cpu else "cuda"
    registry = build_operator_registry()
    records = _top_distinct(Path(args.jsonl), args.count)
    genomes = [genome_from_export(record["genome"], registry) for record in records]
    stages = {}
    for key, _label, begin, finish in WINDOWS:
        print(f"[top5-report] build {key} {begin}..{finish}", flush=True)
        stages[key] = _stage(
            key, begin, finish, genomes, registry, args.minute_parquet,
            args.daily_parquet, args.pit, device, args.chunk_rows,
            args.signal_days, args.align, args.neutralization, args.exposures,
        )
    factors = []
    for index, record in enumerate(records):
        direction = int(record["fitness"]["direction"])
        windows = {
            key: _metrics(
                stages[key]["factors"][index], stages[key]["returns"],
                direction, args.cost_bps,
            )
            for key, *_rest in WINDOWS
        }
        factors.append({
            "id": f"factor_{index + 1:02d}",
            "direction": direction,
            "expression": record["expression"],
            "train_fitness": record["fitness"],
            "windows": windows,
        })
    report = {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": str(args.jsonl),
        "selection": "top five distinct Pareto-rank-0 expressions by 2018-2022 robust RankIC",
        "preprocessing": "daily cross-sectional median +/- 5 * raw MAD",
        "neutralization": args.neutralization,
        "neutralization_detail": (
            "none" if args.neutralization == "none" else
            "daily cross-sectional OLS residual (pinv) on "
            + {
                "both": "sw_level1 one-hot + ln_float_market_cap",
                "industry": "sw_level1 one-hot",
                "market_cap": "intercept + ln_float_market_cap",
            }[args.neutralization]
        ),
        "align": args.align,
        "align_detail": (
            "signal delayed one trading day before scoring"
            if args.align == "close" else "signal scored on its formation date"
        ),
        "signal_average_days": args.signal_days,
        "cost_bps": args.cost_bps,
        "window_labels": {key: label for key, label, _b, _e in WINDOWS},
        "window_ranges": {key: [begin, finish] for key, _l, begin, finish in WINDOWS},
        "factors": factors,
    }
    destination = Path(args.out)
    _html(report, destination)
    destination.with_suffix(".json").write_text(
        json.dumps(_safe(report), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"[top5-report] wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
