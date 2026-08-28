"""Parameter-grid report for the raw candidate_0001_rank0 signal.

Both mean_std_blend nodes are assigned the same trial window.  The resulting
daily factor is then passed through a trailing signal mean.  The factor sign
is kept at the value locked by the original GP run; no neutralisation is
applied.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from min_gp.climb_mountain_html_report import evaluate_period
from min_gp.data import load_pit_codes, load_pit_daily_mask, load_pit_dates
from min_gp.dsl import OperatorNode, evaluate_daily_expressions
from min_gp.evaluation.incremental import trailing_signal_mean
from min_gp.factors.handbook_skeleton import HandbookSkeletonGenome
from min_gp.label import tensor_rebalance_fwd_ret
from min_gp.operators import build_operator_registry
from min_gp.operators.temporal import equal_blend, mean_std_blend
from min_gp.spectral_data import build_minute_slice, load_daily_close_tensor


WINDOWS = (
    ("select", "样本期", "2018-01-02", "2022-03-07"),
    ("valid", "验证期", "2022-03-08", "2024-12-31"),
    ("test", "测试期", "2025-01-02", "2026-07-31"),
)
BASELINE = (20, 5)
MEAN_STD_WINDOWS = (40, 50, 60)
SIGNAL_WINDOWS = (10, 15, 20)


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


def _number(value, digits=4):
    return "—" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def _percent(value, digits=2):
    return "—" if value is None or not np.isfinite(value) else f"{value:.{digits}%}"


def _load_candidate(sidecar: Path):
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    genome = HandbookSkeletonGenome.from_dict(record["genome"])
    if sidecar.name.removesuffix(".parquet.json") != "candidate_0001_rank0":
        raise ValueError(f"expected candidate_0001_rank0, got {sidecar.name}")
    root = genome.root
    if not isinstance(root, OperatorNode) or root.name != "equal_blend":
        raise ValueError("candidate root is not equal_blend")
    if len(root.children) != 2:
        raise ValueError("candidate equal_blend does not have two children")
    for child in root.children:
        if not isinstance(child, OperatorNode) or child.name != "mean_std_blend":
            raise ValueError("candidate children are not both mean_std_blend")
    raw_roots = tuple(child.children[0] for child in root.children)
    return record, genome, raw_roots


def _load_or_build_raw(
    raw_cache: Path,
    raw_roots,
    minute_path,
    instruments,
    dates,
    device,
    chunk_rows,
):
    expected_dates = [str(value) for value in dates]
    expected_instruments = [str(value) for value in instruments]
    if raw_cache.exists():
        cached = torch.load(raw_cache, map_location="cpu", weights_only=False)
        if (
            cached.get("dates") == expected_dates
            and cached.get("instruments") == expected_instruments
            and len(cached.get("components", ())) == 2
        ):
            print(f"[grid] raw component cache -> {raw_cache}", flush=True)
            return tuple(value.to(device) for value in cached["components"])
        print("[grid] raw cache metadata mismatch; rebuilding", flush=True)

    print("[grid] loading minute volume and rebuilding two raw components", flush=True)
    context, meta = build_minute_slice(
        minute_path,
        WINDOWS[0][2],
        WINDOWS[-1][3],
        fields=("volume",),
        instruments=instruments,
        dates=expected_dates,
        device=device,
    )
    actual_dates = [str(value) for value in meta["dates"]]
    if actual_dates != expected_dates:
        raise ValueError(
            f"minute dates do not match PIT dates: {len(actual_dates)} vs {len(expected_dates)}"
        )
    registry = build_operator_registry()
    components = evaluate_daily_expressions(
        raw_roots, context, registry, chunk_rows=chunk_rows
    )
    raw_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "candidate": "candidate_0001_rank0",
            "dates": expected_dates,
            "instruments": expected_instruments,
            "components": tuple(value.detach().cpu() for value in components),
        },
        raw_cache,
    )
    print(f"[grid] cached raw components -> {raw_cache}", flush=True)
    return components


def _variant_id(mean_window, signal_window):
    return f"mean{mean_window}_signal{signal_window}"


def _expression(original, mean_window, signal_window):
    changed = original.replace("mean_std_blend(", "mean_std_blend(")
    changed = changed.replace("window=20)", f"window={mean_window})")
    return f"trailing_signal_mean(({changed}), window={signal_window})"


def _score(row):
    """A descriptive robustness score; never used to choose the factor sign."""
    valid = row["windows"]["valid"]
    test = row["windows"]["test"]
    values = (
        valid["rank_ic"]["mean"], test["rank_ic"]["mean"],
        valid["net_ls_annual"], test["net_ls_annual"],
    )
    if not all(value is not None and np.isfinite(value) for value in values):
        return -float("inf")
    # Rewards consistency: a weak stage cannot be hidden by a very strong one.
    return float(min(values[0], values[1]) + 0.25 * min(values[2], values[3]))


def _build_html(report, destination):
    labels = report["window_labels"]
    rows = []
    for variant in report["variants"]:
        for key, _label, _begin, _finish in WINDOWS:
            m = variant["windows"][key]
            groups = "".join(f"<td>{_percent(value)}</td>" for value in m["group_total"])
            cls = "baseline" if variant["baseline"] else ""
            rows.append(
                f"<tr class='{cls}'><td>{variant['label']}</td><td>{labels[key]}</td>"
                f"<td>{variant['mean_std_window']}</td><td>{variant['signal_window']}</td>"
                f"<td>{_number(m['ic']['mean'])}</td><td>{_number(m['ic']['ir_annual'], 2)}</td>"
                f"<td>{_number(m['rank_ic']['mean'])}</td><td>{_number(m['rank_ic']['ir_annual'], 2)}</td>"
                f"{groups}<td>{_percent(m['gross_ls_total'])}</td>"
                f"<td>{_percent(m['net_ls_total'])}</td><td>{_percent(m['net_ls_annual'])}</td>"
                f"<td>{_percent(m['turnover'], 1)}</td><td>{m['weeks']}</td></tr>"
            )

    test_rows = []
    for variant in report["variants"]:
        m = variant["windows"]["test"]
        delta = variant["comparison_to_baseline"]["test"]
        cls = "baseline" if variant["baseline"] else ""
        test_rows.append(
            f"<tr class='{cls}'><td>{variant['label']}</td><td>{variant['mean_std_window']}</td>"
            f"<td>{variant['signal_window']}</td><td>{_number(m['rank_ic']['mean'])}</td>"
            f"<td>{_number(m['rank_ic']['ir_annual'], 2)}</td><td>{_percent(m['net_ls_total'])}</td>"
            f"<td>{_percent(delta['net_ls_total'])}</td><td>{_percent(m['net_ls_annual'])}</td>"
            f"<td>{_percent(m['turnover'], 1)}</td><td>{_percent(delta['turnover'], 1)}</td></tr>"
        )

    payload = json.dumps(_safe(report), ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    best = report["best_robust_variant"]
    destination.write_text(f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>candidate_0001 参数拉长回测</title>
<style>
:root{{--bg:#f3f5f8;--card:#fff;--ink:#172033;--muted:#697287;--line:#dde3ec;--blue:#315efb;--green:#079669}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1660px;margin:auto;padding:28px}}h1{{margin:0 0 4px}}h2{{font-size:19px;margin:0 0 12px}}h3{{font-size:16px}}
.sub,.note{{color:var(--muted)}}.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin:15px 0;box-shadow:0 2px 12px #1720330a}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.metric{{border:1px solid var(--line);border-radius:9px;padding:12px}}.metric b{{display:block;font-size:19px}}
.scroll{{overflow:auto;max-height:720px}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}th,td{{padding:8px 9px;border-bottom:1px solid var(--line);text-align:right}}th{{position:sticky;top:0;background:#f8f9fc;z-index:1}}th:first-child,td:first-child{{text-align:left}}
.baseline td{{background:#fff9e8}}select{{padding:8px 12px;border:1px solid var(--line);border-radius:7px;background:white}}canvas{{width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:#fff}}code{{word-break:break-all;white-space:normal}}
@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>candidate_0001_rank0：拉长窗口参数回测</h1>
<div class="sub">原始因子（不做行业/风格中性化） · 方向固定为 -1 · 周频调仓 · 五分组 · Q5−Q1 · 双边30bps成本</div>
<section class="card"><h2>参数与阶段</h2><div class="grid">
<div class="metric">基准参数<b>mean 20 / signal 5</b></div><div class="metric">新参数网格<b>3 × 3 = 9组</b></div>
<div class="metric">样本/验证<b>2018-01-02—2024-12-31</b></div><div class="metric">独立测试<b>2025-01-02—2026-07-31</b></div></div>
<p class="note">公式中的两个 mean_std_blend 窗口同步改为40、50、60；最终信号分别取10、15、20日后向均值。方向沿用原GP样本期锁定值，任何参数均未根据验证期或测试期重新翻转方向。</p></section>
<section class="card"><h2>测试期快速对比</h2><div class="grid"><div class="metric">描述性稳健评分最高<b>{best['label']}</b></div>
<div class="metric">测试期 RankIC<b>{_number(best['windows']['test']['rank_ic']['mean'])}</b></div>
<div class="metric">测试期净多空<b>{_percent(best['windows']['test']['net_ls_total'])}</b></div>
<div class="metric">测试期换手<b>{_percent(best['windows']['test']['turnover'], 1)}</b></div></div>
<p class="note">“稳健评分”只用于阅读排序：偏好验证期与测试期同时具有较高 RankIC 和净多空年化，不参与方向选择。</p>
<div class="scroll"><table><thead><tr><th>版本</th><th>MeanStd</th><th>信号均值</th><th>RankIC</th><th>RankICIR</th><th>净多空</th><th>相对基准</th><th>净多空年化</th><th>换手</th><th>换手变化</th></tr></thead><tbody>{''.join(test_rows)}</tbody></table></div></section>
<section class="card"><h2>全部三阶段指标</h2><div class="scroll"><table><thead><tr><th>版本</th><th>阶段</th><th>MeanStd</th><th>信号均值</th><th>IC</th><th>ICIR</th><th>RankIC</th><th>RankICIR</th><th>Q1</th><th>Q2</th><th>Q3</th><th>Q4</th><th>Q5</th><th>毛多空</th><th>净多空</th><th>净多空年化</th><th>换手</th><th>周数</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<section class="card"><h2>参数版本曲线</h2><label>选择版本：<select id="variant"></select></label><div id="detail"></div>
<h3>三阶段五分组累计净值</h3><canvas id="groups" width="1500" height="780"></canvas>
<h3>三阶段净多空累计净值</h3><canvas id="ls" width="1500" height="600"></canvas><p><code id="expr"></code></p></section>
<script id="data" type="application/json">{payload}</script><script>
const R=JSON.parse(document.getElementById('data').textContent),S=document.getElementById('variant'),K=['select','valid','test'];R.variants.forEach((v,i)=>S.add(new Option(v.label,i)));
const pct=x=>Number.isFinite(x)?(x*100).toFixed(2)+'%':'—',num=(x,d=4)=>Number.isFinite(x)?x.toFixed(d):'—',C=['#315efb','#079669','#d97706','#8b5cf6','#df4d4d','#111827'];
function panel(g,series,title,x0,y0,w,h,legend){{const all=series.flatMap(s=>s.v).filter(Number.isFinite);if(!all.length)return;let lo=Math.min(1,...all),hi=Math.max(1,...all);if(hi===lo)hi=lo+1;const L=58,T=38,B=30,X=x0+L,Y=y0+T,W=w-L-20,H=h-T-B;g.strokeStyle='#e1e5ec';for(let z=0;z<=4;z++){{const yy=Y+H*z/4;g.beginPath();g.moveTo(X,yy);g.lineTo(X+W,yy);g.stroke();g.fillStyle='#697287';g.fillText((hi-(hi-lo)*z/4).toFixed(2),x0+5,yy+4)}}g.fillStyle='#172033';g.font='bold 15px system-ui';g.fillText(title,X,y0+20);g.font='12px system-ui';series.forEach((s,j)=>{{g.strokeStyle=C[j%C.length];g.lineWidth=2;g.setLineDash(s.dash||[]);g.beginPath();s.v.forEach((v,i)=>{{const xx=X+i*W/Math.max(1,s.v.length-1),yy=Y+H-(v-lo)*H/(hi-lo);i?g.lineTo(xx,yy):g.moveTo(xx,yy)}});g.stroke();g.setLineDash([]);if(legend){{g.fillStyle=C[j%C.length];g.fillText(s.n,X+j*115,Y-9)}}}})}}
function draw(v){{let c=document.getElementById('groups'),g=c.getContext('2d');g.clearRect(0,0,c.width,c.height);K.forEach((k,i)=>panel(g,v.windows[k].group_nav.map((x,q)=>({{n:'Q'+(q+1),v:[1,...x]}})),R.window_labels[k],10,8+i*255,1480,240,i===0));c=document.getElementById('ls');g=c.getContext('2d');g.clearRect(0,0,c.width,c.height);K.forEach((k,i)=>panel(g,[{{n:'毛多空',v:[1,...v.windows[k].gross_nav],dash:[7,4]}},{{n:'净多空',v:[1,...v.windows[k].net_nav]}}],R.window_labels[k],10,8+i*195,1480,182,i===0))}}
function render(){{const v=R.variants[+S.value];let h='<table><tr><th>阶段</th><th>IC</th><th>ICIR</th><th>RankIC</th><th>RankICIR</th><th>净多空</th><th>年化</th><th>换手</th></tr>';K.forEach(k=>{{const m=v.windows[k];h+=`<tr><td>${{R.window_labels[k]}}</td><td>${{num(m.ic.mean)}}</td><td>${{num(m.ic.ir_annual,2)}}</td><td>${{num(m.rank_ic.mean)}}</td><td>${{num(m.rank_ic.ir_annual,2)}}</td><td>${{pct(m.net_ls_total)}}</td><td>${{pct(m.net_ls_annual)}}</td><td>${{pct(m.turnover,1)}}</td></tr>`}});document.getElementById('detail').innerHTML=h+'</table>';document.getElementById('expr').textContent=v.expression;draw(v)}}S.onchange=render;render();
</script></main></body></html>""", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--minute-parquet", required=True)
    parser.add_argument("--daily-parquet", required=True)
    parser.add_argument("--pit", required=True)
    parser.add_argument("--raw-cache")
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--cost-bps", type=float, default=30.0)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = "cpu" if args.cpu else "cuda"
    record, genome, raw_roots = _load_candidate(Path(args.sidecar))
    direction = int(record["fitness"]["direction"])
    dates = load_pit_dates(args.pit, WINDOWS[0][2], WINDOWS[-1][3])
    instruments = load_pit_codes(args.pit, WINDOWS[0][2], WINDOWS[-1][3])
    date_array = np.asarray([str(value) for value in dates])
    cache_path = Path(args.raw_cache) if args.raw_cache else Path(args.out).with_suffix(".raw_components.pt")
    raw_a, raw_b = _load_or_build_raw(
        cache_path, raw_roots, args.minute_parquet, instruments, dates,
        device, args.chunk_rows,
    )
    close = load_daily_close_tensor(args.daily_parquet, dates, instruments, device=device)
    pool = load_pit_daily_mask(args.pit, dates, instruments, device=device) & torch.isfinite(close)
    fwd = tensor_rebalance_fwd_ret(close, dates, "week_end", 1)
    fwd = torch.where(pool, fwd, torch.full_like(fwd, float("nan")))

    parameters = (BASELINE,) + tuple(
        (mean_window, signal_window)
        for mean_window in MEAN_STD_WINDOWS
        for signal_window in SIGNAL_WINDOWS
    )
    variants = []
    for position, (mean_window, signal_window) in enumerate(parameters, 1):
        print(
            f"[grid] {position}/{len(parameters)} mean={mean_window} signal={signal_window}",
            flush=True,
        )
        daily = equal_blend(
            mean_std_blend(raw_a, mean_window),
            mean_std_blend(raw_b, mean_window),
        )
        factor = trailing_signal_mean(daily, signal_window)
        factor = torch.where(pool, factor, torch.full_like(factor, float("nan")))
        metrics = {}
        for key, _label, begin, finish in WINDOWS:
            mask = torch.as_tensor(
                (date_array >= begin) & (date_array <= finish), device=device
            )
            metrics[key] = evaluate_period(
                factor, fwd, mask, direction, args.groups, args.cost_bps
            )
        variants.append({
            "id": _variant_id(mean_window, signal_window),
            "label": "原始基准 20/5" if (mean_window, signal_window) == BASELINE else f"Mean {mean_window} / Signal {signal_window}",
            "baseline": (mean_window, signal_window) == BASELINE,
            "mean_std_window": mean_window,
            "signal_window": signal_window,
            "direction": direction,
            "expression": _expression(record["expression"], mean_window, signal_window),
            "windows": metrics,
        })
        del daily, factor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    baseline = variants[0]
    for variant in variants:
        comparison = {}
        for key, _label, _begin, _finish in WINDOWS:
            current, base = variant["windows"][key], baseline["windows"][key]
            comparison[key] = {
                "ic_mean": current["ic"]["mean"] - base["ic"]["mean"],
                "rank_ic_mean": current["rank_ic"]["mean"] - base["rank_ic"]["mean"],
                "net_ls_total": current["net_ls_total"] - base["net_ls_total"],
                "net_ls_annual": current["net_ls_annual"] - base["net_ls_annual"],
                "turnover": current["turnover"] - base["turnover"],
            }
        variant["comparison_to_baseline"] = comparison
        variant["robustness_score"] = _score(variant)

    grid_only = [variant for variant in variants if not variant["baseline"]]
    best_robust = max(grid_only, key=lambda row: row["robustness_score"])
    report = {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "candidate_id": "candidate_0001_rank0",
        "original_expression": record["expression"],
        "direction": direction,
        "method": {
            "neutralization": "none",
            "mean_std_nodes_changed_together": 2,
            "mean_std_windows": list(MEAN_STD_WINDOWS),
            "signal_windows": list(SIGNAL_WINDOWS),
            "baseline": {"mean_std_window": 20, "signal_window": 5},
            "rebalance": "week_end",
            "groups": args.groups,
            "spread": "Q5-Q1 after locked direction",
            "cost_bps": args.cost_bps,
            "raw_components_rebuilt_from_minutes": True,
        },
        "window_labels": {key: label for key, label, _begin, _finish in WINDOWS},
        "window_ranges": {key: [begin, finish] for key, _label, begin, finish in WINDOWS},
        "variants": variants,
        "best_robust_variant_id": best_robust["id"],
        "best_robust_variant": best_robust,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _build_html(report, destination)
    destination.with_suffix(".json").write_text(
        json.dumps(_safe(report), ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    print(f"[grid] written -> {destination}", flush=True)


if __name__ == "__main__":
    main()
