"""Diagnose the even entropy-distance branch and test signed replacements."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from min_gp.climb_mountain_html_report import evaluate_period
from min_gp.data import load_pit_codes, load_pit_daily_mask, load_pit_dates
from min_gp.dsl import OperatorNode, evaluate_daily_expression
from min_gp.evaluation.incremental import trailing_signal_mean
from min_gp.evaluation.neutralize import BatchedNeutralizer
from min_gp.factors.handbook_skeleton import HandbookSkeletonGenome
from min_gp.label import tensor_rebalance_fwd_ret
from min_gp.numeric.ranking import cross_section_rank
from min_gp.operators import build_operator_registry
from min_gp.operators.temporal import equal_blend, mean_std_blend
from min_gp.spectral_data import build_minute_slice, load_daily_close_tensor, load_daily_exposures


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


def _load_entropy(
    entropy_cache, sidecar, minute_path, instruments, dates, device, chunk_rows
):
    expected_dates = [str(value) for value in dates]
    expected_instruments = [str(value) for value in instruments]
    if entropy_cache.exists():
        cached = torch.load(entropy_cache, map_location="cpu", weights_only=False)
        if (
            cached.get("dates") == expected_dates
            and cached.get("instruments") == expected_instruments
        ):
            print(f"[entropy] cache -> {entropy_cache}", flush=True)
            return cached["entropy"].to(device)

    record = json.loads(Path(sidecar).read_text(encoding="utf-8"))
    genome = HandbookSkeletonGenome.from_dict(record["genome"])
    root = genome.root
    if not isinstance(root, OperatorNode) or root.name != "equal_blend":
        raise ValueError("unexpected candidate root")
    distance = root.children[0].children[0]
    if not isinstance(distance, OperatorNode) or distance.name != "cross_section_distance":
        raise ValueError("first branch is not cross_section_distance")
    entropy_root = distance.children[0]
    print("[entropy] loading minute volume and rebuilding unsigned entropy", flush=True)
    context, meta = build_minute_slice(
        minute_path,
        WINDOWS[0][2],
        WINDOWS[-1][3],
        fields=("volume",),
        instruments=instruments,
        dates=expected_dates,
        device=device,
    )
    if [str(value) for value in meta["dates"]] != expected_dates:
        raise ValueError("minute dates do not align")
    entropy = evaluate_daily_expression(
        entropy_root, context, build_operator_registry(), chunk_rows=chunk_rows
    )
    entropy_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "dates": expected_dates,
            "instruments": expected_instruments,
            "entropy": entropy.detach().cpu(),
        },
        entropy_cache,
    )
    print(f"[entropy] cached -> {entropy_cache}", flush=True)
    return entropy


def _robust_components(entropy):
    valid = torch.isfinite(entropy)
    median = torch.nanmedian(entropy, dim=0, keepdim=True).values
    dev = entropy - median
    pos = torch.clamp(dev, min=0)
    neg = torch.clamp(-dev, min=0)
    mad = torch.nanmedian(torch.abs(dev), dim=0, keepdim=True).values
    scale = 1.4826 * mad
    robust_z = dev / scale.clamp(min=1e-12)
    robust_z = torch.where(
        valid & torch.isfinite(scale) & scale.gt(1e-12),
        robust_z,
        torch.full_like(robust_z, float("nan")),
    )
    percentile = cross_section_rank(entropy.float())
    pos = torch.where(valid, pos, torch.full_like(pos, float("nan")))
    neg = torch.where(valid, neg, torch.full_like(neg, float("nan")))
    return pos, neg, robust_z, percentile


def _smoothed_branch(value):
    return trailing_signal_mean(mean_std_blend(value, 60), 20)


def _full_factor(first, second):
    return trailing_signal_mean(
        equal_blend(mean_std_blend(first, 60), mean_std_blend(second, 60)), 20
    )


def _locked_direction(factor, fwd, sample_mask):
    raw = evaluate_period(factor, fwd, sample_mask, 1, 5, 30.0)
    return 1 if raw["ic"]["mean"] >= 0 else -1


def _stability(windows):
    values = [windows[key]["rank_ic"]["ir_annual"] for key, *_ in WINDOWS]
    return float(min(values)) if all(np.isfinite(value) for value in values) else -float("inf")


def _build_html(report, destination):
    labels = report["window_labels"]
    diagnostic_rows = []
    for diagnostic in report["diagnostics"]:
        for key, _label, _begin, _finish in WINDOWS:
            m = diagnostic["windows"][key]
            diagnostic_rows.append(
                f"<tr><td>{diagnostic['label']}</td><td>{labels[key]}</td>"
                f"<td>{_fmt(m['ic']['mean'])}</td><td>{_fmt(m['ic']['ir_annual'], 2)}</td>"
                f"<td>{_fmt(m['rank_ic']['mean'])}</td><td>{_fmt(m['rank_ic']['ir_annual'], 2)}</td>"
                f"<td>{_pct(m['gross_ls_total'])}</td><td>{_pct(m['net_ls_total'])}</td></tr>"
            )

    summary_rows = []
    detail_rows = []
    for variant in report["variants"]:
        for mode, mode_label in (("raw", "未中性"), ("industry_neutral", "行业中性")):
            windows = variant[mode]["windows"]
            test = windows["test"]
            summary_rows.append(
                f"<tr><td>{variant['label']}</td><td>{mode_label}</td><td>{variant[mode]['direction']:+d}</td>"
                f"<td>{_fmt(windows['select']['rank_ic']['ir_annual'], 2)}</td>"
                f"<td>{_fmt(windows['valid']['rank_ic']['ir_annual'], 2)}</td>"
                f"<td>{_fmt(windows['test']['rank_ic']['ir_annual'], 2)}</td>"
                f"<td>{_fmt(variant[mode]['rank_icir_stability'], 2)}</td>"
                f"<td>{_fmt(test['ic']['mean'])}</td><td>{_fmt(test['rank_ic']['mean'])}</td>"
                f"<td>{_pct(test['gross_ls_total'])}</td><td>{_pct(test['net_ls_total'])}</td>"
                f"<td>{_pct(test['turnover'], 1)}</td></tr>"
            )
            for key, _label, _begin, _finish in WINDOWS:
                m = windows[key]
                detail_rows.append(
                    f"<tr><td>{variant['label']}</td><td>{mode_label}</td><td>{labels[key]}</td>"
                    f"<td>{variant[mode]['direction']:+d}</td><td>{_fmt(m['ic']['mean'])}</td>"
                    f"<td>{_fmt(m['ic']['ir_annual'], 2)}</td><td>{_fmt(m['rank_ic']['mean'])}</td>"
                    f"<td>{_fmt(m['rank_ic']['ir_annual'], 2)}</td><td>{_pct(m['net_ls_total'])}</td>"
                    f"<td>{_pct(m['net_ls_annual'])}</td><td>{_pct(m['turnover'], 1)}</td></tr>"
                )

    payload = json.dumps(_safe(report), ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    raw_best, neutral_best = report["best_raw"], report["best_industry_neutral"]
    destination.write_text(f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>candidate_0001 成交量熵符号诊断</title>
<style>
:root{{--bg:#f3f5f8;--card:#fff;--ink:#172033;--muted:#687187;--line:#dde3ec;--green:#087f5b}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1700px;margin:auto;padding:28px}}h1{{margin:0 0 5px}}h2{{font-size:19px;margin:0 0 12px}}
.sub,.note{{color:var(--muted)}}.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin:15px 0;box-shadow:0 2px 12px #1720330a}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.metric{{border:1px solid var(--line);border-radius:9px;padding:12px}}.metric b{{display:block;font-size:19px}}
.scroll{{overflow:auto;max-height:720px}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}th,td{{padding:8px 9px;border-bottom:1px solid var(--line);text-align:right}}th{{position:sticky;top:0;background:#f8f9fc}}th:first-child,td:first-child{{text-align:left}}select{{padding:8px 12px;border:1px solid var(--line);border-radius:7px;background:#fff}}canvas{{width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:#fff}}code{{word-break:break-all;white-space:normal}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>candidate_0001_rank0：成交量熵偶函数诊断与替换</h1>
<div class="sub">源码确认 cross_section_distance(x, False)=|x−mean| · 60日MeanStd · 20日信号均值 · 方向仅由样本期锁定 · 周频五分组 · 双边30bps</div>
<section class="card"><h2>实验设计</h2><div class="grid"><div class="metric">截面中心<b>每日中位数</b></div><div class="metric">稳健标准化<b>1.4826 × MAD</b></div><div class="metric">排名替换<b>每日百分位秩</b></div><div class="metric">行业控制<b>点时申万一级</b></div></div>
<p class="note">诊断表中的IC不翻转方向：正数表示该侧偏离越强，未来收益越高；负数表示未来收益越低。完整候选版本则分别用自己的样本期锁定方向。</p></section>
<section class="card"><h2>两侧拆分诊断：原始IC符号</h2><div class="scroll"><table><thead><tr><th>诊断信号</th><th>阶段</th><th>原始IC</th><th>原始ICIR</th><th>原始RankIC</th><th>原始RankICIR</th><th>毛多空</th><th>净多空</th></tr></thead><tbody>{''.join(diagnostic_rows)}</tbody></table></div></section>
<section class="card"><h2>替换方案稳定性对比</h2><div class="grid"><div class="metric">未中性最稳健<b>{raw_best['label']}</b><span>最低三期RankICIR {_fmt(raw_best['stability'],2)}</span></div><div class="metric">行业中性最稳健<b>{neutral_best['label']}</b><span>最低三期RankICIR {_fmt(neutral_best['stability'],2)}</span></div></div>
<div class="scroll"><table><thead><tr><th>第一分量</th><th>模式</th><th>方向</th><th>样本RankICIR</th><th>验证RankICIR</th><th>测试RankICIR</th><th>最低RankICIR</th><th>测试IC</th><th>测试RankIC</th><th>测试毛多空</th><th>测试净多空</th><th>测试换手</th></tr></thead><tbody>{''.join(summary_rows)}</tbody></table></div></section>
<section class="card"><h2>三阶段完整指标</h2><div class="scroll"><table><thead><tr><th>第一分量</th><th>模式</th><th>阶段</th><th>方向</th><th>IC</th><th>ICIR</th><th>RankIC</th><th>RankICIR</th><th>净多空</th><th>净多空年化</th><th>换手</th></tr></thead><tbody>{''.join(detail_rows)}</tbody></table></div></section>
<section class="card"><h2>候选版本曲线</h2><select id="variant"></select><h3>行业中性化后三阶段五分组</h3><canvas id="groups" width="1500" height="780"></canvas><h3>未中性与行业中性扣费多空</h3><canvas id="ls" width="1500" height="600"></canvas></section>
<script id="data" type="application/json">{payload}</script><script>
const R=JSON.parse(document.getElementById('data').textContent),S=document.getElementById('variant'),K=['select','valid','test'],C=['#315efb','#079669','#d97706','#8b5cf6','#df4d4d'];R.variants.forEach((v,i)=>S.add(new Option(v.label,i)));
function panel(g,series,title,x0,y0,w,h,legend){{const all=series.flatMap(s=>s.v).filter(Number.isFinite);if(!all.length)return;let lo=Math.min(1,...all),hi=Math.max(1,...all);if(hi===lo)hi=lo+1;const L=58,T=38,B=30,X=x0+L,Y=y0+T,W=w-L-20,H=h-T-B;g.strokeStyle='#e1e5ec';for(let z=0;z<=4;z++){{const yy=Y+H*z/4;g.beginPath();g.moveTo(X,yy);g.lineTo(X+W,yy);g.stroke();g.fillStyle='#687187';g.fillText((hi-(hi-lo)*z/4).toFixed(2),x0+5,yy+4)}}g.fillStyle='#172033';g.font='bold 15px system-ui';g.fillText(title,X,y0+20);g.font='12px system-ui';series.forEach((s,j)=>{{g.strokeStyle=s.color||C[j%C.length];g.lineWidth=s.width||2;g.setLineDash(s.dash||[]);g.beginPath();s.v.forEach((v,i)=>{{const xx=X+i*W/Math.max(1,s.v.length-1),yy=Y+H-(v-lo)*H/(hi-lo);i?g.lineTo(xx,yy):g.moveTo(xx,yy)}});g.stroke();g.setLineDash([]);if(legend){{g.fillStyle=g.strokeStyle;g.fillText(s.n,X+j*180,Y-9)}}}})}}
function render(){{const v=R.variants[+S.value];let c=document.getElementById('groups'),g=c.getContext('2d');g.clearRect(0,0,c.width,c.height);K.forEach((k,i)=>{{const m=v.industry_neutral.windows[k];panel(g,m.group_nav.map((x,q)=>({{n:'Q'+(q+1),v:[1,...x]}})),R.window_labels[k],10,8+i*255,1480,240,i===0)}});c=document.getElementById('ls');g=c.getContext('2d');g.clearRect(0,0,c.width,c.height);K.forEach((k,i)=>panel(g,[{{n:'未中性',v:[1,...v.raw.windows[k].net_nav],color:'#315efb',dash:[8,4]}},{{n:'行业中性',v:[1,...v.industry_neutral.windows[k].net_nav],color:'#079669',width:2.5}}],R.window_labels[k],10,8+i*195,1480,182,i===0))}}S.onchange=render;render();
</script></main></body></html>""", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--raw-cache", required=True)
    parser.add_argument("--entropy-cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--minute-parquet", required=True)
    parser.add_argument("--daily-parquet", required=True)
    parser.add_argument("--pit", required=True)
    parser.add_argument("--industry-exposures", required=True)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--cost-bps", type=float, default=30.0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    device = "cpu" if args.cpu else "cuda"

    dates = load_pit_dates(args.pit, WINDOWS[0][2], WINDOWS[-1][3])
    instruments = load_pit_codes(args.pit, WINDOWS[0][2], WINDOWS[-1][3])
    date_array = np.asarray([str(value) for value in dates])
    entropy = _load_entropy(
        Path(args.entropy_cache), args.sidecar, args.minute_parquet,
        instruments, dates, device, args.chunk_rows,
    )
    component_cache = torch.load(args.raw_cache, map_location="cpu", weights_only=False)
    if component_cache["dates"] != [str(value) for value in dates]:
        raise ValueError("raw component dates do not align")
    distance, second = (value.to(device) for value in component_cache["components"])
    pos, neg, robust_z, percentile = _robust_components(entropy)

    close = load_daily_close_tensor(args.daily_parquet, dates, instruments, device=device)
    pool = load_pit_daily_mask(args.pit, dates, instruments, device=device) & torch.isfinite(close)
    fwd = tensor_rebalance_fwd_ret(close, dates, "week_end", 1)
    fwd = torch.where(pool, fwd, torch.full_like(fwd, float("nan")))
    _styles, industry, industry_names = load_daily_exposures(
        args.industry_exposures, dates, instruments, continuous_columns=(),
        industry_column="sw_level1", device=device,
    )
    matched_pool = pool & industry.ge(0)
    neutralizer = BatchedNeutralizer(
        pool.shape, industry=industry, rank_space=True, min_cross_section=30
    )
    stage_masks = {
        key: torch.as_tensor((date_array >= begin) & (date_array <= finish), device=device)
        for key, _label, begin, finish in WINDOWS
    }

    print("[entropy] evaluating high/low-side diagnostics", flush=True)
    diagnostic_factors = (
        ("pos_direct", "高熵侧：直接偏离", pos),
        ("neg_direct", "低熵侧：直接偏离", neg),
        ("pos_60_20", "高熵侧：60/20处理", _smoothed_branch(pos)),
        ("neg_60_20", "低熵侧：60/20处理", _smoothed_branch(neg)),
    )
    diagnostics = []
    for identifier, label, factor in diagnostic_factors:
        factor = torch.where(pool, factor, torch.full_like(factor, float("nan")))
        diagnostics.append({
            "id": identifier,
            "label": label,
            "direction": 1,
            "windows": {
                key: evaluate_period(
                    factor, fwd, stage_masks[key], 1, args.groups, args.cost_bps
                )
                for key, *_ in WINDOWS
            },
        })

    first_components = (
        ("distance_abs_mean", "原始绝对均值距离", distance),
        ("median_mad_z", "稳健MAD-Z（方案B）", robust_z),
        ("percentile_rank", "截面百分位排名（方案C）", percentile),
        ("positive_only", "仅高熵侧", pos),
        ("negative_only", "仅低熵侧", neg),
    )
    variants = []
    sample_mask = stage_masks["select"]
    for position, (identifier, label, first) in enumerate(first_components, 1):
        print(f"[entropy] full variant {position}/{len(first_components)} {identifier}", flush=True)
        raw_factor = _full_factor(first, second)
        raw_factor = torch.where(
            matched_pool, raw_factor, torch.full_like(raw_factor, float("nan"))
        )
        neutral_factor = neutralizer(raw_factor)
        neutral_factor = torch.where(
            matched_pool, neutral_factor, torch.full_like(neutral_factor, float("nan"))
        )
        raw_direction = _locked_direction(raw_factor, fwd, sample_mask)
        neutral_direction = _locked_direction(neutral_factor, fwd, sample_mask)
        raw_windows = {
            key: evaluate_period(
                raw_factor, fwd, stage_masks[key], raw_direction,
                args.groups, args.cost_bps,
            )
            for key, *_ in WINDOWS
        }
        neutral_windows = {
            key: evaluate_period(
                neutral_factor, fwd, stage_masks[key], neutral_direction,
                args.groups, args.cost_bps,
            )
            for key, *_ in WINDOWS
        }
        variants.append({
            "id": identifier,
            "label": label,
            "raw": {
                "direction": raw_direction,
                "rank_icir_stability": _stability(raw_windows),
                "windows": raw_windows,
            },
            "industry_neutral": {
                "direction": neutral_direction,
                "rank_icir_stability": _stability(neutral_windows),
                "windows": neutral_windows,
            },
        })

    best_raw = max(variants, key=lambda row: row["raw"]["rank_icir_stability"])
    best_neutral = max(
        variants, key=lambda row: row["industry_neutral"]["rank_icir_stability"]
    )
    report = {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "candidate_id": "candidate_0001_rank0",
        "source_confirmation": {
            "operator": "cross_section_distance",
            "standardize": False,
            "implementation": "abs(x - daily_cross_section_mean)",
            "even_function": True,
        },
        "method": {
            "tensor_axes": "instrument x date; cross-section is dim=0",
            "diagnostic_center": "daily cross-sectional median",
            "robust_z": "(x-median)/(1.4826*MAD+eps)",
            "percentile": "daily fractional cross-sectional rank in [0,1]",
            "mean_std_window": 60,
            "signal_mean_window": 20,
            "direction": "selected separately for each full variant using sample-period Pearson IC only",
            "industry_neutral": "daily rank-space OLS residual on point-in-time SW level-1 dummies",
            "rebalance": "week_end",
            "groups": args.groups,
            "cost_bps": args.cost_bps,
        },
        "industry_level_count": len(industry_names),
        "window_labels": {key: label for key, label, _begin, _finish in WINDOWS},
        "window_ranges": {key: [begin, finish] for key, _label, begin, finish in WINDOWS},
        "diagnostics": diagnostics,
        "variants": variants,
        "best_raw": {
            "id": best_raw["id"], "label": best_raw["label"],
            "stability": best_raw["raw"]["rank_icir_stability"],
        },
        "best_industry_neutral": {
            "id": best_neutral["id"], "label": best_neutral["label"],
            "stability": best_neutral["industry_neutral"]["rank_icir_stability"],
        },
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _build_html(report, destination)
    destination.with_suffix(".json").write_text(
        json.dumps(_safe(report), ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    print(f"[entropy] written -> {destination}", flush=True)


if __name__ == "__main__":
    main()
