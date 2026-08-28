"""Build a focused HTML page from an existing industry-neutralization report."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np


STAGES = ("select", "valid", "test")


def _num(value, digits=4):
    return "—" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def _pct(value, digits=2):
    return "—" if value is None or not np.isfinite(value) else f"{value:.{digits}%}"


def build(report: dict, candidate_id: str, destination: Path) -> None:
    candidate = next(
        (row for row in report["candidates"] if row["id"] == candidate_id), None
    )
    if candidate is None:
        raise SystemExit(f"candidate not found: {candidate_id}")

    labels = report["window_labels"]
    comparison_rows = []
    group_rows = []
    cards = []
    for stage in STAGES:
        window = candidate["windows"][stage]
        raw, neutral = window["raw"], window["neutral"]
        net_delta = neutral["net_ls_total"] - raw["net_ls_total"]
        cards.append(
            f"<div class='metric'><span>{labels[stage]}行业中性净多空</span>"
            f"<b class='{'pos' if neutral['net_ls_total'] >= 0 else 'neg'}'>{_pct(neutral['net_ls_total'])}</b>"
            f"<small>原始 {_pct(raw['net_ls_total'])}，变化 {_pct(net_delta)}</small></div>"
        )
        comparison_rows.append(
            f"<tr><td>{labels[stage]}</td><td>{_pct(window['industry_coverage'], 1)}</td>"
            f"<td>{_num(raw['ic']['mean'])} → {_num(neutral['ic']['mean'])}</td>"
            f"<td>{_num(raw['ic']['ir_annual'], 2)} → {_num(neutral['ic']['ir_annual'], 2)}</td>"
            f"<td>{_num(raw['rank_ic']['mean'])} → {_num(neutral['rank_ic']['mean'])}</td>"
            f"<td>{_num(raw['rank_ic']['ir_annual'], 2)} → {_num(neutral['rank_ic']['ir_annual'], 2)}</td>"
            f"<td>{_pct(raw['gross_ls_total'])} → {_pct(neutral['gross_ls_total'])}</td>"
            f"<td>{_pct(raw['net_ls_total'])} → {_pct(neutral['net_ls_total'])}</td>"
            f"<td class='{'pos' if net_delta >= 0 else 'neg'}'>{_pct(net_delta)}</td>"
            f"<td>{_pct(raw['turnover'], 1)} → {_pct(neutral['turnover'], 1)}</td>"
            f"<td>{_pct(window['raw_industry_rank_r2'], 2)} → {_pct(window['neutral_industry_rank_r2'], 2)}</td></tr>"
        )
        groups = "".join(f"<td>{_pct(value)}</td>" for value in neutral["group_total"])
        group_rows.append(
            f"<tr><td>{labels[stage]}</td>{groups}<td>{_pct(neutral['gross_ls_total'])}</td>"
            f"<td>{_pct(neutral['net_ls_total'])}</td><td>{_pct(neutral['net_ls_annual'])}</td>"
            f"<td>{_pct(neutral['turnover'], 1)}</td><td>{neutral['weeks']}</td></tr>"
        )

    test = candidate["windows"]["test"]
    test_delta = test["neutral"]["net_ls_total"] - test["raw"]["net_ls_total"]
    payload = json.dumps(
        {"candidate": candidate, "labels": labels}, ensure_ascii=False
    ).replace("</", "<\\/")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{candidate_id} 行业中性化绩效</title><style>
:root{{--bg:#f3f5f8;--card:#fff;--ink:#172033;--muted:#687187;--line:#dde3ec;--blue:#315efb;--green:#087f5b;--red:#c33}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1580px;margin:auto;padding:28px}}
h1{{margin:0 0 5px}}h2{{font-size:19px}}.sub,.note,small{{color:var(--muted)}}.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin:15px 0;box-shadow:0 2px 12px #1720330a}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.metric{{border:1px solid var(--line);border-radius:9px;padding:13px}}.metric span,.metric small{{display:block}}.metric b{{display:block;font-size:24px;margin:4px 0}}
.pos{{color:var(--green);font-weight:600}}.neg{{color:var(--red);font-weight:600}}.callout{{border-left:4px solid var(--blue);background:#f6f8ff;padding:12px 14px;border-radius:6px}}
.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}th,td{{padding:8px 9px;border-bottom:1px solid var(--line);text-align:right}}th{{background:#f8f9fc;position:sticky;top:0}}th:first-child,td:first-child{{text-align:left}}
canvas{{width:100%;height:330px;border:1px solid var(--line);border-radius:8px}}code{{word-break:break-all;white-space:normal}}@media(max-width:850px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>{candidate_id}：行业中性化绩效</h1>
<div class="sub">逐日申万一级点时行业 · 因子截面排名对行业哑变量回归取残差 · 周频调仓 · 5日信号均值 · Q5−Q1 · 双边30bps成本</div>
<section class="card"><h2>结论</h2><div class="grid">{''.join(cards)}</div>
<p class="callout">测试期扣费多空由 <b>{_pct(test['raw']['net_ls_total'])}</b> 提升至 <b>{_pct(test['neutral']['net_ls_total'])}</b>，改善 <b>{_pct(test_delta)}</b>；测试期 RankIC 为 <b>{_num(test['neutral']['rank_ic']['mean'])}</b>，RankICIR 为 <b>{_num(test['neutral']['rank_ic']['ir_annual'], 2)}</b>。行业排名 R² 从 {_pct(test['raw_industry_rank_r2'], 2)} 降到 {_pct(test['neutral_industry_rank_r2'], 2)}。</p>
<p class="note">行业中性化明显修复测试期，但样本期和验证期扣费收益分别下降。因此它是“测试期泛化改善”，还不能认定为三个阶段一致增强。</p></section>
<section class="card"><h2>原始与行业中性化完整对比</h2><div class="scroll"><table><thead><tr><th>阶段</th><th>行业覆盖</th><th>IC 原→中</th><th>ICIR 原→中</th><th>RankIC 原→中</th><th>RankICIR 原→中</th><th>毛多空 原→中</th><th>净多空 原→中</th><th>净收益变化</th><th>换手 原→中</th><th>行业R² 原→中</th></tr></thead><tbody>{''.join(comparison_rows)}</tbody></table></div></section>
<section class="card"><h2>行业中性化后五分组表现</h2><div class="scroll"><table><thead><tr><th>阶段</th><th>Q1</th><th>Q2</th><th>Q3</th><th>Q4</th><th>Q5</th><th>毛多空</th><th>扣费多空</th><th>扣费年化</th><th>换手</th><th>周数</th></tr></thead><tbody>{''.join(group_rows)}</tbody></table></div></section>
<section class="card"><h2>三阶段五分组累计收益曲线</h2><p class="note">颜色表示 Q1～Q5；实线、长虚线、点虚线分别表示样本期、验证期、测试期。</p><canvas id="groups" width="1450" height="330"></canvas></section>
<section class="card"><h2>扣费多空：原始与行业中性化</h2><p class="note">虚线为原始因子，实线为行业中性化因子。</p><canvas id="ls" width="1450" height="330"></canvas></section>
<section class="card"><h2>因子表达式</h2><code>{html.escape(candidate['expression'])}</code></section>
<script id="data" type="application/json">{payload}</script><script>
const R=JSON.parse(document.getElementById('data').textContent),C=R.candidate;
function draw(id,series){{const c=document.getElementById(id),g=c.getContext('2d'),W=c.width,H=c.height,p=45,vals=series.flatMap(s=>s.v).filter(Number.isFinite);g.clearRect(0,0,W,H);if(!vals.length)return;let lo=Math.min(...vals),hi=Math.max(...vals);if(hi===lo)hi=lo+1;g.strokeStyle='#dfe4ec';g.beginPath();g.moveTo(p,p);g.lineTo(p,H-p);g.lineTo(W-p,H-p);g.stroke();const colors=['#315efb','#08a36a','#e28a11','#8b5cf6','#e04b4b'];series.forEach((s,j)=>{{g.strokeStyle=colors[s.color%colors.length];g.setLineDash(s.dash||[]);g.lineWidth=1.9;g.beginPath();s.v.forEach((v,i)=>{{const x=p+i*(W-2*p)/Math.max(1,s.v.length-1),y=H-p-(v-lo)*(H-2*p)/(hi-lo);i?g.lineTo(x,y):g.moveTo(x,y)}});g.stroke();g.setLineDash([]);g.fillStyle=g.strokeStyle;g.fillText(s.name,p+(j%5)*220,16+Math.floor(j/5)*15)}});g.fillStyle='#687187';g.fillText(hi.toFixed(2),3,p);g.fillText(lo.toFixed(2),3,H-p)}}
const stages=['select','valid','test'],dashes=[[],[9,4],[2,4]],groups=[];stages.forEach((k,si)=>C.windows[k].neutral.group_nav.forEach((v,qi)=>groups.push({{name:R.labels[k]+' Q'+(qi+1),v:[1,...v],color:qi,dash:dashes[si]}})));draw('groups',groups);
const ls=[];stages.forEach((k,si)=>{{ls.push({{name:R.labels[k]+' 原始',v:[1,...C.windows[k].raw.net_nav],color:si,dash:[8,4]}});ls.push({{name:R.labels[k]+' 行业中性',v:[1,...C.windows[k].neutral.net_nav],color:si,dash:[]}})}});draw('ls',ls);
</script></main></body></html>""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.report_json).read_text(encoding="utf-8"))
    build(report, args.candidate, Path(args.out))
    print(f"written -> {args.out}")


if __name__ == "__main__":
    main()
