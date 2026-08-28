"""Batch liquidity/volatility neutralisation report for exported Climb factors."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from min_gp.climb_candidate_neutralization import exposure_corr
from min_gp.climb_mountain_html_report import WINDOWS, evaluate_period, load_candidates
from min_gp.config import ADJUSTED_CLOSE_PARQUET, MINUTE_PARQUET, RISK_EXPOSURES_PARQUET, ZZ500_PIT_PARQUET
from min_gp.data import load_pit_codes, load_pit_daily_mask, load_pit_dates
from min_gp.evaluation.incremental import trailing_signal_mean
from min_gp.evaluation.neutralize import BatchedNeutralizer, trailing_volatility
from min_gp.label import tensor_rebalance_fwd_ret
from min_gp.spectral_data import build_minute_slice, load_daily_close_tensor, load_daily_exposures


def n(x, d=4):
    return "—" if not np.isfinite(x) else f"{x:.{d}f}"


def p(x, d=2):
    return "—" if not np.isfinite(x) else f"{x:.{d}%}"


def write_html(report, path):
    labels = report["window_labels"]
    rows = []
    for c in report["candidates"]:
        for key, _label, _s, _e in WINDOWS:
            r, z = c["windows"][key]["raw"], c["windows"][key]["neutral"]
            rows.append(
                f"<tr><td>{c['id']}</td><td>{labels[key]}</td>"
                f"<td>{n(r['ic']['mean'])}</td><td>{n(z['ic']['mean'])}</td>"
                f"<td>{n(r['rank_ic']['mean'])}</td><td>{n(z['rank_ic']['mean'])}</td>"
                f"<td>{p(r['net_ls_total'])}</td><td>{p(z['net_ls_total'])}</td>"
                f"<td class='{'pos' if z['net_ls_total']-r['net_ls_total']>=0 else 'neg'}'>{p(z['net_ls_total']-r['net_ls_total'])}</td>"
                f"<td>{n(c['windows'][key]['raw_liq_corr'],3)}</td><td>{n(c['windows'][key]['neutral_liq_corr'],3)}</td>"
                f"<td>{n(c['windows'][key]['raw_vol_corr'],3)}</td><td>{n(c['windows'][key]['neutral_vol_corr'],3)}</td></tr>"
            )
    payload = json.dumps(report, ensure_ascii=False).replace("</", "<\\/")
    path.write_text(f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Climb Mountain全因子中性化报告</title>
<style>body{{margin:0;background:#f3f5f8;color:#172033;font:14px/1.5 system-ui,'Microsoft YaHei'}}main{{max-width:1550px;margin:auto;padding:28px}}.card{{background:#fff;border:1px solid #dfe4ec;border-radius:12px;padding:18px;margin:14px 0}}.scroll{{overflow:auto;max-height:720px}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}th,td{{padding:8px 9px;border-bottom:1px solid #e4e8ef;text-align:right}}th{{position:sticky;top:0;background:#f8f9fc}}th:first-child,td:first-child{{text-align:left}}.pos{{color:#087f5b}}.neg{{color:#c33}}select{{padding:7px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.box{{border:1px solid #dfe4ec;border-radius:9px;padding:12px}}small,p{{color:#667085}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}</style></head><body><main>
<h1>Climb Mountain 13因子：流动性+波动性中性化</h1><p>日截面秩空间 OLS：factor rank ~ 1 + ln_amount rank + 60日波动率 rank。方向使用训练期锁定值，周频Q5−Q1，成本30bps。</p>
<section class='card'><h2>全因子对比</h2><div class='scroll'><table><thead><tr><th>因子</th><th>阶段</th><th>原IC</th><th>中性IC</th><th>原RankIC</th><th>中性RankIC</th><th>原净多空</th><th>中性净多空</th><th>改变</th><th>原流动</th><th>中性后</th><th>原波动</th><th>中性后</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<section class='card'><h2>单因子详情</h2><select id='sel'></select><div id='detail'></div></section>
<script id='data' type='application/json'>{payload}</script><script>const R=JSON.parse(document.getElementById('data').textContent),S=document.getElementById('sel'),L=R.window_labels;R.candidates.forEach((c,i)=>S.add(new Option(c.id,i)));function P(x){{return (x*100).toFixed(2)+'%'}}function N(x,d=4){{return x.toFixed(d)}}function render(){{const c=R.candidates[+S.value];let h='<div class="grid">';for(const [k,v] of Object.entries(c.windows)){{const r=v.raw,z=v.neutral;h+=`<div class='box'><b>${{L[k]}}</b><p>IC ${{N(r.ic.mean)}} → ${{N(z.ic.mean)}}<br>ICIR ${{N(r.ic.ir_annual,2)}} → ${{N(z.ic.ir_annual,2)}}<br>RankIC ${{N(r.rank_ic.mean)}} → ${{N(z.rank_ic.mean)}}<br>RankICIR ${{N(r.rank_ic.ir_annual,2)}} → ${{N(z.rank_ic.ir_annual,2)}}<br>扣费多空 ${{P(r.net_ls_total)}} → <b>${{P(z.net_ls_total)}}</b></p><small>中性化分组：${{z.group_total.map(P).join(' / ')}}</small></div>`}}document.getElementById('detail').innerHTML=h+'</div><p><small>'+c.expression+'</small></p>'}}S.onchange=render;render();</script></main></body></html>""", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--minute-parquet", default=str(MINUTE_PARQUET)); ap.add_argument("--daily-parquet", default=str(ADJUSTED_CLOSE_PARQUET))
    ap.add_argument("--risk-exposures", default=str(RISK_EXPOSURES_PARQUET)); ap.add_argument("--pit", default=str(ZZ500_PIT_PARQUET))
    ap.add_argument("--chunk-rows", type=int, default=4096); ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(); device = "cpu" if args.cpu else "cuda"
    candidates = load_candidates(Path(args.factor_dir)); start,end=WINDOWS[0][2],WINDOWS[-1][3]
    instruments,dates=load_pit_codes(args.pit,start,end),load_pit_dates(args.pit,start,end)
    context,meta=build_minute_slice(args.minute_parquet,start,end,fields=("open","high","low","close"),instruments=instruments,dates=dates,device=device)
    close=load_daily_close_tensor(args.daily_parquet,meta["dates"],meta["instruments"],device=device)
    pool=load_pit_daily_mask(args.pit,meta["dates"],meta["instruments"],device=device)&torch.isfinite(close)
    fwd=tensor_rebalance_fwd_ret(close,meta["dates"],"week_end",1);fwd=torch.where(pool,fwd,torch.full_like(fwd,float("nan")))
    vol=trailing_volatility(close,60);styles,_i,_l=load_daily_exposures(args.risk_exposures,meta["dates"],meta["instruments"],continuous_columns=("ln_amount",),industry_column=None,device=device);liq=styles["ln_amount"]
    neutralizer=BatchedNeutralizer(fwd.shape,continuous=(liq,vol),rank_space=True,min_cross_section=30)
    da=np.asarray(meta["dates"]);out=[]
    for pos,c in enumerate(candidates,1):
        print(f"[climb-all-neutral] {pos}/{len(candidates)} {c['id']}",flush=True)
        raw=trailing_signal_mean(c["genome"].evaluate(context,chunk_rows=args.chunk_rows),5);raw=torch.where(pool,raw,torch.full_like(raw,float("nan")));z=torch.where(pool,neutralizer(raw),torch.full_like(raw,float("nan")))
        windows={}
        for key,_label,ws,we in WINDOWS:
            mask=torch.as_tensor((da>=ws)&(da<=we),device=device)
            windows[key]={"raw":evaluate_period(raw,fwd,mask,c["direction"],5,30.),"neutral":evaluate_period(z,fwd,mask,c["direction"],5,30.),"raw_liq_corr":exposure_corr(raw,liq,fwd,mask),"neutral_liq_corr":exposure_corr(z,liq,fwd,mask),"raw_vol_corr":exposure_corr(raw,vol,fwd,mask),"neutral_vol_corr":exposure_corr(z,vol,fwd,mask)}
        out.append({"id":c["id"],"direction":c["direction"],"expression":c["expression"],"windows":windows});del raw,z
        if torch.cuda.is_available():torch.cuda.empty_cache()
    report={"generated":datetime.now().astimezone().isoformat(timespec="seconds"),"window_labels":{k:l for k,l,_s,_e in WINDOWS},"definitions":{"liquidity":"PIT ln_amount","volatility":"60-day trailing daily-return volatility","method":"daily cross-sectional rank-space OLS"},"candidates":out}
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True);path.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False),encoding='utf-8');write_html(report,path);print(f"[climb-all-neutral] written -> {path}")


if __name__ == "__main__":main()
