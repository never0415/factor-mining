"""Evaluate candidate_0002 after multiplying step 5 by same-day return."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from min_gp.config import ADJUSTED_CLOSE_PARQUET, MINUTE_PARQUET, ZZ500_PIT_PARQUET
from min_gp.data import load_pit_codes, load_pit_daily_mask, load_pit_dates
from min_gp.evaluation.incremental import trailing_signal_mean
from min_gp.factors.handbook_skeleton import HandbookSkeletonGenome
from min_gp.label import tensor_rebalance_fwd_ret
from min_gp.long_short_battle_html_report import WINDOWS, evaluate_period
from min_gp.spectral_data import build_minute_slice, load_daily_close_tensor


def f(x, pct=False, digits=4):
    if x is None or not np.isfinite(x): return "—"
    return f"{x:.{digits}%}" if pct else f"{x:.{digits}f}"


def write_html(report, path):
    rows=[]
    for key,label,_s,_e in WINDOWS:
        for mode,title in (("original","原始因子"),("weighted","乘当日收益后")):
            m=report["windows"][key][mode]
            groups="".join(f"<td>{f(x,True,2)}</td>" for x in m["group_total"])
            rows.append(f"<tr class='{mode}'><td>{label}</td><td>{title}</td><td>{m['direction']:+d}</td>"
                f"<td>{f(m['ic']['mean'])}</td><td>{f(m['ic']['ir_annual'],digits=2)}</td>"
                f"<td>{f(m['rank_ic']['mean'])}</td><td>{f(m['rank_ic']['ir_annual'],digits=2)}</td>{groups}"
                f"<td>{f(m['gross_ls_total'],True,2)}</td><td>{f(m['net_ls_total'],True,2)}</td>"
                f"<td>{f(m['turnover'],True,1)}</td></tr>")
    path.write_text(f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>candidate_0002 收益加权变体</title>
<style>body{{margin:0;background:#f3f5f8;color:#172033;font:14px/1.5 system-ui,'Microsoft YaHei'}}main{{max-width:1500px;margin:auto;padding:28px}}.card{{background:#fff;border:1px solid #dde3ec;border-radius:12px;padding:18px;margin:14px 0}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}th,td{{padding:9px;border-bottom:1px solid #e4e8ef;text-align:right}}th:first-child,td:first-child{{text-align:left}}.weighted{{background:#edf5ff}}code{{word-break:break-all;white-space:normal}}p{{color:#667085}}</style></head><body><main>
<h1>candidate_0002：第5步乘当日收益</h1><p>新因子 = MA5〔第5步结果 × 当日复权收盘收益〕。新方向仅由样本期 IC 确定为 {report['weighted_direction']:+d}；验证期和测试期锁定该方向。原因子方向为 {report['original_direction']:+d}。</p>
<section class='card'><div class='scroll'><table><thead><tr><th>阶段</th><th>因子</th><th>方向</th><th>IC</th><th>ICIR</th><th>RankIC</th><th>RankICIR</th><th>Q1</th><th>Q2</th><th>Q3</th><th>Q4</th><th>Q5</th><th>毛多空</th><th>扣费多空</th><th>换手</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<section class='card'><h2>新表达式</h2><code>{report['weighted_expression']}</code></section></main></body></html>""",encoding='utf-8')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--minute-parquet',default=str(MINUTE_PARQUET));ap.add_argument('--daily-parquet',default=str(ADJUSTED_CLOSE_PARQUET));ap.add_argument('--pit',default=str(ZZ500_PIT_PARQUET));ap.add_argument('--chunk-rows',type=int,default=4096);ap.add_argument('--cpu',action='store_true')
    a=ap.parse_args();device='cpu' if a.cpu else 'cuda';side=Path(a.candidate);rec=json.loads(side.read_text(encoding='utf-8'));genome=HandbookSkeletonGenome.from_dict(rec['genome']);original_direction=int(rec['fitness']['direction'])
    start,end=WINDOWS[0][2],WINDOWS[-1][3];instruments=load_pit_codes(a.pit,start,end);dates=load_pit_dates(a.pit,start,end)
    di=pd.Index([str(x) for x in dates],name='trade_date');ii=pd.Index([str(x) for x in instruments],name='instrument')
    cached=pd.read_parquet(side.with_suffix(''),columns=['instrument','trade_date','factor']);cached['trade_date']=cached['trade_date'].astype(str)
    frame=cached.pivot(index='instrument',columns='trade_date',values='factor').reindex(index=ii,columns=di)
    step5=torch.as_tensor(frame.to_numpy(copy=True),device=device,dtype=torch.float32)
    warm=[str(x) for x in dates if str(x)>='2024-01-02'];print('[return-weighted] rebuild holdout',flush=True)
    context,meta=build_minute_slice(a.minute_parquet,'2024-01-02',end,fields=('open','high','low','close','volume'),instruments=instruments,dates=warm,device=device)
    holdout=genome.evaluate(context,chunk_rows=a.chunk_rows);hd=np.asarray(meta['dates']);use=np.flatnonzero(hd>=WINDOWS[2][2]);pos=np.array([di.get_loc(str(hd[i])) for i in use]);step5[:,pos]=holdout[:,use]
    close=load_daily_close_tensor(a.daily_parquet,dates,instruments,device=device);pool=load_pit_daily_mask(a.pit,dates,instruments,device=device)&torch.isfinite(close)
    daily_return=torch.full_like(close,float('nan'));daily_return[:,1:]=close[:,1:]/close[:,:-1].clamp(min=1e-12)-1
    original=trailing_signal_mean(step5,5);weighted=trailing_signal_mean(step5*daily_return,5)
    original=torch.where(pool,original,torch.full_like(original,float('nan')));weighted=torch.where(pool,weighted,torch.full_like(weighted,float('nan')))
    fwd=tensor_rebalance_fwd_ret(close,dates,'week_end',1);fwd=torch.where(pool,fwd,torch.full_like(fwd,float('nan')));da=np.asarray(dates)
    sample_mask=torch.as_tensor((da>=WINDOWS[0][2])&(da<=WINDOWS[0][3]),device=device)
    unsigned=evaluate_period(weighted,fwd,sample_mask,1,5,30.0);weighted_direction=1 if unsigned['ic']['mean']>=0 else -1
    windows={}
    for key,_label,ws,we in WINDOWS:
        mask=torch.as_tensor((da>=ws)&(da<=we),device=device)
        om=evaluate_period(original,fwd,mask,original_direction,5,30.0);wm=evaluate_period(weighted,fwd,mask,weighted_direction,5,30.0);om['direction']=original_direction;wm['direction']=weighted_direction;windows[key]={'original':om,'weighted':wm}
    report={'generated':datetime.now().astimezone().isoformat(timespec='seconds'),'id':side.name.removesuffix('.parquet.json'),'original_direction':original_direction,'weighted_direction':weighted_direction,'original_expression':rec['expression'],'weighted_expression':f"trailing_signal_mean(daily_mul(({rec['expression']}), adjusted_close_return), window=5)",'definitions':{'daily_return':'adjusted_close[t] / adjusted_close[t-1] - 1','direction_selection':'sample-period Pearson IC sign only'},'window_labels':{k:l for k,l,_s,_e in WINDOWS},'windows':windows}
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);write_html(report,out);out.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False),encoding='utf-8');print(f'[return-weighted] written -> {out}',flush=True)


if __name__=='__main__':main()
