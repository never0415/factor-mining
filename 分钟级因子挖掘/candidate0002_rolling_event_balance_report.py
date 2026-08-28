"""Test 20-day aggregated up/down abnormal-volume balance."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from min_gp.config import ADJUSTED_CLOSE_PARQUET, MINUTE_PARQUET, ZZ500_PIT_PARQUET
from min_gp.data import load_pit_codes, load_pit_daily_mask, load_pit_dates
from min_gp.factor_export import write_factor_parquet
from min_gp.label import tensor_rebalance_fwd_ret
from min_gp.long_short_battle_html_report import WINDOWS, evaluate_period
from min_gp.numeric.ranking import cross_section_rank
from min_gp.spectral_data import build_minute_slice, load_daily_close_tensor


def daily_up_down_strength(volume, close, lookback=5, threshold=1.0, deadzone=0.5):
    q=torch.full_like(volume,float('nan'));ret=torch.full_like(close,float('nan'))
    for start,end in ((0,120),(120,240)):
        past=volume[...,start:end-1].unfold(-1,lookback,1);good=torch.isfinite(past).all(-1);base=torch.nan_to_num(past).mean(-1);current=volume[...,start+lookback:end]
        strength=current/base.clamp(min=1e-12)-1
        q[...,start+lookback:end]=torch.where(good&torch.isfinite(current)&(base>0),strength,torch.full_like(strength,float('nan')))
        p=close[...,start:end-lookback];c=close[...,start+lookback:end];r=c/p.clamp(min=1e-12)-1
        ret[...,start+lookback:end]=torch.where(torch.isfinite(c)&torch.isfinite(p)&(p>0),r,torch.full_like(r,float('nan')))
    rv=torch.isfinite(ret);count=rv.sum(-1);clean=torch.nan_to_num(ret);mean=clean.sum(-1)/count.clamp(min=1);var=((clean-mean.unsqueeze(-1)).square()*rv).sum(-1)/count.clamp(min=1);sigma=var.clamp(min=0).sqrt();z=ret/sigma.unsqueeze(-1).clamp(min=1e-8)
    event=torch.isfinite(q)&(q>threshold);positive=torch.nan_to_num(q).clamp(min=0)
    up=(positive*(event&torch.isfinite(z)&(z>deadzone))).sum(-1);down=(positive*(event&torch.isfinite(z)&(z< -deadzone))).sum(-1)
    day_valid=torch.isfinite(volume).any(-1)&torch.isfinite(close).any(-1)
    nan=torch.full_like(up,float('nan'))
    return torch.where(day_valid,up,nan),torch.where(day_valid,down,nan)


def rolling_sum(x,window=20,min_ratio=.8):
    valid=torch.isfinite(x);clean=torch.nan_to_num(x);cs=torch.cat((torch.zeros_like(x[:,:1]),clean.cumsum(1)),1);cc=torch.cat((torch.zeros_like(x[:,:1]),valid.to(x.dtype).cumsum(1)),1)
    total=cs[:,window:]-cs[:,:-window];count=cc[:,window:]-cc[:,:-window];total=torch.where(count>=window*min_ratio,total,torch.full_like(total,float('nan')))
    return torch.cat((torch.full((x.shape[0],window-1),float('nan'),device=x.device,dtype=x.dtype),total),1)


def fmt(x,pct=False,d=4):
    if x is None or not np.isfinite(x):return '—'
    return f'{x:.{d}%}' if pct else f'{x:.{d}f}'


def write_html(report,path):
    modes=(("original","原始因子"),("daily_rank20","每日方向差→RankMA20"),("rolling_balance","先累计20日U/L→最后排名"));rows=[]
    for key,label,_s,_e in WINDOWS:
        for mode,title in modes:
            m=report['windows'][key][mode];groups=''.join(f'<td>{fmt(x,True,2)}</td>' for x in m['group_total'])
            rows.append(f"<tr class='{mode}'><td>{label}</td><td>{title}</td><td>{m['direction']:+d}</td><td>{fmt(m['ic']['mean'])}</td><td>{fmt(m['ic']['ir_annual'],d=2)}</td><td>{fmt(m['rank_ic']['mean'])}</td><td>{fmt(m['rank_ic']['ir_annual'],d=2)}</td>{groups}<td>{fmt(m['gross_ls_total'],True,2)}</td><td>{fmt(m['net_ls_total'],True,2)}</td><td>{fmt(m['turnover'],True,1)}</td></tr>")
    path.write_text(f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>20日异常放量方向累计</title><style>body{{margin:0;background:#f3f5f8;color:#172033;font:14px/1.5 system-ui,'Microsoft YaHei'}}main{{max-width:1550px;margin:auto;padding:28px}}.card{{background:white;border:1px solid #dde3ec;border-radius:12px;padding:18px;margin:14px 0}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}th,td{{padding:9px;border-bottom:1px solid #e4e8ef;text-align:right}}th:first-child,td:first-child{{text-align:left}}.rolling_balance{{background:#edf8f4}}p{{color:#667085}}</style></head><body><main><h1>先累计20日上涨/下跌异常放量，再做截面排名</h1><p>B20=(Σ20 U−Σ20 L)/(Σ20 U+Σ20 L+ε)。新因子样本期锁定方向为 {report['rolling_direction']:+d}；事件定义与上一版相同。</p><section class='card'><div class='scroll'><table><thead><tr><th>阶段</th><th>因子</th><th>方向</th><th>IC</th><th>ICIR</th><th>RankIC</th><th>RankICIR</th><th>Q1</th><th>Q2</th><th>Q3</th><th>Q4</th><th>Q5</th><th>毛多空</th><th>扣费多空</th><th>换手</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section></main></body></html>""",encoding='utf-8')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',required=True);ap.add_argument('--out',required=True);ap.add_argument('--minute-parquet',default=str(MINUTE_PARQUET));ap.add_argument('--daily-parquet',default=str(ADJUSTED_CLOSE_PARQUET));ap.add_argument('--pit',default=str(ZZ500_PIT_PARQUET));ap.add_argument('--date-chunk',type=int,default=256);ap.add_argument('--cpu',action='store_true');a=ap.parse_args();device='cpu' if a.cpu else 'cuda'
    side=Path(a.candidate);rec=json.loads(side.read_text(encoding='utf-8'));run=side.parent.parent;start,end=WINDOWS[0][2],WINDOWS[-1][3];inst=load_pit_codes(a.pit,start,end);dates=load_pit_dates(a.pit,start,end);I,D=len(inst),len(dates);up=torch.full((I,D),float('nan'),device=device);down=torch.full_like(up,float('nan'))
    for left in range(0,D,a.date_chunk):
        right=min(D,left+a.date_chunk);ds=[str(x) for x in dates[left:right]];print(f'[rolling-balance] dates {left+1}-{right}/{D}',flush=True);ctx,_=build_minute_slice(a.minute_parquet,ds[0],ds[-1],fields=('close','volume'),instruments=inst,dates=ds,device=device);u,l=daily_up_down_strength(ctx['volume'],ctx['close']);up[:,left:right]=u;down[:,left:right]=l;del ctx,u,l
        if torch.cuda.is_available():torch.cuda.empty_cache()
    u20,l20=rolling_sum(up,20),rolling_sum(down,20);den=u20+l20;balance=torch.where(torch.isfinite(den)&(den>0),(u20-l20)/den.clamp(min=1e-12),torch.full_like(den,float('nan')));factor=cross_section_rank(balance)
    close=load_daily_close_tensor(a.daily_parquet,dates,inst,device=device);pool=load_pit_daily_mask(a.pit,dates,inst,device=device)&torch.isfinite(close);factor=torch.where(pool,factor,torch.full_like(factor,float('nan')));fwd=tensor_rebalance_fwd_ret(close,dates,'week_end',1);fwd=torch.where(pool,fwd,torch.full_like(fwd,float('nan')));da=np.asarray(dates);sm=torch.as_tensor((da>=WINDOWS[0][2])&(da<=WINDOWS[0][3]),device=device);unsigned=evaluate_period(factor,fwd,sm,1,5,30.);direction=1 if unsigned['ic']['mean']>=0 else -1
    base=json.loads((run/'long_short_battle_12_factor_report.json').read_text(encoding='utf-8'));base_row=next(x for x in base['candidates'] if x['id']==side.name.removesuffix('.parquet.json'));previous=json.loads((run/'candidate_0002_event_direction_rank20.json').read_text(encoding='utf-8'));windows={}
    for key,_label,ws,we in WINDOWS:
        mask=torch.as_tensor((da>=ws)&(da<=we),device=device);new=evaluate_period(factor,fwd,mask,direction,5,30.);new['direction']=direction;old=dict(base_row['windows'][key]);old['direction']=int(rec['fitness']['direction']);prior=dict(previous['windows'][key]['directional']);windows[key]={'original':old,'daily_rank20':prior,'rolling_balance':new}
    expression='cross_section_rank((rolling_sum_20(up_event_strength)-rolling_sum_20(down_event_strength))/(rolling_sum_20(up_event_strength)+rolling_sum_20(down_event_strength)))';report={'generated':datetime.now().astimezone().isoformat(timespec='seconds'),'source_id':side.name.removesuffix('.parquet.json'),'rolling_direction':direction,'expression':expression,'parameters':{'lookback_minutes':5,'event_threshold':1.,'deadzone_sigma':.5,'session_reset':True,'aggregation_days':20,'min_valid_days':16},'windows':windows}
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);write_html(report,out);out.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False),encoding='utf-8');write_factor_parquet(factor,inst,dates,out.with_suffix('.parquet'),metadata={'family':'rolling_event_direction_balance','direction':direction,'expression':expression,'parameters':report['parameters']});print(f'[rolling-balance] written -> {out}',flush=True)


if __name__=='__main__':main()
