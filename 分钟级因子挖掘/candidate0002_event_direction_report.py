"""Test a session-aware event-volume direction variant of candidate_0002."""

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
from min_gp.operators.temporal import rank_smooth_daily
from min_gp.spectral_data import build_minute_slice, load_daily_close_tensor


def event_direction_daily(volume, close, lookback=5, threshold=1.0, deadzone=0.5):
    """Daily intensity * directional balance; morning/afternoon reset separately."""
    q = torch.full_like(volume, float("nan"))
    ret = torch.full_like(close, float("nan"))
    for start, end in ((0, 120), (120, 240)):
        past = volume[..., start:end - 1].unfold(-1, lookback, 1)
        good = torch.isfinite(past).all(-1)
        base = torch.nan_to_num(past).mean(-1)
        current = volume[..., start + lookback:end]
        strength = current / base.clamp(min=1e-12) - 1.0
        q[..., start + lookback:end] = torch.where(
            good & torch.isfinite(current) & (base > 0), strength,
            torch.full_like(strength, float("nan")),
        )
        prior_close = close[..., start:end - lookback]
        current_close = close[..., start + lookback:end]
        r = current_close / prior_close.clamp(min=1e-12) - 1.0
        ret[..., start + lookback:end] = torch.where(
            torch.isfinite(current_close) & torch.isfinite(prior_close) & (prior_close > 0),
            r, torch.full_like(r, float("nan")),
        )
    rv = torch.isfinite(ret)
    count = rv.sum(-1)
    clean = torch.nan_to_num(ret)
    mean = clean.sum(-1) / count.clamp(min=1)
    var = ((clean - mean.unsqueeze(-1)).square() * rv).sum(-1) / count.clamp(min=1)
    sigma = var.clamp(min=0).sqrt()
    z = ret / sigma.unsqueeze(-1).clamp(min=1e-8)
    event = torch.isfinite(q) & (q > threshold)
    up = event & torch.isfinite(z) & (z > deadzone)
    down = event & torch.isfinite(z) & (z < -deadzone)
    clean_q = torch.nan_to_num(q).clamp(min=0)
    up_strength = (clean_q * up).sum(-1)
    down_strength = (clean_q * down).sum(-1)
    directed_total = up_strength + down_strength
    balance = torch.where(
        directed_total > 0,
        (up_strength - down_strength) / directed_total.clamp(min=1e-12),
        torch.zeros_like(directed_total),
    )
    event_values = torch.where(event, q, torch.full_like(q, float("nan")))
    intensity = torch.nanmedian(event_values, dim=-1).values
    daily = intensity * balance
    return torch.where(event.any(-1), daily, torch.full_like(daily, float("nan")))


def fmt(x, pct=False, digits=4):
    if x is None or not np.isfinite(x): return "—"
    return f"{x:.{digits}%}" if pct else f"{x:.{digits}f}"


def write_html(report, path):
    rows=[]
    for key,label,_s,_e in WINDOWS:
        for mode,title in (("original","原始 candidate_0002"),("directional","事件方向差＋RankMA20")):
            m=report["windows"][key][mode]
            groups="".join(f"<td>{fmt(x,True,2)}</td>" for x in m["group_total"])
            rows.append(f"<tr class='{mode}'><td>{label}</td><td>{title}</td><td>{m['direction']:+d}</td>"
                f"<td>{fmt(m['ic']['mean'])}</td><td>{fmt(m['ic']['ir_annual'],digits=2)}</td>"
                f"<td>{fmt(m['rank_ic']['mean'])}</td><td>{fmt(m['rank_ic']['ir_annual'],digits=2)}</td>{groups}"
                f"<td>{fmt(m['gross_ls_total'],True,2)}</td><td>{fmt(m['net_ls_total'],True,2)}</td>"
                f"<td>{fmt(m['turnover'],True,1)}</td><td>{m['weeks']}</td></tr>")
    path.write_text(f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>candidate_0002 事件分钟方向差</title>
<style>body{{margin:0;background:#f3f5f8;color:#172033;font:14px/1.5 system-ui,'Microsoft YaHei'}}main{{max-width:1500px;margin:auto;padding:28px}}.card{{background:#fff;border:1px solid #dde3ec;border-radius:12px;padding:18px;margin:14px 0}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}th,td{{padding:9px;border-bottom:1px solid #e4e8ef;text-align:right}}th:first-child,td:first-child{{text-align:left}}.directional{{background:#edf8f4}}code{{word-break:break-all;white-space:normal}}p{{color:#667085}}</style></head><body><main>
<h1>异常放量事件方向差＋20日排名平滑</h1><p>上午/下午分别预热5分钟；Q=V/前5分钟均量−1，Q&gt;1触发事件；事件方向使用过去5分钟收益/当日日内波动率，±0.5σ死区；日信号=异常放量中位强度×(上涨放量−下跌放量)/(上涨放量+下跌放量)；最后对截面排名做20日均值。样本期确定方向为 {report['directional_direction']:+d}。</p>
<section class='card'><div class='scroll'><table><thead><tr><th>阶段</th><th>因子</th><th>方向</th><th>IC</th><th>ICIR</th><th>RankIC</th><th>RankICIR</th><th>Q1</th><th>Q2</th><th>Q3</th><th>Q4</th><th>Q5</th><th>毛多空</th><th>扣费多空</th><th>换手</th><th>周数</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<section class='card'><h2>定义</h2><code>{report['directional_expression']}</code></section></main></body></html>""",encoding='utf-8')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--minute-parquet',default=str(MINUTE_PARQUET));ap.add_argument('--daily-parquet',default=str(ADJUSTED_CLOSE_PARQUET));ap.add_argument('--pit',default=str(ZZ500_PIT_PARQUET));ap.add_argument('--date-chunk',type=int,default=256);ap.add_argument('--cpu',action='store_true')
    a=ap.parse_args();device='cpu' if a.cpu else 'cuda';side=Path(a.candidate);rec=json.loads(side.read_text(encoding='utf-8'));original_direction=int(rec['fitness']['direction'])
    start,end=WINDOWS[0][2],WINDOWS[-1][3];instruments=load_pit_codes(a.pit,start,end);dates=load_pit_dates(a.pit,start,end);I,D=len(instruments),len(dates)
    daily=torch.full((I,D),float('nan'),device=device,dtype=torch.float32)
    for left in range(0,D,a.date_chunk):
        right=min(D,left+a.date_chunk);chunk_dates=[str(x) for x in dates[left:right]]
        print(f'[event-direction] dates {left+1}-{right}/{D}',flush=True)
        context,_=build_minute_slice(a.minute_parquet,chunk_dates[0],chunk_dates[-1],fields=('close','volume'),instruments=instruments,dates=chunk_dates,device=device)
        daily[:,left:right]=event_direction_daily(context['volume'],context['close'])
        del context
        if torch.cuda.is_available():torch.cuda.empty_cache()
    factor=rank_smooth_daily(daily,window=20)
    close=load_daily_close_tensor(a.daily_parquet,dates,instruments,device=device);pool=load_pit_daily_mask(a.pit,dates,instruments,device=device)&torch.isfinite(close);factor=torch.where(pool,factor,torch.full_like(factor,float('nan')))
    fwd=tensor_rebalance_fwd_ret(close,dates,'week_end',1);fwd=torch.where(pool,fwd,torch.full_like(fwd,float('nan')));da=np.asarray(dates)
    smask=torch.as_tensor((da>=WINDOWS[0][2])&(da<=WINDOWS[0][3]),device=device);unsigned=evaluate_period(factor,fwd,smask,1,5,30.0);direction=1 if unsigned['ic']['mean']>=0 else -1
    run_dir=side.parent.parent;existing=run_dir/'long_short_battle_12_factor_report.json';original_by={}
    if existing.exists():
        prior=json.loads(existing.read_text(encoding='utf-8'));row=next(x for x in prior['candidates'] if x['id']==side.name.removesuffix('.parquet.json'));original_by=row['windows']
    windows={}
    for key,_label,ws,we in WINDOWS:
        mask=torch.as_tensor((da>=ws)&(da<=we),device=device);new=evaluate_period(factor,fwd,mask,direction,5,30.0);new['direction']=direction
        old=original_by[key] if key in original_by else {};old=dict(old);old['direction']=original_direction;windows[key]={'original':old,'directional':new}
    expression='rank_smooth_daily(event_intensity * (up_event_strength-down_event_strength)/(up_event_strength+down_event_strength), window=20)'
    report={'generated':datetime.now().astimezone().isoformat(timespec='seconds'),'source_id':side.name.removesuffix('.parquet.json'),'original_direction':original_direction,'directional_direction':direction,'directional_expression':expression,'parameters':{'lookback_minutes':5,'event_threshold':1.0,'deadzone_sigma':0.5,'session_reset':True,'rank_smooth_days':20,'extra_signal_average_days':0},'window_labels':{k:l for k,l,_s,_e in WINDOWS},'windows':windows}
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);write_html(report,out);out.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False),encoding='utf-8')
    factor_path=out.with_suffix('.parquet');write_factor_parquet(factor,instruments,dates,factor_path,metadata={'family':'event_direction_variant','direction':direction,'expression':expression,'parameters':report['parameters']})
    print(f'[event-direction] written -> {out}',flush=True)


if __name__=='__main__':main()
