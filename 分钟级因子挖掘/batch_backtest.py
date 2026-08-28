"""Batch: daily vs weekly LS for all seeds (single data load, pool + industry-neutral)."""
import os
import sys
import time

import torch

sys.path.insert(0, ".")
from min_gp.data import (build_slice, load_industry, industry_ids,
                         load_index_codes, load_pit_daily_mask)
from min_gp.expr import Ctx, parse
from min_gp.fitness import daily_spearman_ic, industry_neutral, summarize_ic
from min_gp.seeds import SEEDS
from min_gp.backtest import backtest
from min_gp.config import MINUTE_PARQUET, ZZ500_PIT_PARQUET

PARQUET = str(MINUTE_PARQUET)
PIT_PQ = str(ZZ500_PIT_PARQUET)
IND_PQ = ""


def main():
    start, end = sys.argv[1] if len(sys.argv) > 1 else "2018-01-02", sys.argv[2] if len(sys.argv) > 2 else "2019-12-31"
    fee = float(sys.argv[3]) if len(sys.argv) > 3 else 0.003
    t0 = time.time()
    universe = load_index_codes(PIT_PQ)
    tens, masks, fwd_ret, meta = build_slice(
        PARQUET, start, end, device="cuda", instruments=universe)
    print(f"[data] {meta['I']}x{meta['D']} ({time.time()-t0:.0f}s)", flush=True)
    ctx = Ctx(tens, masks, meta, device="cuda")
    pm = load_pit_daily_mask(PIT_PQ, meta["dates"], meta["instruments"], "cuda")
    bw = None
    ind_ids = (industry_ids(load_industry(IND_PQ), meta["instruments"]).to("cuda")
               if IND_PQ and os.path.isfile(IND_PQ) else None)

    print(f"{'seed':26s} {'IC':>7s} {'ICIR':>5s} {'LS_d':>8s} {'LS_w':>8s} {'sharpe_d':>8s} {'sharpe_w':>8s} {'w/d':>5s}")
    for name, s in sorted(SEEDS.items()):
        try:
            f = ctx.eval(parse(s))
        except Exception as e:
            print(f"{name:26s} EVAL_FAIL {type(e).__name__}")
            continue
        f = torch.where(pm, f, torch.full_like(f, float("nan")))
        if ind_ids is not None:
            f = industry_neutral(f, ind_ids)
        ic = daily_spearman_ic(f, fwd_ret)
        si = summarize_ic(ic)
        if si["ic_mean"] < 0:
            f = -f
        rows = {}
        for period, tag in ((1, "d"), (5, "w")):
            rets, costs, turn = backtest(f, fwd_ret, fee=fee, period=period, weights=bw)
            net = rets - costs
            ls = torch.nan_to_num(net[0] - net[-1])
            cum = (torch.cumprod(1 + ls, 0) - 1) * 100
            r = ls[~torch.isnan(ls)]
            vol = r.std() * (252 / period) ** 0.5
            cagr = ((1 + cum[~torch.isnan(cum)][-1] / 100) ** (252 / r.numel()) - 1) * 100
            sharpe = (r.mean() * 252 / period) / vol if vol > 0 else float("nan")
            rows[tag] = (cagr, sharpe)
        print(f"{name:26s} {si['ic_mean']:7.4f} {si['icir']:5.2f} "
              f"{rows['d'][0]:8.1f} {rows['w'][0]:8.1f} {rows['d'][1]:8.2f} {rows['w'][1]:8.2f} "
              f"{(rows['w'][0]/rows['d'][0] if abs(rows['d'][0])>1 else float('nan')):5.2f}", flush=True)


if __name__ == "__main__":
    main()
