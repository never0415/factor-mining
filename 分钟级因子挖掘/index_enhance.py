"""
Huatai-style index enhancement: quadratic/linear program on top of min_gp factors.

Reference: 华泰证券《再探基于遗传规划的选择策略》(2019) 附录 — 换手率控制的数学推导.

Key constraints (from the paper):
  - Benchmark = CSI 500 index weights (w_B)
  - Individual active weight: |w_i - w_B,i| ≤ 0.01  (1% deviation)
  - Turnover: ||w - w_0||_1 ≤ δ
  - Full investment: Σ w_i = 1
  - Long only: w_i ≥ 0

The non-smooth turnover constraint is handled via auxiliary variables (u, v):
  w - w_0 = u - v,  u≥0, v≥0,  Σ(u+v) ≤ δ

Usage:
  python -m min_gp.index_enhance --seed s27_valley_vwap --start 2018-01-02 --end 2019-12-31 --period 5
"""
import argparse
import os
import sys
import time

import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")
from min_gp.data import (build_slice, load_pit_pool, build_pool_mask,
                         load_industry, industry_ids, load_market_cap,
                         build_mcap_tensor, load_index_codes)
from min_gp.expr import Ctx, parse
from min_gp.fitness import daily_spearman_ic, industry_neutral, market_cap_neutral, summarize_ic
from min_gp.seeds import SEEDS
from min_gp.config import (MINUTE_PARQUET, ZZ500_PIT_PARQUET,
                           DAILY_PRICE_PARQUET)

PARQUET = str(MINUTE_PARQUET)
PIT_PQ = str(ZZ500_PIT_PARQUET)
DAY_PQ = str(DAILY_PRICE_PARQUET)
IND_PQ = ""
MCAP_PQ = ""

plt.style.use("seaborn-v0_8-whitegrid")


# ═══════════════════════════════════════════════════════
# 1. Optimizer: LP with turnover + individual weight constraints
# ═══════════════════════════════════════════════════════

def solve_weights(r_vec, w_b, w_0, delta=0.5, max_dev=0.01, lambda_risk=0.0, Sigma=None):
    """Solve index-enhancement portfolio weights via LP/QP.

    Args:
        r_vec: (N,) predicted returns (higher = better).
        w_b: (N,) benchmark weights (CSI 500), sum=1.
        w_0: (N,) previous portfolio weights, sum=1. None → equal to w_b.
        delta: turnover cap (L1 norm of weight change). Default 0.5 = 50% turnover.
        max_dev: max active weight deviation per stock. Default 0.01 (1%).
        lambda_risk: risk aversion. 0 = LP (no covariance), >0 = QP.
        Sigma: (N,N) covariance matrix (required if lambda_risk > 0).

    Returns:
        w_opt: (N,) optimal weights (sum=1, ≥0).
        status: solver status string.
    """
    N = len(r_vec)
    if w_0 is None:
        w_0 = w_b.copy()

    # Decision variables: w (absolute weights), u, v (auxiliary for turnover L1)
    w = cp.Variable(N, nonneg=True)
    u = cp.Variable(N, nonneg=True)
    v = cp.Variable(N, nonneg=True)

    constraints = [
        cp.sum(w) == 1,                           # fully invested
        w >= cp.maximum(0, w_b - max_dev),         # lower bound: ≥ benchmark - 1%
        w <= w_b + max_dev,                        # upper bound: ≤ benchmark + 1%
        w - w_0 == u - v,                          # auxiliary: w - w_0 = u - v
        cp.sum(u + v) <= delta,                    # turnover: L1(w - w_0) ≤ delta
    ]

    # Objective
    if lambda_risk > 0 and Sigma is not None:
        risk = cp.quad_form(w, Sigma)
        obj = cp.Maximize(r_vec @ w - lambda_risk * risk)
    else:
        obj = cp.Maximize(r_vec @ w)

    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver=cp.OSQP, verbose=False, max_iter=5000, time_limit=5.0)
    except (cp.error.SolverError, Exception):
        try:
            prob.solve(solver=cp.SCS, verbose=False, max_iters=2000, time_limit_secs=5.0)
        except Exception:
            pass

    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        # fallback: equal-weight benchmark
        return w_b.copy(), prob.status

    w_opt = np.maximum(w.value, 0)
    w_opt = w_opt / w_opt.sum()                    # re-normalize
    return w_opt, prob.status


# ═══════════════════════════════════════════════════════
# 2. Daily weight computation pipeline
# ═══════════════════════════════════════════════════════

def run_index_enhance(factor, fwd_ret, meta, period=5, delta=0.5, max_dev=0.01,
                      lambda_risk=0.0, fee_pct=0.3):
    """Compute daily enhanced portfolio weights and track excess return vs CSI 500.

    Returns:
        excess_ret: (D,) daily excess return over benchmark.
        weights_history: list of (date, {instrument: weight}) for rebalance days.
    """
    import pyarrow.parquet as pq
    insts = meta["instruments"]
    dates = meta["dates"]
    I, D = factor.shape
    fnp = factor.cpu().numpy()

    # daily benchmark weights from CSV (PIT, same as pool)
    pool_data = load_pit_pool(PIT_PQ, meta["start"], meta["end"])
    date_to_pool = {}
    pdates = [p[0] for p in pool_data]
    for j, d in enumerate(dates):
        pi = 0
        while pi + 1 < len(pdates) and pdates[pi + 1] <= d:
            pi += 1
        _, codes, bweights = pool_data[pi]
        inst_to_w = {}
        for c, bw in zip(codes, bweights):
            cvt = ("sh" if "." not in c else c.split(".")[0].replace("sh", "sh").replace("sz", "sz"))
            # parse code: 000006.XSHE → sz000006
            if "." in c:
                num, exch = c.split(".")
                cvt = ("sh" if exch == "XSHG" else "sz") + num
            else:
                cvt = c
            inst_to_w[cvt] = bw
        date_to_pool[d] = inst_to_w

    # daily OHLCV for benchmark return
    filters = [("trade_date", ">=", meta["start"]), ("trade_date", "<=", meta["end"])]
    pdf_day = pq.read_table(DAY_PQ, filters=filters).to_pandas()
    # The local daily parquet stores instrument/trade_date as regular columns,
    # whereas the legacy cache used a prebuilt MultiIndex.
    pdf_day["trade_date"] = pdf_day["trade_date"].astype(str)
    pdf_day = pdf_day.set_index(["trade_date", "instrument"]).sort_index()

    # build per-day benchmark return
    bm_rets = []
    for j, d in enumerate(dates):
        bm_w = date_to_pool.get(d, {})
        day_ret = 0.0
        total_w = 0.0
        for inst, bw in bm_w.items():
            bw_pct = bw / 100.0  # CSV weights are in %
            try:
                sub = pdf_day.loc[(d, inst)] if (d, inst) in pdf_day.index else None
                if sub is None:
                    sub = pdf_day.loc[(slice(None), inst), :].droplevel("instrument")
                    if d not in sub.index:
                        continue
                    sub = sub.loc[d]
                if j + 1 < D:
                    next_d = dates[j + 1]
                    sub_next = pdf_day.loc[(slice(None), inst), :].droplevel("instrument")
                    if next_d in sub_next.index:
                        ret_i = sub_next.loc[next_d, "close"] / sub["close"] - 1.0
                    else:
                        ret_i = 0.0
                else:
                    ret_i = 0.0
                day_ret += bw_pct * ret_i
                total_w += bw_pct
            except (KeyError, TypeError):
                continue
        if total_w > 0:
            bm_rets.append(day_ret / total_w)
        else:
            bm_rets.append(0.0)
    bm_ret = np.array(bm_rets)

    # portfolio optimization at each rebalance day
    w_prev = None
    port_ret = np.zeros(D)
    weights_history = []
    for d0 in range(0, D, period):
        if d0 % (period * 20) == 0:  # progress every ~20 rebalances
            print(f"  [opt] day {d0}/{D} ({d0*100/D:.0f}%)", flush=True)
        d_end = min(d0 + period, D)
        d_str = dates[d0]
        bm_w_raw = date_to_pool.get(d_str, {})
        if not bm_w_raw:
            # can't trade — use previous weights or benchmark
            w_prev = None
            for d1 in range(d0, d_end):
                port_ret[d1] = bm_ret[d1] if d1 < len(bm_ret) else 0.0
            continue

        # Build aligned vectors for stocks in both factor AND benchmark
        common_insts = []
        r_vals = []
        w_b_vals = []
        for inst, bw in bm_w_raw.items():
            if inst not in insts:
                continue
            i = insts.index(inst)
            fv = fnp[i, d0]
            if np.isnan(fv):
                continue
            common_insts.append(inst)
            r_vals.append(fv)
            w_b_vals.append(bw / 100.0)  # % → fraction

        if len(common_insts) < 50:
            w_prev = None
            for d1 in range(d0, d_end):
                port_ret[d1] = bm_ret[d1] if d1 < len(bm_ret) else 0.0
            continue

        N = len(common_insts)
        r_vec = np.array(r_vals)
        # calibrate: factor_zscore × IC → expected daily return
        # IC_mean ~0.015 daily, so a +1σ factor stock gets ~1.5bp/day expected excess
        r_mean = np.nanmean(r_vec)
        r_std = np.nanstd(r_vec) or 1.0
        r_vec = (r_vec - r_mean) / r_std * 0.015   # z-score × IC
        w_b = np.array(w_b_vals)
        w_b = w_b / w_b.sum()

        if w_prev is None:
            w_0 = w_b.copy()
        else:
            # build w_0 from previous weights
            w_0 = np.zeros(N)
            for k, inst in enumerate(common_insts):
                w_0[k] = w_prev.get(inst, 0.0)
            w_0 = w_0 / (w_0.sum() or 1.0)

        w_opt, status = solve_weights(r_vec, w_b, w_0, delta=delta, max_dev=max_dev,
                                      lambda_risk=lambda_risk)

        w_prev = {inst: w_opt[k] for k, inst in enumerate(common_insts)}
        weights_history.append((d_str, w_prev.copy()))

        # compute portfolio returns for holding period
        for d1 in range(d0, d_end):
            if d1 >= len(dates):
                break
            day_ret = 0.0
            for k, inst in enumerate(common_insts):
                try:
                    sub = pdf_day.loc[(dates[d1], inst)] if (dates[d1], inst) in pdf_day.index else None
                    if sub is None:
                        sub = pdf_day.loc[(slice(None), inst), :].droplevel("instrument")
                        if dates[d1] not in sub.index:
                            continue
                        sub = sub.loc[dates[d1]]
                    if d1 + 1 < D:
                        next_d = dates[d1 + 1]
                        sub_next = pdf_day.loc[(slice(None), inst), :].droplevel("instrument")
                        if next_d in sub_next.index:
                            ret_i = sub_next.loc[next_d, "close"] / sub["close"] - 1.0
                        else:
                            ret_i = 0.0
                    else:
                        ret_i = 0.0
                    day_ret += w_opt[k] * ret_i
                except (KeyError, TypeError):
                    continue
            port_ret[d1] = day_ret

    # excess return over benchmark
    excess = port_ret - bm_ret[:D]

    # fee: turnover cost at rebalance days
    fee_drag = np.zeros(D)
    for idx, (_, wh) in enumerate(weights_history):
        d0 = idx * period
        if idx == 0:
            to = 1.0
        else:
            prev_w = weights_history[idx - 1][1]
            all_insts = set(list(wh.keys()) + list(prev_w.keys()))
            to = sum(abs(wh.get(i, 0) - prev_w.get(i, 0)) for i in all_insts) / 2.0
        for d1 in range(d0, min(d0 + period, D)):
            fee_drag[d1] = min(to * fee_pct / 100.0 / period, fee_pct / 100.0)

    excess_net = excess - fee_drag

    # print summary
    excess_cum = np.cumprod(1 + excess_net) - 1
    ann_excess = (1 + excess_cum[-1]) ** (252 / max(D, 1)) - 1 if D > 0 else 0
    vol = np.nanstd(excess_net) * np.sqrt(252)
    ir = ann_excess / vol if vol > 0 else float("nan")
    mdd = ((np.maximum.accumulate(1 + excess_cum) - (1 + excess_cum)) / np.maximum.accumulate(1 + excess_cum)).max()

    print(f"[enhance] period={period}d, delta={delta}, max_dev={max_dev*100:.0f}%")
    print(f"[enhance] excess CAGR={ann_excess*100:.1f}%, IR={ir:.2f}, MaxDD={mdd*100:.1f}%")
    print(f"[enhance] benchmark CAGR={(np.prod(1+bm_ret[~np.isnan(bm_ret)])**(252/D)-1)*100:.1f}%")
    print(f"[enhance] fee drag avg={np.nanmean(fee_drag)*100:.2f}%/day")

    return excess_net, excess_cum, fee_drag, weights_history, dates, bm_ret


# ═══════════════════════════════════════════════════════
# 3. Plotting
# ═══════════════════════════════════════════════════════

def plot_enhance(dates, excess_cum, bm_cum, fee_drag, period, delta, max_dev, out_png):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                   gridspec_kw={"height_ratios": [3, 1]})

    dts = [pd.Timestamp(d) for d in dates]

    # benchmark cumulative
    bm_pct = np.array(bm_cum) * 100
    ax1.plot(dts, bm_pct, color="gray", lw=1.0, ls="--", label="CSI 500 Benchmark")

    # enhanced portfolio cumulative (benchmark + excess)
    port_pct = bm_pct + np.array(excess_cum) * 100
    ax1.plot(dts, port_pct, color="#4C72B0", lw=1.5, label="Enhanced Portfolio")
    ax1.axhline(0, color="gray", lw=0.8)
    gain = np.maximum(port_pct - bm_pct, 0)
    loss = np.minimum(port_pct - bm_pct, 0)
    # shade excess region
    ax1.fill_between(dts, bm_pct, port_pct, where=(port_pct >= bm_pct),
                     color="#4C72B0", alpha=0.12)
    ax1.fill_between(dts, bm_pct, port_pct, where=(port_pct < bm_pct),
                     color="#C44E52", alpha=0.12)
    ax1.set_ylabel("Cumulative Return (%)")
    ax1.set_title(f"Index Enhancement (Huatai) — period={period}d, δ={delta}, max_dev={max_dev*100:.0f}%")
    ax1.legend(loc="upper left")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    fig.autofmt_xdate()

    # excess return on bottom panel
    excess_pct = np.array(excess_cum) * 100
    ax2.fill_between(dts, 0, excess_pct, where=(excess_pct >= 0),
                     color="#4C72B0", alpha=0.3)
    ax2.fill_between(dts, 0, excess_pct, where=(excess_pct < 0),
                     color="#C44E52", alpha=0.3)
    ax2.plot(dts, excess_pct, color="#333333", lw=1.0)
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.set_ylabel("Excess Return (%)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"[plot] {out_png}")


# ═══════════════════════════════════════════════════════
# 4. Full-history year-by-year chaining
# ═══════════════════════════════════════════════════════

def run_full_enhance(node, device, period, delta, max_dev, lambda_risk, fee_pct):
    """Year-by-year: slice with 45d warmup → factor → enhance → concat. No OOM."""
    years = list(range(2018, 2027))
    excess_all, bm_all, date_all = [], [], []
    for yr in years:
        t0 = time.time()
        s = f"{yr}-01-01" if yr == 2018 else f"{yr-1}-11-15"
        e = f"{yr}-12-31"
        universe = load_index_codes(PIT_PQ)
        tens, masks, fwd_ret, meta = build_slice(
            PARQUET, s, e, device=device, instruments=universe)
        ctx = Ctx(tens, masks, meta, device=device)
        f = ctx.eval(node)

        # pool + neutral (same as single-run)
        pool_data = load_pit_pool(PIT_PQ, meta["start"], meta["end"])
        pm, _ = build_pool_mask(pool_data, meta["dates"], meta["instruments"], device)
        f = torch.where(pm, f, torch.full_like(f, float("nan")))
        if IND_PQ and os.path.isfile(IND_PQ):
            ind_ids = industry_ids(load_industry(IND_PQ), meta["instruments"]).to(device)
            f = industry_neutral(f, ind_ids)
        if MCAP_PQ and os.path.isfile(MCAP_PQ):
            mcap_dict = load_market_cap(MCAP_PQ)
            mcap_t = build_mcap_tensor(mcap_dict, meta["dates"], meta["instruments"], device)
            f = market_cap_neutral(f, mcap_t)
        ic = daily_spearman_ic(f, fwd_ret)
        if torch.nanmean(ic).item() < 0:
            f = -f

        # trim warmup
        if yr > 2018:
            keep0 = next(i for i, d in enumerate(meta["dates"]) if d >= f"{yr}-01-01")
        else:
            keep0 = 0
        f = f[:, keep0:]
        fwd = fwd_ret[:, keep0:]
        meta_warm = dict(meta, dates=meta["dates"][keep0:],
                         start=meta["dates"][keep0], end=meta["dates"][-1],
                         instruments=meta["instruments"], I=meta["I"], D=f.shape[1], NM=meta["NM"])

        ex_net, ex_cum, fee_d, wh, dts, bm_r = run_index_enhance(
            f, fwd, meta_warm, period=period, delta=delta, max_dev=max_dev,
            lambda_risk=lambda_risk, fee_pct=fee_pct)

        excess_all.append(ex_net)
        bm_all.append(bm_r)
        date_all.extend(dts)
        print(f"[{yr}] ic={torch.nanmean(ic).item():+.4f}, "
              f"excess_cagr={(np.prod(1+ex_net[~np.isnan(ex_net)])**(252/max(len(ex_net),1))-1)*100:+.1f}%  "
              f"({time.time()-t0:.0f}s)", flush=True)

    # concat
    ex_full = np.concatenate(excess_all)
    bm_full = np.concatenate(bm_all)
    n = len(ex_full)
    exc_cum = np.cumprod(1 + np.nan_to_num(ex_full)) - 1
    bm_cum = np.cumprod(1 + np.nan_to_num(bm_full)) - 1
    ann_ex = (1 + exc_cum[-1]) ** (252 / n) - 1 if n > 0 else 0
    vol = np.nanstd(ex_full) * np.sqrt(252)
    ir = ann_ex / vol if vol > 0 else float("nan")
    bm_cagr = (1 + bm_cum[-1]) ** (252 / n) - 1 if n > 0 else 0
    print(f"\n[full] {n} days, benchmark CAGR={bm_cagr*100:.1f}%, "
          f"excess CAGR={ann_ex*100:.1f}%, IR={ir:.2f}")

    out = f"min_gp/enhance_full_p{period}.png"
    plot_enhance(date_all, exc_cum, bm_cum, np.zeros(n), period, delta, max_dev, out)
    return exc_cum, bm_cum, date_all


# ═══════════════════════════════════════════════════════
# 5. CLI
# ═══════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default=None)
    ap.add_argument("--expr", default=None)
    ap.add_argument("--start", default="2018-01-02")
    ap.add_argument("--end", default="2019-12-31")
    ap.add_argument("--period", type=int, default=5, help="rebalance period (days)")
    ap.add_argument("--delta", type=float, default=0.5, help="turnover cap (L1, default 0.5)")
    ap.add_argument("--max-dev", type=float, default=0.01, help="max active weight deviation (default 1%%)")
    ap.add_argument("--fee", type=float, default=0.3, help="fee per side in %%")
    ap.add_argument("--risk-lambda", type=float, default=0.0, help="risk aversion (0 = LP)")
    ap.add_argument("--full", action="store_true",
                    help="full-history year-by-year backtest (2018 → latest)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not args.seed and not args.expr:
        sys.exit("need --seed or --expr")

    device = "cpu" if args.cpu else "cuda"
    name = args.seed or "expr"
    if args.expr:
        node = parse(args.expr)
    else:
        node = parse(SEEDS[name])

    if args.full:
        run_full_enhance(node, device, args.period, args.delta, args.max_dev,
                         args.risk_lambda, args.fee)
        return

    # 1. Factor (single slice)
    t0 = time.time()
    universe = load_index_codes(PIT_PQ)
    tens, masks, fwd_ret, meta = build_slice(
        PARQUET, args.start, args.end, device=device, instruments=universe)
    ctx = Ctx(tens, masks, meta, device=device)
    factor = ctx.eval(node)

    # pool
    pool_data = load_pit_pool(PIT_PQ, meta["start"], meta["end"])
    pm, _ = build_pool_mask(pool_data, meta["dates"], meta["instruments"], device)
    factor = torch.where(pm, factor, torch.full_like(factor, float("nan")))

    # neutral
    if IND_PQ and os.path.isfile(IND_PQ):
        ind_ids = industry_ids(load_industry(IND_PQ), meta["instruments"]).to(device)
        factor = industry_neutral(factor, ind_ids)
    if MCAP_PQ and os.path.isfile(MCAP_PQ):
        mcap_dict = load_market_cap(MCAP_PQ)
        mcap_t = build_mcap_tensor(mcap_dict, meta["dates"], meta["instruments"], device)
        factor = market_cap_neutral(factor, mcap_t)

    ic = daily_spearman_ic(factor, fwd_ret)
    if torch.nanmean(ic).item() < 0:
        factor = -factor
    print(f"[factor {name}] IC={torch.nanmean(ic).item():+.4f} ({time.time()-t0:.0f}s)", flush=True)

    # 2. Index enhancement
    excess_net, excess_cum, fee_drag, wh, dates, bm_ret = run_index_enhance(
        factor, fwd_ret, meta, period=args.period, delta=args.delta,
        max_dev=args.max_dev, lambda_risk=args.risk_lambda, fee_pct=args.fee)
    bm_cum = np.cumprod(1 + np.nan_to_num(bm_ret)) - 1

    # 3. Plot
    out = args.out or f"min_gp/enhance_{name}_p{args.period}.png"
    plot_enhance(dates, excess_cum, bm_cum, fee_drag, args.period, args.delta, args.max_dev, out)


if __name__ == "__main__":
    main()
