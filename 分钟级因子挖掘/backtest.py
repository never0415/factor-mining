"""
min_gp backtest: 5-layer daily-rebalance, long-only top layer + long-short line,
fee configurable (default 0.3% per side), IC chart (daily bars + cumulative line, twin axis).

Usage:
  python -m min_gp.backtest --seed s27_valley_vwap --start 2018-01-02 --end 2019-12-31 --fee 0.003
  python -m min_gp.backtest --expr "ts_mean(div(day_sum(mul(mul(tp,v),f(is_valley))),day_sum(mul(v,f(is_valley)))),20)" ...
"""
import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, ".")
from min_gp.data import (build_slice, load_industry, load_pool, pool_mask,
                         load_component_pool, build_pool_mask, industry_ids,
                         load_market_cap, build_mcap_tensor, load_index_codes,
                         load_pit_daily_mask)
from min_gp.expr import Ctx, _cs_rank, parse
from min_gp.fitness import daily_spearman_ic, industry_neutral, market_cap_neutral, summarize_ic
from min_gp.seeds import SEEDS
from min_gp.config import MINUTE_PARQUET, ZZ500_PIT_PARQUET

PARQUET = str(MINUTE_PARQUET)
OUT_PNG = "min_gp/backtest.png"
OUT_IC_PNG = "min_gp/backtest_ic.png"
MCAP_PQ = ""

plt.style.use("seaborn-v0_8-whitegrid")


def backtest(factor, daily_ret, layers=5, fee=0.003, turnover=1.0, min_n=30, period=1, weights=None):
    """Cross-sectional layer backtest on GPU tensors — signal/return separation.

    Signal:  layer assignment fixed at rebalance days (rank at d0, held `period` days).
    Returns: DAILY close-to-close returns of held stocks, compounded daily → continuous NAV.
    Costs:   fee × turnover × entry-fraction charged ONLY on rebalance days.
    This matches backtrader semantics: daily NAV, no gaps.

    daily_ret: (I, D) actual daily returns close[t+1]/close[t]-1 (NOT forward N-day!).
    weights: (I, D) float32 — if given, layer returns are weight-weighted.
    Returns: rets (layers, D) daily layer returns (NaN where suspended); costs, turn (layers, D).
    """
    rank = _cs_rank(factor)                       # 0..1 per day
    I, D = daily_ret.shape
    dev = factor.device
    layer = torch.full((I, D), -1, dtype=torch.int64, device=dev)
    for d0 in range(0, D, period):                # rebalance days
        d_end = min(d0 + period, D)
        r = rank[:, d0]
        l = layers - 1 - torch.floor(r * layers).clamp(0, layers - 1).long()
        l = torch.where(torch.isnan(r), torch.full_like(l, -1), l)
        layer[:, d0:d_end] = l.unsqueeze(1)
    valid = ~(torch.isnan(factor) | torch.isnan(daily_ret))
    fr = torch.nan_to_num(daily_ret).clamp(-0.11, 0.11)     # clip: 除权日假跳变保护
    if weights is not None:
        w = torch.nan_to_num(weights)                     # (I, D)
    else:
        w = None
    rets = torch.full((layers, D), float("nan"), device=dev)
    for L in range(layers):
        m = (layer == L) & valid
        if w is not None:
            wr = torch.where(m, fr * w, torch.zeros_like(fr)).sum(0)
            denom = torch.where(m, w, torch.zeros_like(w)).sum(0)
            r = wr / denom.clamp(min=1e-8)
            n = m.sum(0)
        else:
            n = m.sum(0)
            r = torch.where(m, fr, torch.zeros_like(fr)).sum(0) / n.clamp(min=1)
        rets[L] = torch.where(n >= min_n, r, torch.full_like(r, float("nan")))
    # actual turnover per layer: fraction of layer members newly entered at each rebalance
    costs = torch.zeros(layers, D, device=dev)
    turn = torch.zeros(layers, D, device=dev)
    for L in range(layers):
        h = (layer == L) & valid
        rebal_idx = torch.arange(0, D, period, device=dev)
        for d0 in rebal_idx:
            if d0 == 0:
                n_cur = h[:, 0].sum().clamp(min=1)
                to = (h[:, 0]).sum() / n_cur
                costs[L, 0] = fee * turnover * to
                turn[L, 0] = to
            else:
                n_cur = h[:, d0].sum().clamp(min=1)
                to = (h[:, d0] & ~h[:, d0-1]).sum() / n_cur
                costs[L, d0] = fee * turnover * to
                turn[L, d0] = to
    return rets, costs, turn


def stats(cum_ret_pct, daily_ret, n_per_year=252):
    r = daily_ret[~torch.isnan(daily_ret)]
    if r.numel() < 10:
        return dict(cagr=float("nan"), sharpe=float("nan"), mdd=float("nan"), win=float("nan"))
    final = cum_ret_pct[~torch.isnan(cum_ret_pct)]
    cagr = (1 + final[-1] / 100) ** (n_per_year / r.numel()) - 1 if final.numel() else float("nan")
    vol = r.std() * np.sqrt(n_per_year)
    sharpe = (r.mean() * n_per_year) / vol if vol > 0 else float("nan")
    nav = torch.cumprod(1 + r, 0)
    mdd = ((nav.cummax(0)[0] - nav) / nav.cummax(0)[0]).max().item()
    win = (r > 0).float().mean().item()
    return dict(cagr=cagr * 100, sharpe=sharpe, mdd=mdd * 100, win=win * 100)


def plot_nav(dates, rets, layers, cost, out_png, bench=None):
    """5-layer cumulative return % + long-short line + CSI500 benchmark line.
    rets: period returns at rebalance days (NaN elsewhere) — cumprod steps only on rebalance.
    bench: period benchmark returns (same grid) — plotted as dashed line."""
    D = rets.shape[1]
    x = np.arange(D)
    fig, ax = plt.subplots(figsize=(18, 9))
    for L in range(layers - 1, -1, -1):
        r = torch.nan_to_num(rets[L])
        cum = (torch.cumprod(1 + r, 0) - 1) * 100
        ax.plot(x, cum.cpu().numpy(), lw=1.2, label=f"Layer {L+1}")
    ls = torch.nan_to_num(rets[0] - rets[-1])
    cum_ls = (torch.cumprod(1 + ls, 0) - 1) * 100
    ax.plot(x, cum_ls.cpu().numpy(), lw=2.5, color="black", label=f"Long-Short (L1-L{layers})")
    if bench is not None:
        b = torch.nan_to_num(bench)
        cum_b = (torch.cumprod(1 + b, 0) - 1) * 100
        ax.plot(x, cum_b.cpu().numpy(), lw=2.5, color="red", ls="--", label="CSI500 index (weighted)")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title(f"{layers}-Layer, net of fees")
    ax.legend(loc="upper left", fontsize=10, ncol=2)
    # real dates on x axis
    if dates is not None and len(dates) == D:
        dt = np.array([np.datetime64(d) for d in dates])
        ax.set_xticks(x[:: max(1, D // 10)])
        ax.set_xticklabels([str(d)[:10] for d in dt[:: max(1, D // 10)]], rotation=30, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"[plot] {out_png}")


def plot_ic(dates, ic, out_png):
    """Daily IC bars (left axis) + cumulative IC line (right axis), one figure."""
    D = ic.shape[0]
    x = np.arange(D)
    ic_cpu = torch.nan_to_num(ic).cpu().numpy()
    cum = np.cumsum(ic_cpu)
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(x, ic_cpu, width=1.0, color="#4C72B0", alpha=0.55, label="Daily IC")
    ax1.axhline(0, color="gray", lw=0.8)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Daily IC")
    ax2 = ax1.twinx()
    ax2.plot(x, cum, color="#C44E52", lw=1.8, label="Cumulative IC")
    ax2.set_ylabel("Cumulative IC")
    ax1.legend(loc="upper left", fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)
    if dates is not None and len(dates) == D:
        dt = np.array([np.datetime64(d) for d in dates])
        step = max(1, D // 8)
        ax1.set_xticks(x[::step])
        ax1.set_xticklabels([str(d)[:10] for d in dt[::step]], rotation=30, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"[plot] {out_png}")


def run_full(seed_name, node, device, fee, turnover, period, layers, parquet_path,
             pool_csv, pit_pq, ind_pq, mcap_pq, flip=False):
    """Year-by-year chained backtest over the full history (2018 → latest). Each year is
    sliced with a 45-day warmup so 20d rolling leaves are valid; only the in-year window
    is kept. Net returns are concatenated and compounded across years.
    flip: global sign (computed once from full-sample IC in main) — NO per-year flip."""
    years = list(range(2018, 2027))
    ic_all, net_all, turn_all, date_all, bench_all = [], [], [], [], []
    pool_data = load_component_pool(pool_csv, top_n=500) if pool_csv and os.path.isfile(pool_csv) else None
    universe = load_index_codes(pit_pq) if pit_pq and os.path.isfile(pit_pq) else None
    ind_map = load_industry(ind_pq) if ind_pq and os.path.isfile(ind_pq) else None
    mcap_dict = load_market_cap(mcap_pq) if mcap_pq and os.path.isfile(mcap_pq) else None
    for yr in years:
        t0 = time.time()
        s = f"{yr}-01-01" if yr == 2018 else f"{yr-1}-11-15"
        e = f"{yr}-12-31"
        tens, masks, fwd_ret, meta = build_slice(
            parquet_path, s, e, device=device, instruments=universe)
        close_d = tens['close'][:, :, -1]  # daily close, always needed for IR
        if period > 1:
            fwd_ret = torch.full_like(close_d, float('nan'))
            fwd_ret[:, :-period] = close_d[:, period:] / close_d[:, :-period] - 1.0
        ctx = Ctx(tens, masks, meta, device=device)
        f = ctx.eval(node)
        if pool_data is not None:
            pm, bw = build_pool_mask(pool_data, meta["dates"], meta["instruments"], device)
            f = torch.where(pm, f, torch.full_like(f, float("nan")))
        elif pit_pq and os.path.isfile(pit_pq):
            pm = load_pit_daily_mask(pit_pq, meta["dates"], meta["instruments"], device)
            f = torch.where(pm, f, torch.full_like(f, float("nan")))
            bw = None
        else:
            bw = None
        # exclude suspended stocks from ranking (their factor values are degenerate: 峰岭≈0)
        f = torch.where(torch.isnan(close_d), torch.full_like(f, float("nan")), f)
        if ind_map is not None:
            ind_ids = industry_ids(ind_map, meta["instruments"]).to(device)
            f = industry_neutral(f, ind_ids)
        if mcap_dict is not None:
            mcap_t = build_mcap_tensor(mcap_dict, meta["dates"], meta["instruments"], device)
            f = market_cap_neutral(f, mcap_t)
        # keep only in-year window (drop warmup days)
        if yr > 2018:
            keep0 = next(i for i, d in enumerate(meta["dates"]) if d >= f"{yr}-01-01")
        else:
            keep0 = 0
        f = f[:, keep0:]
        fwd = fwd_ret[:, keep0:]
        # daily close-to-close returns for layer PnL (signal/return separation:
        # IC direction uses fwd=t+period; layer PnL uses ACTUAL daily returns)
        dret = torch.full_like(f, float('nan'))
        dret[:, 1:] = close_d[:, keep0+1:] / close_d[:, keep0:-1] - 1.0
        if bw is not None:
            bw = bw[:, keep0:]
        dates = meta["dates"][keep0:]
        ic = daily_spearman_ic(f, fwd)
        ic_all.append(ic)
        if flip:
            f = -f
        # layer PnL from daily returns (continuous NAV); NO per-year flip.
        # layers are EQUAL-WEIGHT (weights=None) — CSI500 weights only for benchmark line
        rets, costs, turn = backtest(f, dret, layers=layers, fee=fee, turnover=turnover, period=period, weights=None)
        net = rets - costs
        net_all.append(net)
        # actual turnover at REBALANCE days only (non-rebalance days are 0 → don't dilute)
        to_rebal = turn[:, ::period]
        turn_all.append(torch.nan_to_num(to_rebal[to_rebal > 0]).mean().item())
        date_all.extend(dates)
        # benchmark: CSI500 index return = component weights × daily returns
        daily_ret = close_d[:, keep0+1:] / close_d[:, keep0:-1] - 1.0
        if bw is not None:
            bw_b = bw[:, 1:] / bw[:, 1:].sum(0).clamp(min=1e-12)
            bench = (torch.nan_to_num(daily_ret[:, :bw_b.shape[1]]) * bw_b).sum(0)
            bench_all.append(bench)
        print(f"[{yr}] {meta['I']:4d} insts, {f.shape[1]:3d} days, ic={torch.nanmean(ic).item():+.4f}, "
              f"load {(time.time()-t0):.0f}s", flush=True)
    ic_full = torch.cat(ic_all)
    net_full = torch.cat(net_all, dim=1)
    s_ic = summarize_ic(ic_full)
    print(f"[full] IC_mean={s_ic['ic_mean']:.4f} ICIR={s_ic['icir']:.2f} n={s_ic['n_days']} days, "
          f"avg turnover {sum(turn_all)/len(turn_all)*100:.0f}%/period", flush=True)
    for L in range(layers):
        r = net_full[L]  # keep NaN (suspended days)
        cum = torch.nan_to_num(torch.cumprod(1 + torch.nan_to_num(r, nan=0), 0) - 1) * 100
        st = stats(cum, r)   # daily returns → n_per_year=252
        print(f"[layer {L+1}] CAGR={st['cagr']:.1f}%  Sharpe={st['sharpe']:.2f}  MaxDD={st['mdd']:.1f}%  Win={st['win']:.1f}%")
    ls = net_full[0] - net_full[-1]
    ls = torch.where(torch.isnan(net_full[0]) | torch.isnan(net_full[-1]),
                     torch.full_like(ls, float("nan")), ls)
    cum_ls = torch.nan_to_num(torch.cumprod(1 + torch.nan_to_num(ls, nan=0), 0) - 1) * 100
    st = stats(cum_ls, ls)
    print(f"[LS L1-L{layers}] CAGR={st['cagr']:.1f}%  Sharpe={st['sharpe']:.2f}  MaxDD={st['mdd']:.1f}%  Win={st['win']:.1f}%")

    # ── IR vs CSI500 (daily excess, annualized √252) ──
    bench_grid = None
    if bench_all:
        bench_daily = torch.cat(bench_all)
        l1_d = net_full[0][1:bench_daily.numel()+1]
        excess = l1_d - bench_daily
        excess = excess[~torch.isnan(excess)]
        if excess.numel() > 20:
            daily_ir = excess.mean().item() / excess.std().item()
            print(f"[IR vs CSI500] daily_IR={daily_ir:.4f}  annual_IR={daily_ir*np.sqrt(252):.2f}  n={excess.numel()}d")
        # benchmark on the daily grid for plotting (compound to cumulative)
        bench_grid = torch.full_like(net_full[0], float('nan'))
        bench_grid[1:len(bench_daily)+1] = bench_daily

    plot_nav(date_all, net_full, layers, None, f"min_gp/backtest_full_p{period}.png", bench=bench_grid)
    plot_ic(date_all, ic_full, f"min_gp/backtest_full_ic_p{period}.png")
    return ic_full, net_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default=None, help="seed factor name (see min_gp/seeds.py)")
    ap.add_argument("--expr", default=None, help="raw expression string (overrides --seed)")
    ap.add_argument("--start", default="2018-01-02")
    ap.add_argument("--end", default="2019-12-31")
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--fee", type=float, default=0.003, help="round-trip (bilateral) fee: 0.003=千三")
    ap.add_argument("--turnover", type=float, default=1.0, help="turnover per rebalance (default 1.0)")
    ap.add_argument("--period", type=int, default=1, help="rebalance period in days (5 = weekly, 21 = monthly)")
    ap.add_argument("--out", default=None, help="nav plot path (default min_gp/backtest_p{period}.png)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--pool", default="",
                    help="CSI 500 weekly snapshot dir (legacy); empty string disables")
    ap.add_argument("--pool-csv", default="",
                    help="CSI 500 monthly component CSV with weights; overrides --pool")
    ap.add_argument("--pit", default=str(ZZ500_PIT_PARQUET),
                    help="daily point-in-time CSI 500 membership parquet")
    ap.add_argument("--parquet", default=PARQUET, help="minute OHLCV parquet")
    ap.add_argument("--industry", default="",
                    help="industry map parquet; empty string disables neutralization")
    ap.add_argument("--mcap", default="",
                    help="market cap parquet cache; empty string disables mcap neutralization")
    ap.add_argument("--full", action="store_true",
                    help="full-history year-by-year chained backtest (2018 → latest)")
    args = ap.parse_args()

    device = "cpu" if args.cpu else "cuda"
    if args.expr:
        node = parse(args.expr)
        name = "expr"
    elif args.seed:
        if args.seed not in SEEDS:
            sys.exit(f"unknown seed {args.seed}; available: {sorted(SEEDS)[:10]} ...")
        node = parse(SEEDS[args.seed])
        name = args.seed
    else:
        sys.exit("need --seed or --expr")

    if args.full:
        # compute global factor direction from a representative slice (2018-2019)
        flip = False
        try:
            t0 = time.time()
            universe0 = load_index_codes(args.pit) if args.pit and os.path.isfile(args.pit) else None
            t0_, m0_, fr0_, meta0_ = build_slice(
                args.parquet, "2018-01-02", "2019-12-31", device=device,
                instruments=universe0)
            close0 = t0_['close'][:, :, -1]
            if args.period > 1:
                fr0_ = torch.full_like(close0, float('nan'))
                fr0_[:, :-args.period] = close0[:, args.period:] / close0[:, :-args.period] - 1.0
            ctx0 = Ctx(t0_, m0_, meta0_, device=device)
            f0 = ctx0.eval(node)
            ic0 = daily_spearman_ic(f0, fr0_)
            ic0 = ic0[~torch.isnan(ic0)]
            if ic0.numel() > 50 and ic0.mean().item() < 0:
                flip = True
            print(f"[direction] full-sample IC={ic0.mean().item():+.4f} → flip={flip} "
                  f"({time.time()-t0:.0f}s)")
            del t0_, m0_, fr0_, meta0_, ctx0, f0, ic0
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[direction] failed ({e}), flip=False")
        run_full(name, node, device, args.fee, args.turnover, args.period, args.layers,
                 args.parquet, args.pool_csv, args.pit, args.industry, args.mcap,
                 flip=flip)
        return

    t0 = time.time()
    universe = load_index_codes(args.pit) if args.pit and os.path.isfile(args.pit) else None
    tens, masks, fwd_ret, meta = build_slice(
        args.parquet, args.start, args.end, device=device, instruments=universe)
    print(f"[data] {meta['I']} x {meta['D']} x {meta['NM']}  ({time.time()-t0:.0f}s)")

    # rebuild forward return to match period (affects IC computation)
    close_d = tens['close'][:, :, -1]
    if args.period > 1:
        fwd_ret = torch.full_like(close_d, float('nan'))
        fwd_ret[:, :-args.period] = close_d[:, args.period:] / close_d[:, :-args.period] - 1.0

    # actual daily returns for layer PnL (NOT forward N-day)
    daily_ret = torch.full_like(close_d, float('nan'))
    daily_ret[:, 1:] = close_d[:, 1:] / close_d[:, :-1] - 1.0

    ctx = Ctx(tens, masks, meta, device=device)

    factor = ctx.eval(node)
    backtest_weights = None

    # ── universe pool (point-in-time CSI 500, with component weights) ──
    pool_csv = args.pool_csv
    if pool_csv and os.path.isfile(pool_csv):
        pool_data = load_component_pool(pool_csv, top_n=500)
        pm, bw = build_pool_mask(pool_data, meta["dates"], meta["instruments"], device)
        factor = torch.where(pm, factor, torch.full_like(factor, float("nan")))
        n_in = pm.sum(1).clamp(max=1).sum().item()
        backtest_weights = bw
        print(f"[pool] CSI500 {len(pool_data)} monthly snapshots, {n_in}/{meta['I']} instruments ever in pool")
    elif args.pit and os.path.isfile(args.pit):
        pm = load_pit_daily_mask(args.pit, meta["dates"], meta["instruments"], device)
        factor = torch.where(pm, factor, torch.full_like(factor, float("nan")))
        n_in = pm.any(1).sum().item()
        print(f"[pool] CSI500 daily PIT: {n_in}/{meta['I']} instruments ever in pool")
    elif args.pool and os.path.isdir(args.pool):
        pools = load_pool(args.pool)
        pm, _ = pool_mask(pools, meta["dates"], meta["instruments"])
        factor = torch.where(pm.to(factor.device), factor, torch.full_like(factor, float("nan")))
        n_in = pm.sum(1).clamp(max=1).sum().item()
        print(f"[pool] legacy CSI500 weekly snapshots: {n_in}/{meta['I']} instruments ever in pool")

    # ── industry neutralization ──
    if args.industry:
        ind_map = load_industry(args.industry)
        ind_ids = industry_ids(ind_map, meta["instruments"])
        factor = industry_neutral(factor, ind_ids.to(factor.device))
        print(f"[neutral] industry-neutralized ({int(ind_ids.max().item())+1} buckets incl. unknown)")

    # ── market cap neutralization ──
    if args.mcap:
        mcap_dict = load_market_cap(args.mcap)
        mcap_t = build_mcap_tensor(mcap_dict, meta["dates"], meta["instruments"], factor.device)
        factor = market_cap_neutral(factor, mcap_t)
        print(f"[neutral] market-cap neutralized")

    ic = daily_spearman_ic(factor, fwd_ret)
    s = summarize_ic(ic)
    # negative-IC factors: flip sign so Layer 1 = best predicted return (全样本统一方向)
    if s["ic_mean"] < 0:
        factor = -factor
        print(f"[factor {name}] IC_mean={s['ic_mean']:.4f} < 0 → factor flipped (L1 = highest predicted ret)")
    else:
        print(f"[factor {name}] IC_mean={s['ic_mean']:.4f} (positive, no flip)")
    print(f"ICIR={s['icir']:.2f} n={s['n_days']}")

    # daily close-to-close returns for layer PnL (signal/return separation)
    close_d = tens['close'][:, :, -1]
    dret = torch.full_like(close_d, float('nan'))
    dret[:, 1:] = close_d[:, 1:] / close_d[:, :-1] - 1.0
    # layers EQUAL-WEIGHT; CSI500 weights only for benchmark line
    rets, costs, turn = backtest(factor, dret, layers=args.layers, fee=args.fee, turnover=args.turnover,
                                 period=args.period, weights=None)
    net = rets - costs
    for L in range(args.layers):
        to_rebal = turn[L][::args.period]
        avg_to = torch.nan_to_num(to_rebal[to_rebal > 0]).mean().item()
        cost_rebal = costs[L][::args.period]
        avg_cost = torch.nan_to_num(cost_rebal[cost_rebal > 0]).mean().item()
        print(f"layer {L+1}: avg turnover {avg_to*100:.0f}%/rebalance, avg cost {avg_cost*100:.2f}%/rebalance")
    print(f"[cost] fee={args.fee}/rt (千{args.fee*1000:.0f}), actual turnover (period={args.period}d)")
    dates = meta["dates"]

    for L in range(args.layers):
        r = net[L]  # keep NaN (suspended days)
        cum = torch.nan_to_num(torch.cumprod(1 + torch.nan_to_num(r, nan=0), 0) - 1) * 100
        st = stats(cum, r)
        print(f"[layer {L+1}] CAGR={st['cagr']:.1f}%  Sharpe={st['sharpe']:.2f}  MaxDD={st['mdd']:.1f}%  Win={st['win']:.1f}%")
    ls = torch.nan_to_num(net[0] - net[-1])
    cum_ls = torch.nan_to_num(torch.cumprod(1 + torch.nan_to_num(ls, nan=0), 0) - 1) * 100
    st = stats(cum_ls, ls)
    print(f"[LS L1-L{args.layers}] CAGR={st['cagr']:.1f}%  Sharpe={st['sharpe']:.2f}  MaxDD={st['mdd']:.1f}%  Win={st['win']:.1f}%")

    # ── IR vs CSI500 benchmark (daily excess) ──
    if backtest_weights is not None and backtest_weights.ndim == 2:
        daily_ret = close_d[:, 1:] / close_d[:, :-1] - 1.0
        bw_a = backtest_weights[:, 1:min(backtest_weights.shape[1], daily_ret.shape[1]+1)]
        bw_a = bw_a / bw_a.sum(0).clamp(min=1e-12)
        bench_daily = (torch.nan_to_num(daily_ret[:, :bw_a.shape[1]-1]) * bw_a[:, 1:]).sum(0)
        l1_d = net[0][1:bw_a.shape[1]]
        excess = l1_d - bench_daily
        excess = excess[~torch.isnan(excess)]
        if excess.numel() > 20:
            daily_ir = excess.mean().item() / excess.std().item()
            ann_ir = daily_ir * np.sqrt(252)
            print(f"[IR vs CSI500] daily_IR={daily_ir:.4f}  annual_IR={ann_ir:.2f}  n={excess.numel()}d")
        # benchmark on daily grid for plotting
        bench_grid = torch.full_like(net[0], float('nan'))
        bench_grid[1:bw_a.shape[1]] = bench_daily[:bw_a.shape[1]-1]
    else:
        bench_grid = None

    plot_nav(dates, net, args.layers, None, args.out or f"min_gp/backtest_p{args.period}.png", bench=bench_grid)
    plot_ic(dates, ic, f"min_gp/backtest_ic_p{args.period}.png")


if __name__ == "__main__":
    main()
