"""
Backtrader bridge for min_gp daily factor backtesting.
GPU computes daily factor → per-stock daily DataFrames (OHLCV + factor) → backtrader.

Usage:
  python -m min_gp.bt_run --seed s27_valley_vwap --start 2018-01-02 --end 2019-12-31
  python -m min_gp.bt_run --seed s17_err --period 5 --top 0.2 --fee 0.3
"""
import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")          # MUST be before any other matplotlib import
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import torch

import backtrader as bt

sys.path.insert(0, ".")
from min_gp.data import (build_slice, load_industry, industry_ids,
                         load_index_codes, load_pit_daily_mask)
from min_gp.expr import Ctx, parse, _cs_rank
from min_gp.fitness import daily_spearman_ic, industry_neutral
from min_gp.seeds import SEEDS
from min_gp.config import (MINUTE_PARQUET, ZZ500_PIT_PARQUET,
                           DAILY_PRICE_PARQUET)

PARQUET = str(MINUTE_PARQUET)
PIT_PQ = str(ZZ500_PIT_PARQUET)
IND_PQ = ""
DAY_PQ = str(DAILY_PRICE_PARQUET)


# ═══════════════════════════════════════════════════════
# 1. Build per-stock daily DataFrames (OHLCV + factor)
# ═══════════════════════════════════════════════════════

def build_daily_dfs(factor, meta, start, end, max_stocks=200, day_pq=DAY_PQ):
    """Convert (I,D) factor tensor + daily OHLCV → list of clean DataFrames.

    Each DataFrame has index=datetime, columns=open,high,low,close,volume,factor.
    All stocks use the SAME date index (union of all trading days), with NaN-filled
    rows for suspension days — backtrader needs aligned bars across all feeds.
    Limited to max_stocks most frequently in pool.
    """
    import pyarrow.parquet as pq
    insts = meta["instruments"]
    dates = meta["dates"]
    I = len(insts)
    fnp = factor.cpu().numpy()

    # which stocks to load (most frequently in pool)
    valid_count = (~np.isnan(fnp)).sum(1)
    top_i = np.argsort(-valid_count)[:max_stocks]
    keep_insts = [insts[i] for i in top_i]

    # load daily OHLCV
    filters = [("trade_date", ">=", start), ("trade_date", "<=", end)]
    t = pq.read_table(day_pq, filters=filters)
    pdf = t.to_pandas()
    pdf["trade_date"] = pd.to_datetime(pdf["trade_date"])

    date_idx = {d: j for j, d in enumerate(dates)}

    dfs = []
    for i, inst in zip(top_i, keep_insts):
        sub = pdf.loc[pdf["instrument"] == inst].copy()
        if sub.empty:
            continue
        sub.set_index("trade_date", inplace=True)
        sub.sort_index(inplace=True)
        sub = sub[["open", "high", "low", "close", "volume"]].copy()
        # add factor
        fvals = [fnp[i, date_idx[d.strftime("%Y-%m-%d")]]
                 if d.strftime("%Y-%m-%d") in date_idx else np.nan
                 for d in sub.index]
        sub["factor"] = np.array(fvals, dtype=np.float64)
        sub.dropna(subset=["factor"], inplace=True)
        if len(sub) < 20:
            continue
        # keep natural date index — backtrader aligns feeds, skips bars where
        # stocks are suspended. No fill (bfill = future function, ffill = stale prices).
        dfs.append((inst, sub))
    return dfs


# ═══════════════════════════════════════════════════════
# 2. Backtrader data feed + strategy
# ═══════════════════════════════════════════════════════

class FactorData(bt.feeds.PandasData):
    """OHLCV + daily factor line."""
    lines = ("factor",)
    params = (
        ("datetime", None),
        ("open", 0),
        ("high", 1),
        ("low", 2),
        ("close", 3),
        ("volume", 4),
        ("factor", 5),
        ("openinterest", -1),
    )


class FactorStrategy(bt.Strategy):
    """Daily cross-sectional factor: rank all stocks, buy top pct, equal-weight.

    Args:
        top_pct: fraction long (default 0.2 = top quintile)
        period: rebalance every N bars
    """
    params = (
        ("top_pct", 0.2),
        ("period", 1),
    )

    def __init__(self):
        self.bar_count = 0
        self._name_to_data = {d._name: d for d in self.datas}
        self._nav_history = []  # (bar_idx, date, nav)

    def start(self):
        # record initial NAV before any bar
        self._nav_history.append((0, None, self.broker.getvalue()))

    def next(self):
        self.bar_count += 1
        # record NAV at bar close (before any orders placed in this next() call)
        self._nav_history.append((self.bar_count,
                                   self.datas[0].datetime.date(0).isoformat(),
                                   self.broker.getvalue()))
        if self.bar_count % self.p.period != 0:
            return

        # collect factor values from all live feeds
        items = []
        for d in self.datas:
            if len(d) > 0:
                fv = float(d.factor[0])
                if fv == fv:  # not NaN
                    items.append((d._name, fv))
        if len(items) < 5:
            return

        # rank descending (higher factor = better predicted return)
        items.sort(key=lambda x: x[1], reverse=True)
        n_hold = max(1, int(len(items) * self.p.top_pct))
        target = {name for name, _ in items[:n_hold]}

        # sell non-target
        for name, d in self._name_to_data.items():
            pos = self.getposition(d)
            if pos and name not in target:
                self.close(data=d)

        # buy/equal-weight target
        value = self.broker.getvalue()
        cash_per = value / max(len(target), 1)
        for name in target:
            try:
                d = self._name_to_data[name]
                if len(d) < 1:
                    continue
                price = d.close[0]
                if isinstance(price, float):
                    pass
                else:
                    price = float(price) if hasattr(price, '__float__') else None
                if price is None or not (price > 0):
                    continue
                pos = self.getposition(d)
                target_val = cash_per
                current_val = pos.size * price if pos else 0
                diff_val = target_val - current_val
                lot_val = price * 100
                if abs(diff_val) < lot_val * 0.5:
                    continue
                size = int(abs(diff_val) / price / 100) * 100
                if size <= 0:
                    continue
                if diff_val > 0:
                    self.buy(data=d, size=size)
                else:
                    self.sell(data=d, size=min(size, pos.size))
            except (ValueError, TypeError, KeyError):
                continue

    def notify_order(self, order):
        if order.status in (order.Completed,):
            pass
        elif order.status in (order.Canceled, order.Margin, order.Rejected):
            pass  # ignore partial fills


# ═══════════════════════════════════════════════════════
# 3. Run
# ═══════════════════════════════════════════════════════

def run_bt(dfs, fee_pct=0.3, period=1, top_pct=0.2, cash=1_000_000,
           plot=True, out_png=None):
    """Run backtrader Cerebro with per-stock factor data."""
    t0 = time.time()
    cerebro = bt.Cerebro()

    cerebro.addstrategy(FactorStrategy, top_pct=top_pct, period=period)

    for inst, df in dfs:
        data = FactorData(dataname=df, name=inst)
        cerebro.adddata(data)

    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=fee_pct / 100.0)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.0,
                        timeframe=bt.TimeFrame.Days, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    print(f"[bt] {len(dfs)} stocks, fee={fee_pct}%/side, period={period}d, top={top_pct*100:.0f}%, "
          f"running...", flush=True)
    results = cerebro.run()
    strat = results[0]
    elapsed = time.time() - t0
    print(f"[bt] done in {elapsed:.0f}s")

    # stats
    sharpe = strat.analyzers.sharpe.get_analysis()
    dd = strat.analyzers.dd.get_analysis()
    ret = strat.analyzers.returns.get_analysis()
    trades = strat.analyzers.trades.get_analysis()

    final_val = cerebro.broker.getvalue()
    total_ret = (final_val / cash - 1) * 100
    print(f"[bt] final: {final_val:,.0f}  ({total_ret:+.1f}%)")
    sr = sharpe.get("sharperatio")
    if sr is not None:
        print(f"[bt] Sharpe: {sr:.2f}")
    mdd_pct = dd.get("max", {}).get("drawdown", 0)
    if mdd_pct:
        print(f"[bt] MaxDD: {mdd_pct:.1f}%")
    rtot = ret.get("rtot")
    if rtot is not None:
        n_years = max(len(dfs[0][1]) / 252 if dfs else 1, 0.5)
        cagr = ((1 + rtot) ** (1 / n_years) - 1) * 100
        print(f"[bt] CAGR: {cagr:.1f}%")
    total_trades = trades.get("total", {}).get("total", 0)
    if total_trades:
        print(f"[bt] Trades: {total_trades}")

    if plot:
        try:
            nav_hist = strat._nav_history
            if len(nav_hist) > 1:
                dates_bt = [h[1] for h in nav_hist[1:]]  # skip initial
                navs = np.array([h[2] for h in nav_hist])
                # daily returns from consecutive NAVs
                rets = np.diff(navs) / navs[:-1]
                cum_pct = (navs[1:] / navs[0] - 1.0) * 100
                dd_series = np.minimum.accumulate(navs[1:])
                dd_pct = (dd_series - navs[1:]) / dd_series * 100

                print(f"[bt] nav: {len(dates_bt)} bars, first 5 ret: {[f'{v*100:.3f}%' for v in rets[:5]]}, "
                      f"last 5 ret: {[f'{v*100:.3f}%' for v in rets[-5:]]}", flush=True)

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                               gridspec_kw={"height_ratios": [3, 1]})
                dts = [pd.Timestamp(d) for d in dates_bt]
                ax1.plot(dts, cum_pct, color="#4C72B0", lw=1.5, label="NAV")
                ax1.axhline(0, color="gray", lw=0.8)
                ax1.fill_between(dts, 0, cum_pct,
                                 where=(np.array(cum_pct) >= 0), color="#4C72B0", alpha=0.15)
                ax1.fill_between(dts, 0, cum_pct,
                                 where=(np.array(cum_pct) < 0), color="#C44E52", alpha=0.15)
                ax1.set_ylabel("Cumulative Return (%)")
                ax1.set_title(f"Backtrader — long top {top_pct*100:.0f}%, period={period}d, fee={fee_pct}%/side")
                ax1.legend(loc="upper left")
                ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                fig.autofmt_xdate()

                ax2.fill_between(dts, 0, dd_pct, color="#C44E52", alpha=0.35)
                ax2.plot(dts, dd_pct, color="#C44E52", lw=1.0)
                ax2.set_ylabel("Drawdown (%)")
                ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                fig.autofmt_xdate()

                out = out_png or "min_gp/bt_backtest.png"
                fig.tight_layout()
                fig.savefig(out, dpi=130)
                plt.close(fig)
                print(f"[bt] plot → {out}")
            else:
                print("[bt] no NAV history to plot")
        except Exception as e:
            print(f"[bt] plot failed: {e}")
            import traceback; traceback.print_exc()

    return results


# ═══════════════════════════════════════════════════════
# 4. CLI
# ═══════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default=None)
    ap.add_argument("--expr", default=None)
    ap.add_argument("--start", default="2018-01-02")
    ap.add_argument("--end", default="2019-06-30")
    ap.add_argument("--fee", type=float, default=0.3, help="fee per side in %%")
    ap.add_argument("--period", type=int, default=1, help="rebalance period in days")
    ap.add_argument("--top", type=float, default=0.2, help="fraction long")
    ap.add_argument("--cash", type=float, default=1_000_000)
    ap.add_argument("--max-stocks", type=int, default=100,
                    help="max stocks for backtrader feeds (speed)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--parquet", default=PARQUET)
    ap.add_argument("--pit", default=PIT_PQ)
    ap.add_argument("--daily", default=DAY_PQ)
    ap.add_argument("--industry", default=IND_PQ)
    args = ap.parse_args()

    if not args.seed and not args.expr:
        sys.exit("need --seed or --expr")

    device = "cpu" if args.cpu else "cuda"
    name = args.seed or "expr"

    # 1. GPU factor
    t0 = time.time()
    universe = load_index_codes(args.pit)
    tens, masks, fwd_ret, meta = build_slice(
        args.parquet, args.start, args.end, device=device, instruments=universe)
    ctx = Ctx(tens, masks, meta, device=device)
    if args.expr:
        node = parse(args.expr)
    else:
        node = parse(SEEDS[args.seed])
    factor = ctx.eval(node)

    # pool + neutral
    pm = load_pit_daily_mask(args.pit, meta["dates"], meta["instruments"], device)
    factor = torch.where(pm, factor, torch.full_like(factor, float("nan")))
    if args.industry and os.path.isfile(args.industry):
        ind_ids = industry_ids(load_industry(args.industry), meta["instruments"]).to(device)
        factor = industry_neutral(factor, ind_ids)

    # auto-flip
    ic = daily_spearman_ic(factor, fwd_ret)
    ic_mean = torch.nanmean(ic).item()
    if ic_mean < 0:
        factor = -factor
        print(f"[factor {name}] IC={ic_mean:+.4f} → flipped", flush=True)
    else:
        print(f"[factor {name}] IC={ic_mean:+.4f}", flush=True)
    print(f"[data] {meta['I']}x{meta['D']} ({time.time()-t0:.0f}s)", flush=True)

    # 2. Build daily per-stock DataFrames
    dfs = build_daily_dfs(
        factor, meta, args.start, args.end, max_stocks=args.max_stocks,
        day_pq=args.daily)
    print(f"[data] {len(dfs)} stock DataFrames ready", flush=True)
    if not dfs:
        sys.exit("no stocks with both factor + OHLCV")

    # 3. Backtrader
    run_bt(dfs, fee_pct=args.fee, period=args.period, top_pct=args.top,
           cash=args.cash, plot=not args.no_plot, out_png=args.out)


if __name__ == "__main__":
    main()
