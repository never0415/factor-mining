"""
Validate all seed factors: parse → type-check → GPU eval → daily Spearman IC.
Usage: python -m min_gp.run_seeds [--start 2018-01-02] [--end 2018-12-31] [--cpu]
"""
import argparse
import sys
import time

import torch

sys.path.insert(0, ".")
from min_gp.data import build_slice
from min_gp.expr import Ctx, TypeTagError
from min_gp.seeds import all_seeds, PAPER_DIR
from min_gp.fitness import daily_spearman_ic, summarize_ic
from min_gp.config import MINUTE_PARQUET

PARQUET = str(MINUTE_PARQUET)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-02")
    ap.add_argument("--end", default="2018-12-31")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--parquet", default=PARQUET)
    args = ap.parse_args()

    device = "cpu" if args.cpu else "cuda"
    t0 = time.time()
    tens, masks, fwd_ret, meta = build_slice(args.parquet, args.start, args.end, device=device)
    print(f"[data] {meta['I']} inst x {meta['D']} days x {meta['NM']} min  ({time.time()-t0:.1f}s)", flush=True)
    ctx = Ctx(tens, masks, meta, device=device)

    seeds = all_seeds()
    rows = []
    n_ok = n_fail = 0
    for name, node in seeds.items():
        t1 = time.time()
        try:
            factor = ctx.eval(node)
            ic = daily_spearman_ic(factor, fwd_ret)
            s = summarize_ic(ic)
            rows.append((name, s["ic_mean"], s["icir"], s["n_days"], ""))
            n_ok += 1
        except TypeTagError as e:
            rows.append((name, float("nan"), float("nan"), 0, f"TYPEFAIL: {e}"))
            n_fail += 1
        except Exception as e:
            rows.append((name, float("nan"), float("nan"), 0, f"ERR: {type(e).__name__}: {e}"))
            n_fail += 1
        print(f"[seed] {name:26s} {time.time()-t1:5.1f}s", flush=True)

    print(f"\n{'seed':26s} {'IC_mean':>9s} {'ICIR':>7s} {'days':>5s}  dir vs paper")
    for name, icm, icir, nd, err in rows:
        if err:
            print(f"{name:26s} {'-':>9s} {'-':>7s} {'-':>5s}  {err}")
            continue
        paper = PAPER_DIR.get(name)
        if paper is None:
            print(f"{name:26s} {icm:9.4f} {icir:7.2f} {nd:5d}  (no paper dir)")
            continue
        match = "✓" if ((icm > 0) == (paper > 0)) else "✗"
        print(f"{name:26s} {icm:9.4f} {icir:7.2f} {nd:5d}  {match} (paper {paper})")
    print(f"\nparsed+evaluated {n_ok}/{len(seeds)}, failed {n_fail}")


if __name__ == "__main__":
    main()
