"""
Standalone test-set evaluation for valid-passed factors.
Loads passed_valid.txt from engine.py, evaluates on held-out test period.

Usage:
  python -m min_gp.eval_test --start 2022-01-02 --end 2023-12-31 --period 5
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, ".")
from min_gp.data import build_slice, load_index_codes, load_pit_daily_mask
from min_gp.expr import Ctx, parse
from min_gp.fitness import daily_spearman_ic, summarize_ic, multi_fitness, factor_health
from min_gp.config import (MINUTE_PARQUET, ZZ500_PIT_PARQUET, output_path,
                           require_path)

PARQUET = str(MINUTE_PARQUET)


def parse_valid_record(line):
    """Return (expr, raw_valid_ic, train_direction) for a saved factor."""
    line = line.strip()
    if not line or line.startswith("#") or "  " not in line:
        return "", float("nan"), 0
    header, expr = line.rsplit("  ", 1)
    raw_ic, direction = float("nan"), 0
    for token in header.split():
        try:
            if token.startswith("vIC="):
                raw_ic = float(token.split("=", 1)[1])
            elif token.startswith("direction="):
                direction = int(token.split("=", 1)[1])
        except ValueError:
            pass
    return expr.strip(), raw_ic, direction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-01-02")
    ap.add_argument("--end", default="2026-07-31")
    ap.add_argument("--period", type=int, default=5,
                    help="forward return horizon (1=daily, 5=weekly)")
    ap.add_argument("--exprs", default=str(output_path("valid_pass.txt")),
                    help="factors to evaluate (vIC=... expr per line)")
    ap.add_argument("--parquet", default=PARQUET)
    ap.add_argument("--pit", default=str(ZZ500_PIT_PARQUET))
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = "cpu" if args.cpu else "cuda"
    parquet_path = require_path(args.parquet, "minute parquet")
    pit_path = require_path(args.pit, "CSI500 PIT parquet")

    if not os.path.isfile(args.exprs):
        sys.exit(f"file not found: {args.exprs}")

    # Load the direction fixed on train and carried through validation.
    directions = {}
    exprs = []
    with open(args.exprs) as f:
        for line in f:
            expr, _, direction = parse_valid_record(line)
            if expr and direction in (-1, 1) and expr not in directions:
                directions[expr] = direction
                exprs.append(expr)

    if not exprs:
        sys.exit("no direction-aware expressions found; rerun mining/validation with the new format")

    print(f"[test] {len(exprs)} expressions from {args.exprs}")
    print(f"[test] period: {args.start} ~ {args.end}, target=t+{args.period}")

    # 与挖掘同口径: 成分池 (CSI500 全历史成员) 加载 + 日频 PIT mask + 停牌剔除
    zz500_codes = load_index_codes(pit_path)
    tens, masks, fwd_ret, meta = build_slice(parquet_path, args.start, args.end, device=device,
                                             instruments=zz500_codes)
    pm = load_pit_daily_mask(pit_path, meta["dates"],
                             meta["instruments"], device)
    pm = pm & ~torch.isnan(tens['close'][:, :, -1])
    miss = torch.isnan(tens['close']).float().mean(dim=2)
    ctx = Ctx(tens, masks, meta, device=device)

    if args.period > 1:
        from min_gp.label import tensor_fwd_ret
        close_d = tens['close'][:, :, -1]
        # 重叠标签 (与挖掘同口径): fwd[t] = close[t+p]/close[t]-1
        # (close 入场 = 收盘前几分钟执行; 其他口径见 min_gp.label.LABEL);
        # 重叠使 IC 序列自相关 → ICIR 虚高 → 报告时 Newey-West 修正 (见下)
        fwd_ret = tensor_fwd_ret(close_d, close_d, args.period)

    print(f"[test] data: {meta['I']} x {meta['D']} x {meta['NM']}")

    # evaluate
    results = []
    print(f"\n{'#':>3s} {'|IC|':>8s} {'win':>7s} {'ndcg':>7s} {'tIC':>8s} {'tICIR':>7s} {'to':>7s}  expr")
    for i, expr_str in enumerate(exprs):
        try:
            direction = directions[expr_str]
            node = parse(expr_str)
            factor = ctx.eval(node)
            factor = torch.where(pm, factor, torch.full_like(factor, float("nan")))
            ic = daily_spearman_ic(factor, fwd_ret)
            s = summarize_ic(ic)
            obj = multi_fitness(factor, fwd_ret, direction=direction)
            raw_tIC = s['ic_mean']
            aligned_tIC = direction * raw_tIC
            ok_health, hrep = factor_health(factor, miss)
            # Newey-West 修正 ICIR: 重叠标签 (t+period) 使 IC 序列自相关 → 原 ICIR 虚高
            icir = s['icir']
            if args.period > 1:
                import numpy as _np
                a = ic[~torch.isnan(ic)].cpu().numpy()
                if len(a) > args.period:
                    L = args.period - 1
                    rhos = []
                    for _k in range(1, min(L + 1, len(a) // 2)):
                        _r = _np.corrcoef(a[:-_k], a[_k:])[0, 1]
                        rhos.append(_r if _np.isfinite(_r) else 0.0)
                    _w = 1 + 2 * sum((1 - _k / (L + 1)) * rhos[_k - 1]
                                     for _k in range(1, len(rhos) + 1))
                    _var_nw = a.std() ** 2 * max(_w, 0.1)
                    if _var_nw > 0:
                        icir = float(a.mean() / _np.sqrt(_var_nw))
            aligned_icir = direction * icir
            print(f"{i:3d} {obj[0]:8.4f} {obj[1]:7.3f} {obj[2]:7.4f} "
                  f"{raw_tIC:8.4f} {aligned_tIC:8.4f} dir={direction:+d} "
                  f"{aligned_icir:7.2f} {-obj[3]:7.3f}  "
                  f"{'OK ' if ok_health else 'DROP'} "
                  f"zero={hrep['zero_frac']:.2f} miss={hrep['miss_corr_med']:+.2f}  {expr_str[:60]}")
            # Direction remains fixed from train; test only confirms persistence.
            # n_days 约束: 单日/极少日 IC 均值可 >0.05 (幽灵因子: test 段退化到 n_days=1),
            # 重叠标签下 test 段 ~381 个样本, 阈值 60
            if raw_tIC is not None and aligned_tIC > 0.05 and ok_health and s['n_days'] >= 60:
                results.append((aligned_tIC, raw_tIC, aligned_icir, direction, expr_str))
        except Exception as e:
            print(f"{i:3d} {'-':>8s} {'-':>7s} {'-':>7s} {'-':>8s} {'-':>7s} {'-':>7s}  ERR {type(e).__name__}")

    # save final factor pool
    if results:
        results.sort(key=lambda x: x[0], reverse=True)
        test_pass_file = output_path("test_pass.txt")
        with open(test_pass_file, "a", encoding="utf-8") as f:
            for aligned_tIC, raw_tIC, tICIR, direction, expr in results:
                f.write(
                    f"tIC={raw_tIC:.6f} direction={direction:+d} "
                    f"alignedTIC={aligned_tIC:.6f} tICIR={tICIR:.4f}  {expr}\n")
        print(f"\n[test] {len(results)}/{len(exprs)} factors passed → min_gp/test_pass.txt")
    else:
        print("\n[test] no factors survived (aligned tIC<=0.05 or health/n_days failed)")


if __name__ == "__main__":
    main()
