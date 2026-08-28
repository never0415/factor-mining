"""Disjoint hold-out evaluator for second-layer daily typed trees."""

import argparse
import json

import torch

from min_gp.config import (
    ADJUSTED_CLOSE_PARQUET, INDUSTRY_VALUE_EXPOSURES_PARQUET,
    ZZ500_PIT_PARQUET, output_path, require_path,
)
from min_gp.daily_gp import DEFAULT_LEAVES, parse_leaf
from min_gp.data import load_pit_codes, load_pit_daily_mask, load_pit_dates
from min_gp.evaluation import BatchedNeutralizer, DEFAULT_COST_BPS
from min_gp.gp.typed_tree import TypedTreeGenome
from min_gp.holdout_eval import (
    assert_disjoint, evaluate_factor, load_candidates,
    resolve_period, resolve_rebalance, resolve_signal_average_days,
)
from min_gp.label import tensor_rebalance_fwd_ret
from min_gp.operators import build_operator_registry
from min_gp.spectral_data import (
    load_daily_close_tensor, load_daily_exposures, load_daily_factor_leaves,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--leaf", action="append", default=[])
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--period", type=int, default=None)
    parser.add_argument(
        "--rebalance", choices=("week_end", "daily"), default=None
    )
    parser.add_argument("--signal-average-days", type=int, default=None)
    parser.add_argument("--max-rank", type=int, default=1)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--daily-parquet", default=str(ADJUSTED_CLOSE_PARQUET))
    parser.add_argument("--pit", default=str(ZZ500_PIT_PARQUET))
    parser.add_argument("--exposures", default=str(INDUSTRY_VALUE_EXPOSURES_PARQUET))
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--min-cross-section", type=int, default=30)
    parser.add_argument("--min-ic-days", type=int, default=40)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--quantile", type=float, default=.2)
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS,
                        help="round-trip cost in bp charged on traded "
                             "notional; 30 = 0.15%% per side")
    parser.add_argument("--min-ic", type=float, default=0.)
    parser.add_argument("--min-consistency", type=float, default=.75)
    parser.add_argument("--min-coverage", type=float, default=.25)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    device = "cpu" if args.cpu else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; pass --cpu")
    records = load_candidates(args.candidates, args.max_rank, args.limit)
    if not records:
        raise SystemExit("no valid daily-tree candidates")
    if any(record.get("family") != "daily_tree" for record in records):
        raise SystemExit("daily_holdout only accepts family=daily_tree")
    args.period = resolve_period(records, args.period)
    args.rebalance = resolve_rebalance(records, args.rebalance)
    args.signal_average_days = resolve_signal_average_days(
        records, args.signal_average_days
    )
    assert_disjoint(records, args.start)
    neutralized = {bool(record.get("provenance", {}).get("neutralized")) for record in records}
    if len(neutralized) != 1:
        raise SystemExit("candidate file mixes neutralized and raw trees")

    pit = require_path(args.pit, "PIT universe")
    dates = load_pit_dates(pit, args.start, args.end)
    instruments = load_pit_codes(pit, args.start, args.end)
    specs = [parse_leaf(value) for value in args.leaf] or DEFAULT_LEAVES
    leaves = {}
    for name, path, column in specs:
        leaves[name] = load_daily_factor_leaves(
            require_path(path, f"factor leaf {name}"), dates, instruments,
            [column], device=device,
        )[column]
    close = load_daily_close_tensor(
        require_path(args.daily_parquet, "adjusted daily close"),
        dates, instruments, device=device,
    )
    fwd_ret = tensor_rebalance_fwd_ret(
        close, dates, args.rebalance, args.period
    )
    pool = load_pit_daily_mask(pit, dates, instruments, device=device)
    pool &= torch.isfinite(close)
    fwd_ret = torch.where(pool, fwd_ret, torch.full_like(fwd_ret, float("nan")))
    continuous, industry = (), None
    if neutralized.pop():
        styles, industry, _ = load_daily_exposures(
            require_path(args.exposures, "industry/value exposures"),
            dates, instruments, device=device,
        )
        continuous = tuple(styles.values())
    registry = build_operator_registry()
    neutralizer = None
    if continuous or industry is not None:
        neutralizer = BatchedNeutralizer(
            fwd_ret.shape, continuous, industry, args.min_cross_section
        )

    def transform(value):
        if neutralizer is not None:
            value = neutralizer(value)
        return torch.where(pool, value, torch.full_like(value, float("nan")))

    anchor = transform(leaves[sorted(leaves)[0]])
    missing_rate = torch.stack([torch.isnan(value).float() for value in leaves.values()]).mean(0)
    results = []
    for record in records:
        genome = TypedTreeGenome.from_dict(record["genome"], registry)
        factor = transform(genome.evaluate(leaves, registry))
        direction = int(record["fitness"]["direction"])
        metrics = evaluate_factor(
            factor, anchor, fwd_ret, direction, args, missing_rate
        )
        if metrics is not None:
            results.append({
                "family": "daily_tree", "direction": direction,
                "genome": genome.to_dict(), "expression": str(genome),
                "holdout": metrics, "train_fitness": record.get("fitness"),
                "provenance": record.get("provenance"),
                "holdout_window": [args.start, args.end], "period": args.period,
                "rebalance_rule": args.rebalance,
                "signal_average_days": args.signal_average_days,
            })
    destination = args.out or str(output_path("daily_holdout.jsonl"))
    with open(destination, "w", encoding="utf-8") as handle:
        for row in sorted(results, key=lambda value: -value["holdout"]["aligned_ic"]):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[daily-holdout] {len(results)} candidates -> {destination}")


if __name__ == "__main__":
    main()
