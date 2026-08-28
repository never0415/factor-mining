"""CLI for the second-layer typed GP over cached daily factor leaves."""

import argparse
import json

import torch

from min_gp.config import (
    ADJUSTED_CLOSE_PARQUET, DRIPPING_STONE_STAGE1_DIR,
    INDUSTRY_VALUE_EXPOSURES_PARQUET, LIQUIDITY_ELASTICITY_DIR,
    RUSHING_FORWARD_DIR, VOLUME_ENTROPY_DIR, ZZ500_PIT_PARQUET,
    output_path, require_path,
)
from min_gp.data import load_pit_codes, load_pit_daily_mask, load_pit_dates
from min_gp.evaluation import (
    DEFAULT_COST_BPS, FactorArchive, WalkForwardConfig, trailing_signal_mean,
)
from min_gp.gp import DailyFactorGP, DailyGPConfig
from min_gp.label import tensor_rebalance_fwd_ret
from min_gp.preflight import check_or_exit
from min_gp.spectral_data import (
    load_daily_close_tensor, load_daily_exposures, load_daily_factor_leaves,
)
from min_gp.experiment import atomic_json, experiment_directory


DEFAULT_LEAVES = (
    ("dripping_stone", DRIPPING_STONE_STAGE1_DIR, "ds_raw_all"),
    ("liquidity_elasticity", LIQUIDITY_ELASTICITY_DIR / "factor.parquet", "factor"),
    ("rushing_forward", RUSHING_FORWARD_DIR / "factor.parquet", "factor"),
    ("volume_entropy", VOLUME_ENTROPY_DIR / "factor.parquet", "factor"),
)


def parse_leaf(value):
    """NAME=PATH:COLUMN; rsplit preserves the drive-letter colon on Windows."""
    name, source = value.split("=", 1)
    path, column = source.rsplit(":", 1)
    return name, path, column


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaf", action="append", default=[], help="NAME=PATH:COLUMN")
    parser.add_argument("--start", default="2018-01-02")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--period", type=int, default=1)
    parser.add_argument(
        "--rebalance", choices=("week_end", "daily"), default="week_end",
    )
    parser.add_argument("--signal-average-days", type=int, default=None)
    parser.add_argument("--pop", type=int, default=80)
    parser.add_argument("--gens", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--direction", choices=("paper", "discovery"), default="discovery")
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS,
                        help="round-trip cost in bp charged on traded "
                             "notional; 30 = 0.15%% per side")
    parser.add_argument("--min-train-days", type=int, default=504)
    parser.add_argument("--valid-days", type=int, default=126)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--min-valid-ic-days", type=int, default=None)
    parser.add_argument("--min-cross-section", type=int, default=30)
    parser.add_argument("--min-fold-consistency", type=float, default=.75)
    parser.add_argument("--min-folds", type=int, default=3)
    parser.add_argument("--neutralize", action="store_true")
    parser.add_argument("--max-correlation", type=float, default=0.85)
    parser.add_argument("--daily-parquet", default=str(ADJUSTED_CLOSE_PARQUET))
    parser.add_argument("--pit", default=str(ZZ500_PIT_PARQUET))
    parser.add_argument("--exposures", default=str(INDUSTRY_VALUE_EXPOSURES_PARQUET))
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-stocks", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume and not args.run_id:
        parser.error("--resume requires the original --run-id")
    args.signal_average_days = args.signal_average_days or (
        5 if args.rebalance == "week_end" else 1
    )
    args.min_valid_ic_days = args.min_valid_ic_days or (
        12 if args.rebalance == "week_end" else 40
    )
    run_dir = experiment_directory("daily_tree", args.run_id)
    atomic_json(run_dir / "config.json", vars(args))

    device = "cpu" if args.cpu else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; pass --cpu")
    pit = require_path(args.pit, "PIT universe")
    daily = require_path(args.daily_parquet, "adjusted daily close")
    instruments = load_pit_codes(pit, args.start, args.end)
    if args.max_stocks:
        instruments = instruments[:args.max_stocks]
    dates = load_pit_dates(pit, args.start, args.end)
    fitness_config = WalkForwardConfig(
        min_train_days=args.min_train_days,
        valid_days=args.valid_days,
        n_splits=args.folds,
        embargo_days=(
            5 if args.rebalance == "week_end" else max(1, args.period)
        ),
        holding_period=1 if args.rebalance == "week_end" else args.period,
        signal_average_days=args.signal_average_days,
        min_cross_section=args.min_cross_section,
        min_valid_ic_days=args.min_valid_ic_days,
        min_fold_consistency=args.min_fold_consistency,
        min_folds=args.min_folds,
        cost_bps=args.cost_bps,
        direction_mode=args.direction,
        paper_direction=1,
    )
    check_or_exit(dates, args.rebalance, fitness_config, parser)
    specifications = [parse_leaf(value) for value in args.leaf] or DEFAULT_LEAVES
    leaves = {}
    for name, path, column in specifications:
        loaded = load_daily_factor_leaves(
            require_path(path, f"factor leaf {name}"), dates, instruments,
            [column], device=device,
        )
        leaves[name] = loaded[column]
    close = load_daily_close_tensor(daily, dates, instruments, device=device)
    fwd_ret = tensor_rebalance_fwd_ret(
        close, dates, args.rebalance, args.period
    )
    pool = load_pit_daily_mask(pit, dates, instruments, device=device)
    pool &= torch.isfinite(close)
    fwd_ret = torch.where(pool, fwd_ret, torch.full_like(fwd_ret, float("nan")))

    continuous, industry = (), None
    if args.neutralize:
        styles, industry, _ = load_daily_exposures(
            require_path(args.exposures, "industry/value exposures"),
            dates, instruments, device=device,
        )
        continuous = tuple(styles.values())
    engine = DailyFactorGP(
        leaves, fwd_ret, pool_mask=pool,
        gp_config=DailyGPConfig(
            population_size=args.pop, generations=args.gens,
            max_depth=args.max_depth, seed=args.seed,
            checkpoint_path=str(run_dir / "checkpoint.json"),
            error_log_path=str(run_dir / "failures.jsonl"),
            resume=args.resume,
        ),
        fitness_config=fitness_config,
        continuous_exposures=continuous, industry_exposure=industry,
    )
    ranked = engine.run()
    archive = FactorArchive(args.max_correlation)
    destination = args.out or str(run_dir / "daily_gp.jsonl")
    with open(destination, "w", encoding="utf-8") as handle:
        for score, genome, rank, crowd in ranked:
            accepted, max_corr = (False, float("nan"))
            if score.valid:
                signal = trailing_signal_mean(
                    engine._factor(genome), args.signal_average_days
                )
                signal = torch.where(
                    torch.isfinite(fwd_ret), signal,
                    torch.full_like(signal, float("nan")),
                )
                accepted, max_corr = archive.add(str(genome), signal)
            record = {
                "family": "daily_tree", "pareto_rank": rank,
                "crowding": None if crowd == float("inf") else crowd,
                "archive_accepted": accepted, "max_archive_correlation": max_corr,
                "fitness": score.__dict__, "genome": genome.to_dict(),
                "expression": str(genome),
                "provenance": {
                    "train_start": args.start, "train_end": args.end,
                    "period": args.period, "direction_mode": args.direction,
                    "rebalance_rule": args.rebalance,
                    "signal_average_days": args.signal_average_days,
                    "neutralized": args.neutralize, "seed": args.seed,
                    "leaves": [name for name, _, _ in specifications],
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[daily-gp] {len(ranked)} candidates, {len(archive.entries)} distinct -> {destination}")


if __name__ == "__main__":
    main()
