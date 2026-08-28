"""CLI for handbook event-factor GP islands."""

import argparse
import json
from datetime import date, timedelta

import torch

from min_gp.config import (
    ADJUSTED_CLOSE_PARQUET,
    MINUTE_PARQUET,
    ZZ500_PIT_PARQUET,
    output_path,
    require_path,
)
from min_gp.data import load_pit_codes, load_pit_daily_mask, load_pit_dates
from min_gp.evaluation import DEFAULT_COST_BPS, WalkForwardConfig
from min_gp.gp import (
    EventFactorGP,
    EventGPConfig,
    ExhaustiveConfig,
    ExhaustiveEventSearch,
)
from min_gp.gp.event import EVENT_FAMILIES
from min_gp.label import tensor_rebalance_fwd_ret
from min_gp.evaluation.neutralize import BatchedNeutralizer, trailing_volatility
from min_gp.preflight import check_or_exit
from min_gp.spectral_data import build_minute_slice, load_daily_close_tensor
from min_gp.experiment import atomic_json, experiment_directory


def _shift_back(start: str, trading_days: int) -> str:
    """A calendar date far enough before `start` to cover N trading days."""
    return (
        date.fromisoformat(start) - timedelta(days=int(trading_days * 1.6) + 20)
    ).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=EVENT_FAMILIES, default="moderate_risk")
    parser.add_argument(
        "--skeleton-fields", choices=("volume", "close-volume"),
        default="volume",
        help="event_skeleton leaf space; volume avoids loading the close cube",
    )
    parser.add_argument("--search", choices=("exhaustive", "gp"), default="exhaustive",
                        help="event spaces are small (324/768), so enumerating "
                             "them beats sampling with a GP")
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--store-device", default="cpu",
                        help="where exhaustive intraday results are parked")
    parser.add_argument("--start", default="2018-01-02")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--period", type=int, default=1)
    parser.add_argument(
        "--rebalance", choices=("week_end", "daily"), default="week_end",
        help="portfolio schedule; week_end uses each calendar week's last trading day",
    )
    parser.add_argument("--signal-average-days", type=int, default=None)
    parser.add_argument("--pop", type=int, default=60)
    parser.add_argument("--gens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--direction", choices=("paper", "discovery"), default="discovery")
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS,
                        help="round-trip cost in bp charged on traded "
                             "notional; 30 = 0.15%% per side")
    parser.add_argument("--min-train-days", type=int, default=504)
    parser.add_argument("--valid-days", type=int, default=126)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--min-valid-ic-days", type=int, default=None)
    parser.add_argument("--min-fold-consistency", type=float, default=0.75,
                        help="reject a genome unless this share of folds agrees in sign")
    parser.add_argument("--min-folds", type=int, default=3,
                        help="reject a genome scored on fewer usable folds")
    parser.add_argument("--max-turnover", type=float, default=0.5,
                        help="reject a genome whose median per-rebalance turnover "
                             "exceeds this; 2.0 is a full long-short replacement")
    parser.add_argument("--neutralize", choices=("none", "vol"), default="vol",
                        help="strip trailing-volatility exposure inside the "
                             "fitness; the v1 front was dominated by it")
    parser.add_argument("--vol-window", type=int, default=60,
                        help="trailing trading days used for the volatility exposure")
    parser.add_argument("--minute-parquet", default=str(MINUTE_PARQUET))
    parser.add_argument("--daily-parquet", default=str(ADJUSTED_CLOSE_PARQUET),
                        help="adjusted daily close (ex-dividend safe labels)")
    parser.add_argument("--pit", default=str(ZZ500_PIT_PARQUET))
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-stocks", type=int, default=None,
                        help="diagnostic only: restrict the universe")
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
    if args.family == "event_skeleton" and args.search == "exhaustive":
        parser.error("event_skeleton has an open operator space; use --search gp")
    run_dir = experiment_directory(f"event_{args.family}", args.run_id)
    atomic_json(run_dir / "config.json", vars(args))

    minute_path = require_path(args.minute_parquet, "minute parquet")
    daily_path = require_path(args.daily_parquet, "adjusted daily close parquet")
    pit_path = require_path(args.pit, "CSI500 PIT parquet")
    if not args.cpu and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; pass --cpu for a CPU diagnostic run")
    device = "cpu" if args.cpu else "cuda"
    instruments = load_pit_codes(pit_path, args.start, args.end)
    dates = load_pit_dates(pit_path, args.start, args.end)
    if args.max_stocks:
        instruments = instruments[:args.max_stocks]
    fitness_config = WalkForwardConfig(
        min_train_days=args.min_train_days,
        valid_days=args.valid_days,
        n_splits=args.folds,
        embargo_days=5 if args.rebalance == "week_end" else max(1, args.period),
        holding_period=1 if args.rebalance == "week_end" else args.period,
        signal_average_days=args.signal_average_days,
        min_valid_ic_days=args.min_valid_ic_days,
        cost_bps=args.cost_bps,
        min_fold_consistency=args.min_fold_consistency,
        min_folds=args.min_folds,
        max_turnover=args.max_turnover,
        direction_mode=args.direction,
        paper_direction=-1,
    )
    check_or_exit(dates, args.rebalance, fitness_config, parser)
    if args.family == "moderate_risk":
        fields = ("close", "volume")
    elif args.family == "event_skeleton" and args.skeleton_fields == "close-volume":
        fields = ("close", "volume")
    else:
        fields = ("volume",)
    print(
        f"[event-gp] family={args.family}, search={args.search}, "
        f"stocks={len(instruments)}, "
        f"fields={','.join(fields)}, {args.start}..{args.end}, device={device}",
        flush=True,
    )
    minute, meta = build_minute_slice(
        minute_path, args.start, args.end, fields=fields,
        instruments=instruments, dates=dates, device=device,
    )
    close = load_daily_close_tensor(
        daily_path, meta["dates"], meta["instruments"], device=device
    )
    fwd_ret = tensor_rebalance_fwd_ret(
        close, meta["dates"], args.rebalance, args.period
    )
    pool_mask = load_pit_daily_mask(
        pit_path, meta["dates"], meta["instruments"], device=device
    )
    pool_mask &= torch.isfinite(close)
    fwd_ret = torch.where(
        pool_mask, fwd_ret, torch.full_like(fwd_ret, float("nan"))
    )
    neutralizer = None
    if args.neutralize == "vol":
        warmup_dates = load_pit_dates(
            pit_path, _shift_back(args.start, args.vol_window), args.end
        )
        close_full = load_daily_close_tensor(
            daily_path, warmup_dates, meta["instruments"], device=device
        )
        offset = warmup_dates.index(meta["dates"][0])
        vol = trailing_volatility(close_full, args.vol_window)[:, offset:]
        neutralizer = BatchedNeutralizer(
            fwd_ret.shape, continuous=(vol,), rank_space=True,
            min_cross_section=30,
        )
        share = float(torch.isfinite(vol[pool_mask]).float().mean())
        print(f"[event-gp] volatility exposure defined on {share:.1%} of pool "
              f"cells ({args.vol_window}d trailing, rank space)", flush=True)
        del close_full

    memory = sum(t.numel() * t.element_size() for t in minute.values()) / 1024**3
    print(
        f"[event-gp] grid={meta['I']}x{meta['D']}x{meta['NM']} "
        f"minute_tensors={memory:.2f}GiB", flush=True,
    )

    if args.search == "exhaustive":
        engine = ExhaustiveEventSearch(
            minute, fwd_ret, pool_mask=pool_mask,
            config=ExhaustiveConfig(
                family=args.family,
                chunk_rows=args.chunk_rows,
                store_device=args.store_device,
                checkpoint_path=str(run_dir / "exhaustive_stage1.pt"),
                error_log_path=str(run_dir / "failures.jsonl"),
                resume=args.resume,
            ),
            fitness_config=fitness_config,
        )
    else:
        engine = EventFactorGP(
            minute, fwd_ret, pool_mask=pool_mask,
            gp_config=EventGPConfig(
                family=args.family,
                population_size=args.pop,
                generations=args.gens,
                chunk_rows=args.chunk_rows,
                seed=args.seed,
                checkpoint_path=str(run_dir / "checkpoint.json"),
                error_log_path=str(run_dir / "failures.jsonl"),
                resume=args.resume,
            ),
            fitness_config=fitness_config,
            neutralizer=neutralizer,
        )
    ranked = engine.run()
    suffix = "exhaustive" if args.search == "exhaustive" else "gp"
    destination = args.out or str(run_dir / f"{args.family}_{suffix}.jsonl")
    with open(destination, "w", encoding="utf-8") as handle:
        for score, genome, rank, crowd in ranked:
            record = {
                "family": args.family,
                "pareto_rank": rank,
                "crowding": None if crowd == float("inf") else crowd,
                "fitness": {
                    "robust_ic": score.robust_ic,
                    "incremental_ic": score.incremental_ic,
                    "net_long_short": score.net_long_short,
                    "complexity": score.complexity,
                    "fold_consistency": score.fold_consistency,
                    "coverage": score.coverage,
                    "valid": score.valid,
                    "direction": score.direction,
                    # The value the turnover gate actually tested. Without it
                    # the gate's input cannot be audited after the fact.
                    "turnover": score.turnover,
                },
                "provenance": {
                    "train_start": args.start,
                    "train_end": args.end,
                    "period": args.period,
                    "rebalance_rule": args.rebalance,
                    "signal_average_days": args.signal_average_days,
                    "direction_mode": args.direction,
                    "search": args.search,
                    "min_fold_consistency": args.min_fold_consistency,
                    "seed": args.seed,
                    "minute_fields": list(fields),
                    "baseline_genome": engine.baseline_genome.to_dict(),
                },
                "genome": genome.to_dict(),
                "expression": str(genome),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[event-gp] {len(ranked)} candidates -> {destination}")


if __name__ == "__main__":
    main()
