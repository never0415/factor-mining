"""Production CLI for the ten fine-grained handbook GP anchors."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from min_gp.config import (
    ADJUSTED_CLOSE_PARQUET, INDEX_DAILY_PARQUET,
    INDUSTRY_VALUE_EXPOSURES_PARQUET, MINUTE_PARQUET,
    ZZ500_PIT_PARQUET, require_path,
)
from min_gp.data import load_pit_codes, load_pit_daily_mask, load_pit_dates
from min_gp.evaluation import DEFAULT_COST_BPS, WalkForwardConfig
from min_gp.experiment import (
    atomic_json, configuration_fingerprint, experiment_directory,
)
from min_gp.factor_export import export_ranked_factors
from min_gp.factors.handbook_skeleton import ANCHOR_PARAMS, handbook_anchor
from min_gp.gp import HandbookFactorGP, HandbookGPConfig
from min_gp.label import tensor_rebalance_fwd_ret
from min_gp.preflight import check_or_exit
from min_gp.spectral_data import (
    build_minute_slice, load_daily_close_tensor, load_daily_exposures,
)


RAW_MINUTE_FIELDS = {"open", "high", "low", "close", "volume"}


def parse_leaf(value):
    name, source = value.split("=", 1)
    path, key = source.rsplit(":", 1)
    return name, path, key


def load_npz_leaves(specifications, device):
    leaves = {}
    for name, path, key in specifications:
        with np.load(require_path(path, f"handbook leaf {name}"), allow_pickle=False) as archive:
            if key not in archive:
                raise KeyError(f"{key!r} is absent from {path}")
            leaves[name] = torch.as_tensor(archive[key], device=device)
    return leaves


def load_market_close(path, dates, device):
    schema = pq.ParquetFile(path).schema_arrow
    column = next((name for name in ("close_badj", "close", "收盘") if name in schema.names), None)
    if column is None:
        raise ValueError(f"market parquet has no close column: {schema.names}")
    frame = pd.read_parquet(path, columns=["trade_date", column])
    frame["trade_date"] = frame["trade_date"].astype(str).str.slice(0, 10)
    values = frame.drop_duplicates("trade_date").set_index("trade_date")[column]
    aligned = values.reindex(dates).to_numpy(np.float32)
    return torch.as_tensor(aligned, device=device)


def daily_return(close):
    result = torch.full_like(close, float("nan"))
    result[:, 1:] = close[:, 1:] / close[:, :-1].clamp(min=1e-12) - 1
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Search one handbook anchor and compatible typed parts.",
    )
    parser.add_argument("--factor", required=True, choices=tuple(ANCHOR_PARAMS))
    parser.add_argument(
        "--leaf-npz", action="append", default=[], metavar="NAME=PATH:KEY",
        help="exact report-specific leaf aligned as (I,D[,M]) to PIT order",
    )
    parser.add_argument("--start", default="2018-01-02")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--rebalance", choices=("week_end", "daily"), default="week_end")
    parser.add_argument("--period", type=int, default=1)
    parser.add_argument("--signal-average-days", type=int, default=None)
    parser.add_argument("--pop", type=int, default=60)
    parser.add_argument("--gens", type=int, default=8)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument(
        "--fitness-batch-size", type=int, default=8,
        help="candidate chunk size for batched weekly fitness (8 is 8-GiB safe)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--direction", choices=("paper", "discovery"), default="discovery")
    parser.add_argument("--paper-direction", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument("--min-train-days", type=int, default=504)
    parser.add_argument("--valid-days", type=int, default=126)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--min-valid-ic-days", type=int, default=None)
    parser.add_argument("--min-cross-section", type=int, default=30)
    parser.add_argument("--min-fold-consistency", type=float, default=.75)
    parser.add_argument("--min-folds", type=int, default=3)
    parser.add_argument("--minute-parquet", default=str(MINUTE_PARQUET))
    parser.add_argument("--daily-parquet", default=str(ADJUSTED_CLOSE_PARQUET))
    parser.add_argument("--market-parquet", default=str(INDEX_DAILY_PARQUET))
    parser.add_argument("--exposures", default=str(INDUSTRY_VALUE_EXPOSURES_PARQUET))
    parser.add_argument("--pit", default=str(ZZ500_PIT_PARQUET))
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-stocks", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--resume", action="store_true",
        help="resume the generation checkpoint in the same --run-id",
    )
    parser.add_argument("--export-rank", type=int, default=None,
                        help="also export valid candidates up to this Pareto rank")
    args = parser.parse_args(argv)
    if args.resume and not args.run_id:
        parser.error("--resume requires the original --run-id")
    args.signal_average_days = args.signal_average_days or (5 if args.rebalance == "week_end" else 1)
    args.min_valid_ic_days = args.min_valid_ic_days or (12 if args.rebalance == "week_end" else 40)
    device = "cpu" if args.cpu else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; pass --cpu")

    pit = require_path(args.pit, "PIT universe")
    daily_path = require_path(args.daily_parquet, "adjusted daily close")
    instruments = load_pit_codes(pit, args.start, args.end)
    dates = load_pit_dates(pit, args.start, args.end)
    if args.max_stocks:
        instruments = instruments[:args.max_stocks]
    fitness = WalkForwardConfig(
        min_train_days=args.min_train_days, valid_days=args.valid_days,
        n_splits=args.folds,
        embargo_days=5 if args.rebalance == "week_end" else max(1, args.period),
        holding_period=1 if args.rebalance == "week_end" else args.period,
        signal_average_days=args.signal_average_days,
        min_cross_section=args.min_cross_section,
        min_valid_ic_days=args.min_valid_ic_days,
        min_fold_consistency=args.min_fold_consistency, min_folds=args.min_folds,
        cost_bps=args.cost_bps, direction_mode=args.direction,
        paper_direction=args.paper_direction,
    )
    check_or_exit(dates, args.rebalance, fitness, parser)
    run_dir = experiment_directory(f"handbook_{args.factor}", args.run_id)
    config_path = run_dir / "config.json"
    if args.resume:
        if not config_path.exists():
            parser.error(f"cannot resume: missing original config {config_path}")
    else:
        atomic_json(config_path, vars(args))

    anchor = handbook_anchor(args.factor)
    required = set(anchor.required_fields)
    context = load_npz_leaves(
        [parse_leaf(value) for value in args.leaf_npz], device
    )
    minute_fields = sorted(required & RAW_MINUTE_FIELDS)
    if minute_fields:
        minute, meta = build_minute_slice(
            require_path(args.minute_parquet, "minute parquet"),
            args.start, args.end, fields=minute_fields,
            instruments=instruments, dates=dates, device=device,
        )
        context.update(minute)
        instruments, dates = meta["instruments"], meta["dates"]
    close = load_daily_close_tensor(daily_path, dates, instruments, device=device)
    if "daily_close" in required:
        context["daily_close"] = close
    if "market_close" in required:
        context["market_close"] = load_market_close(
            require_path(args.market_parquet, "market daily parquet"), dates, device
        )
    if "daily_return" in required:
        context["daily_return"] = daily_return(close)
    if "float_market_cap" in required:
        styles, _industry, _levels = load_daily_exposures(
            require_path(args.exposures, "market-cap exposures"),
            dates, instruments, device=device,
        )
        context["float_market_cap"] = styles["ln_float_market_cap"].exp()
    if "volume_share" in required and "volume" in context:
        denominator = torch.nansum(context["volume"].float(), dim=0, keepdim=True)
        context["volume_share"] = context["volume"].float() / denominator.clamp(min=1e-12)

    missing = required - set(context)
    if missing:
        parser.error(
            f"{args.factor} needs exact report-specific leaves {sorted(missing)}; "
            "provide each as --leaf-npz NAME=PATH:KEY. No proxy is substituted."
        )
    for name, value in context.items():
        expected_prefix = (len(instruments), len(dates))
        if value.ndim >= 2 and tuple(value.shape[:2]) != expected_prefix:
            parser.error(
                f"leaf {name} starts with {tuple(value.shape[:2])}, expected {expected_prefix}"
            )
    fwd_ret = tensor_rebalance_fwd_ret(close, dates, args.rebalance, args.period)
    pool = load_pit_daily_mask(pit, dates, instruments, device=device)
    pool &= torch.isfinite(close)
    fwd_ret = torch.where(pool, fwd_ret, torch.full_like(fwd_ret, float("nan")))
    data_fingerprint = configuration_fingerprint({
        "start": args.start, "end": args.end,
        "rebalance": args.rebalance, "period": args.period,
        "instruments": [str(value) for value in instruments],
        "dates": [str(value) for value in dates],
        "required_fields": sorted(required),
        "minute_parquet": str(args.minute_parquet),
        "daily_parquet": str(args.daily_parquet),
        "market_parquet": str(args.market_parquet),
        "exposures": str(args.exposures), "pit": str(args.pit),
        "leaf_npz": list(args.leaf_npz),
    })
    engine = HandbookFactorGP(
        context, fwd_ret, pool_mask=pool,
        gp_config=HandbookGPConfig(
            population_size=args.pop, generations=args.gens,
            chunk_rows=args.chunk_rows, seed=args.seed,
            fitness_batch_size=args.fitness_batch_size,
            baseline_name=args.factor,
            error_log_path=str(run_dir / "failures.jsonl"),
            checkpoint_path=str(run_dir / "checkpoint.json"),
            resume=args.resume, data_fingerprint=data_fingerprint,
        ),
        fitness_config=fitness,
    )
    ranked = engine.run()
    destination = args.out or str(run_dir / "handbook_gp.jsonl")
    with open(destination, "w", encoding="utf-8") as handle:
        for score, genome, rank, crowd in ranked:
            handle.write(json.dumps({
                "family": "handbook", "anchor": args.factor,
                "pareto_rank": rank,
                "crowding": None if crowd == float("inf") else crowd,
                "fitness": score.__dict__, "genome": genome.to_dict(),
                "expression": str(genome),
                "provenance": {
                    "train_start": args.start, "train_end": args.end,
                    "rebalance_rule": args.rebalance,
                    "signal_average_days": args.signal_average_days,
                    "period": args.period, "direction_mode": args.direction,
                    "seed": args.seed, "baseline": args.factor,
                    "context_fields": sorted(context),
                },
            }, ensure_ascii=False) + "\n")
    if args.export_rank is not None:
        outputs = export_ranked_factors(
            ranked, engine._factor, instruments, dates,
            run_dir / "daily_factors", args.export_rank,
        )
        print(f"[handbook-gp] exported {len(outputs)} daily parquet leaves", flush=True)
    print(f"[handbook-gp] {len(ranked)} candidates -> {destination}")


if __name__ == "__main__":
    main()


def load_npz_leaves_aligned(specifications, instruments, dates, device):
    """Load leaves and reindex them onto a requested (instrument, date) grid.

    A leaf archive is written on the PIT grid. ``build_slice`` instead derives
    its instrument list from the minute file itself, so it drops PIT members
    that never traded in the range and the two panels differ by a handful of
    rows. Reindexing by name rather than trusting positions keeps a silently
    misaligned panel -- one stock's minutes scored against another's returns --
    out of the search. Cells the archive does not hold come back missing.
    """
    aligned = {}
    for name, path, key in specifications:
        resolved = Path(require_path(path, f"handbook leaf {name}"))
        sidecar = resolved.parent / "metadata.json"
        with np.load(resolved, allow_pickle=False) as archive:
            if key not in archive:
                raise KeyError(f"{key!r} is absent from {path}")
            values = archive[key]
        if not sidecar.exists():
            raise FileNotFoundError(
                f"{resolved.parent} has no metadata.json; cannot align leaf {name}"
            )
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        source_rows = {str(v): i for i, v in enumerate(payload["instruments"])}
        source_cols = {str(v): i for i, v in enumerate(payload["dates"])}
        rows = np.array([source_rows.get(str(v), -1) for v in instruments])
        cols = np.array([source_cols.get(str(v), -1) for v in dates])
        selected = values[np.where(rows >= 0, rows, 0)][
            :, np.where(cols >= 0, cols, 0)
        ]
        missing = (rows < 0)[:, None] | (cols < 0)[None, :]
        if missing.any():
            expanded = missing if selected.ndim == 2 else missing[:, :, None]
            if selected.dtype == np.bool_:
                selected = selected & ~expanded
            else:
                selected = np.where(expanded, np.nan, selected)
        aligned[name] = torch.as_tensor(selected, device=device)
        print(
            f"[leaf-align] {name} {values.shape} -> {tuple(selected.shape)}; "
            f"{int((rows < 0).sum())} instruments and {int((cols < 0).sum())} "
            "dates absent from the archive",
            flush=True,
        )
    return aligned
