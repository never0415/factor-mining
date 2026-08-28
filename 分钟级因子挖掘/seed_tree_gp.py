"""CLI for the unified typed-tree GP with external-factor organ reuse."""

import argparse
import json
from pathlib import Path

import torch

from min_gp.config import (
    ADJUSTED_CLOSE_PARQUET, INDEX_DAILY_PARQUET,
    INDUSTRY_VALUE_EXPOSURES_PARQUET, MINUTE_PARQUET,
    ZZ500_PIT_PARQUET, require_path,
)
from min_gp.data import build_slice, load_pit_codes, load_pit_daily_mask, load_pit_dates
from min_gp.evaluation import DEFAULT_COST_BPS, WalkForwardConfig
from min_gp.experiment import atomic_json, experiment_directory
from min_gp.factor_export import export_ranked_factors, load_factor_parquet
from min_gp.factor_leaf_factory import (
    LeafFactoryConfig, build_external_factor_leaves,
)
from min_gp.factors.catalog import load_exported_genome
from min_gp.factors.rushing_skeleton import (
    RUSHING_LEAF_FIELDS, rushing_imbalance_node,
)
from min_gp.dsl import (
    LeafNode, OperatorNode, SemanticType, evaluate_daily_expression,
)
from min_gp.factors.seed_tree import SeedTreeGenome
from min_gp.gp.seed_tree import SeedTreeFactorGP, SeedTreeGPConfig
from min_gp.gp.organs import (
    EXTERNAL_FACTOR_NAMES, build_external_organ_library,
)
from min_gp.handbook_gp import load_npz_leaves_aligned, parse_leaf
from min_gp.label import tensor_rebalance_fwd_ret
from min_gp.operators import build_operator_registry
from min_gp.preflight import check_or_exit
from min_gp.seeds import SEEDS
from min_gp.spectral_data import load_daily_close_tensor


# Handbook cores that are worth handing to the open tree search. Each entry
# names the minute leaves it consumes and the builder for its bare subtree.
EXTRA_ANCHORS = {
    "rushing_forward": (RUSHING_LEAF_FIELDS, rushing_imbalance_node),
}


# Seed operators that turn a daily leaf into a starting genome. They exist only
# so the injected anchor enters the population as a growable tree; the search
# is free to replace them immediately.
ANCHOR_WRAPPERS = (
    ("rank", "seed_cs_rank", {}),
    ("smooth", "seed_ts_mean_daily", {"window": 5}),
)


def install_anchor_leaves(names, context, registry, chunk_rows=4096):
    """Collapse each requested handbook core to a daily leaf in the context.

    The core is evaluated once and installed under its own name, so the search
    sees it exactly the way it sees ``close`` -- a terminal that any daily seed
    operator may consume and that ``random_tree`` may pick when it needs one.

    Carrying the core as a subtree instead was the wrong shape twice over. Its
    root is ``DAILY_RAW_FACTOR``, a type no seed operator produces, so mutation
    could never regenerate it: once the last genome holding it left the
    population the core was gone for good, and it could only ever spread by
    crossover from a genome that already had it. It also re-collapsed the whole
    minute cube on every candidate that contained it, which is the single most
    expensive thing in the tree.
    """
    installed = []
    for name in names:
        fields, builder = EXTRA_ANCHORS[name]
        missing = sorted(set(fields) - set(context))
        if missing:
            raise SystemExit(
                f"--extra-anchor {name} needs leaves {missing}; "
                "supply each as --leaf-npz NAME=PATH:KEY"
            )
        for invert in (False, True):
            core, _ = builder(registry, invert_mask=invert)
            leaf_name = f"{name}_inverted" if invert else name
            context[leaf_name] = evaluate_daily_expression(
                core, context, registry, chunk_rows
            ).float()
            installed.append(leaf_name)
            print(
                f"[seed-tree-gp] anchor leaf {leaf_name} "
                f"{tuple(context[leaf_name].shape)} installed from {core}",
                flush=True,
            )
    return installed


def install_daily_leaves(specifications, context, instruments, dates, device):
    """Install exported daily factors as leaves the search can build on."""
    installed = []
    for value in specifications:
        name, _, path = value.partition("=")
        if not path:
            raise SystemExit(f"--daily-leaf expects NAME=PATH, got {value!r}")
        context[name] = load_factor_parquet(path, instruments, dates, device=device)
        installed.append(name)
        print(
            f"[seed-tree-gp] daily leaf {name} {tuple(context[name].shape)} "
            f"loaded from {Path(path).name}",
            flush=True,
        )
    return installed


def build_leaf_anchors(leaf_names, registry):
    """Seed genomes rooted at a daily leaf, one per wrapper."""
    genomes = []
    for leaf_name in leaf_names:
        leaf = LeafNode(leaf_name, SemanticType.DAILY_RAW_FACTOR)
        for label, operator, params in ANCHOR_WRAPPERS:
            root = OperatorNode(operator, (leaf,), params).bind(registry)
            genomes.append(SeedTreeGenome(f"{leaf_name}_{label}", root))
    return genomes


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Unified typed-tree GP; external PDF factors donate mutable organs.",
    )
    parser.add_argument("--anchor", action="append", choices=tuple(SEEDS),
                        help="seed anchor to admit; omitted means all 59")
    parser.add_argument(
        "--extra-anchor", action="append", default=[], choices=tuple(EXTRA_ANCHORS),
        help="handbook core to collapse into a daily leaf, both polarities; "
             "the search then builds on it as it would on close or volume",
    )
    parser.add_argument(
        "--daily-leaf", action="append", default=[], metavar="NAME=PATH",
        help="exported daily factor to install as a leaf the search can grow on",
    )
    parser.add_argument(
        "--drop-leaf", action="append", default=[], metavar="NAME",
        help="remove a leaf from the context after the anchors are collapsed; "
             "the minute cubes an anchor was built from are dead weight once "
             "it is a daily leaf, and they dominate resident GPU memory",
    )
    parser.add_argument(
        "--leaf-npz", action="append", default=[], metavar="NAME=PATH:KEY",
        help="report-specific leaf aligned as (I,D[,M]) to PIT order",
    )
    parser.add_argument(
        "--seed-candidate", action="append", default=[], metavar="SIDECAR",
        help="export sidecar (candidate_*.parquet.json) whose genome seeds "
             "the search; any family, and it leads the initial population",
    )
    parser.add_argument(
        "--pool-parquet", action="append", default=[], metavar="PATH",
        help="exported factor already accepted for trading; widens the "
             "incremental-IC residual and gates rediscoveries",
    )
    parser.add_argument(
        "--max-pool-correlation", type=float, default=None,
        help="reject a candidate whose |rank correlation| with any pooled "
             "factor exceeds this (e.g. 0.6)",
    )
    parser.add_argument("--start", default="2018-01-02")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--rebalance", choices=("week_end", "daily"), default="week_end")
    parser.add_argument("--period", type=int, default=1)
    parser.add_argument("--signal-average-days", type=int, default=None)
    parser.add_argument("--pop", type=int, default=60)
    parser.add_argument("--gens", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument(
        "--no-external-organs", action="store_false", dest="external_organs",
        help="disable organs extracted from factor_reproduction_handbook.pdf",
    )
    parser.add_argument(
        "--organ-source", action="append", choices=EXTERNAL_FACTOR_NAMES,
        help="external failed factor allowed to donate organs; repeat as needed; "
             "omitted means all PDF factors",
    )
    parser.add_argument("--organ-min-levels", type=int, default=2)
    parser.add_argument("--organ-max-levels", type=int, default=3)
    parser.add_argument("--organ-graft-probability", type=float, default=.30)
    parser.add_argument("--random-initialization-fraction", type=float, default=.30)
    parser.add_argument(
        "--elite", type=int, default=None,
        help="genomes carried to the next generation unchanged "
             "(default 6, tuned for the default population of 60). Scale it "
             "with --pop or elitism becomes negligible.",
    )
    parser.add_argument(
        "--tournament", type=int, default=None,
        help="candidates sampled per parent draw (default 4). Selection "
             "pressure comes from this relative to --pop, so a large "
             "population with the default reads as near-random sampling.",
    )
    parser.add_argument(
        "--leaves-from-context", action="store_true",
        help="let random trees use every minute panel and session mask that "
             "is loaded, not only the ones a seed expression mentions. The "
             "leaf pool is otherwise a property of the seed library: a "
             "measured run held 37 tensors and reached 15, leaving open, "
             "is_jump, has_gap, 18 multi-scale session masks and w_time "
             "computed but unselectable.",
    )
    parser.add_argument(
        "--intra-run-max-correlation", type=float, default=None,
        metavar="RHO",
        help="drop an elite whose factor correlates at least RHO with a "
             "better-ranked elite. NSGA-II keeps diversity in objective "
             "space only, so without this one signal can occupy several "
             "slots on the front as near-identical expressions. Unlike "
             "--max-pool-correlation this acts on the live population and "
             "needs no --pool-parquet. Try 0.9.",
    )
    parser.add_argument(
        "--seed-only-operators", action="store_false",
        dest="unified_operator_space",
        help="compatibility mode: mutation may generate only seed_* operators",
    )
    parser.add_argument(
        "--keep-whole-factor-anchors", action="store_false",
        dest="organs_replace_catalog_anchors",
        help="also seed complete catalog factors; normally external factors "
             "donate internal organs only",
    )
    parser.add_argument(
        "--max-peak-bytes", type=float, default=None,
        help="reject a candidate whose calibrated peak, resident inputs "
             "included, would exceed this many bytes; the guard that keeps "
             "a heavy same-minute tree from thrashing a small card",
    )
    parser.add_argument("--max-cost-units", type=int, default=None)
    parser.add_argument("--max-estimated-seconds", type=float, default=None)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--direction", choices=("paper", "discovery"), default="discovery")
    parser.add_argument("--paper-direction", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument("--min-train-days", type=int, default=504)
    parser.add_argument("--valid-days", type=int, default=126)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--min-valid-ic-days", type=int, default=None)
    parser.add_argument("--min-cross-section", type=int, default=30)
    parser.add_argument(
        "--outlier-mad", type=float, default=5.0,
        help="PIT factor winsorization before IC: median +/- N * raw MAD",
    )
    parser.add_argument("--min-fold-consistency", type=float, default=.75)
    parser.add_argument("--min-folds", type=int, default=3)
    parser.add_argument("--minute-parquet", default=str(MINUTE_PARQUET))
    parser.add_argument("--daily-parquet", default=str(ADJUSTED_CLOSE_PARQUET))
    parser.add_argument("--market-parquet", default=str(INDEX_DAILY_PARQUET))
    parser.add_argument(
        "--exposures", default=str(INDUSTRY_VALUE_EXPOSURES_PARQUET),
        help="PIT daily exposure parquet containing ln_float_market_cap",
    )
    parser.add_argument(
        "--no-leaf-api-fallback", action="store_false", dest="leaf_api_fallback",
        help="do not fetch CSI 500/float market cap with AkShare when "
             "the corresponding local parquet is missing",
    )
    parser.add_argument(
        "--skip-pair-similarity", action="store_false",
        dest="build_pair_similarity",
        help="memory escape hatch: omit the I×I×D cooperation leaf and its organs",
    )
    parser.add_argument("--pit", default=str(ZZ500_PIT_PARQUET))
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-stocks", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--export-rank", type=int, default=None,
                        help="also export valid candidates up to this Pareto rank")
    args = parser.parse_args(argv)
    args.signal_average_days = args.signal_average_days or (5 if args.rebalance == "week_end" else 1)
    args.min_valid_ic_days = args.min_valid_ic_days or (12 if args.rebalance == "week_end" else 40)

    device = "cpu" if args.cpu else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; pass --cpu")
    pit = require_path(args.pit, "PIT universe")
    minute_path = require_path(args.minute_parquet, "minute parquet")
    daily_path = require_path(args.daily_parquet, "adjusted daily close")
    instruments = load_pit_codes(pit, args.start, args.end)
    dates = load_pit_dates(pit, args.start, args.end)
    if args.max_stocks:
        instruments = instruments[:args.max_stocks]
    if args.max_pool_correlation is not None and not args.pool_parquet:
        parser.error("--max-pool-correlation has no effect without --pool-parquet")
    if args.external_organs and (args.extra_anchor or args.daily_leaf):
        parser.error(
            "unified organ mode forbids opaque --extra-anchor/--daily-leaf "
            "terminals; omit them, or pass --no-external-organs for a legacy run"
        )
    if not 0 <= args.organ_graft_probability <= 1:
        parser.error("--organ-graft-probability must be in [0, 1]")
    if not 0 <= args.random_initialization_fraction < 1:
        parser.error("--random-initialization-fraction must be in [0, 1)")
    if args.intra_run_max_correlation is not None and not (
        0 < args.intra_run_max_correlation <= 1
    ):
        parser.error("--intra-run-max-correlation must lie in (0, 1]")
    if args.elite is not None and not 0 <= args.elite <= args.pop:
        parser.error("--elite must lie in [0, --pop]")
    if args.tournament is not None and not 2 <= args.tournament <= args.pop:
        parser.error("--tournament must lie in [2, --pop]")
    fitness = WalkForwardConfig(
        min_train_days=args.min_train_days, valid_days=args.valid_days,
        n_splits=args.folds,
        embargo_days=5 if args.rebalance == "week_end" else max(1, args.period),
        holding_period=1 if args.rebalance == "week_end" else args.period,
        signal_average_days=args.signal_average_days,
        min_cross_section=args.min_cross_section,
        outlier_mad=args.outlier_mad,
        min_valid_ic_days=args.min_valid_ic_days,
        min_fold_consistency=args.min_fold_consistency, min_folds=args.min_folds,
        cost_bps=args.cost_bps, direction_mode=args.direction,
        paper_direction=args.paper_direction,
        max_pool_correlation=args.max_pool_correlation,
    )
    check_or_exit(dates, args.rebalance, fitness, parser)
    run_dir = experiment_directory("seed_tree", args.run_id)
    atomic_json(run_dir / "config.json", vars(args))

    # This legacy-compatible loader intentionally preserves the original
    # 241-bar and regime-leaf definitions used by the numerical migration tests.
    tensors, masks, _old_label, meta = build_slice(
        minute_path, args.start, args.end, instruments=instruments,
        device=device,
    )
    context = {**tensors, **{name: value.to(device) for name, value in masks.items()}}
    grid = (len(meta["instruments"]), len(meta["dates"]))
    context.update(load_npz_leaves_aligned(
        [parse_leaf(value) for value in args.leaf_npz],
        meta["instruments"], meta["dates"], device,
    ))
    for name, value in context.items():
        if value.ndim >= 2 and tuple(value.shape[:2]) != grid:
            parser.error(
                f"leaf {name} starts with {tuple(value.shape[:2])}, expected {grid}"
            )
    close = load_daily_close_tensor(daily_path, meta["dates"], meta["instruments"], device=device)
    fwd_ret = tensor_rebalance_fwd_ret(close, meta["dates"], args.rebalance, args.period)
    pool = load_pit_daily_mask(pit, meta["dates"], meta["instruments"], device=device)
    pool &= torch.isfinite(close)
    fwd_ret = torch.where(pool, fwd_ret, torch.full_like(fwd_ret, float("nan")))
    leaf_report = {"requested": [], "built": {}, "missing": []}
    if args.external_organs:
        leaf_report = build_external_factor_leaves(
            context, close, pool, meta["dates"], meta["instruments"],
            args.organ_source or EXTERNAL_FACTOR_NAMES,
            LeafFactoryConfig(
                market_parquet=args.market_parquet,
                exposures_parquet=args.exposures,
                api_fallback=args.leaf_api_fallback,
                cache_directory=str(run_dir / "leaf_cache"),
                build_pair_similarity=args.build_pair_similarity,
            ),
        )
        atomic_json(run_dir / "leaf_sources.json", leaf_report)
        print(
            f"[seed-tree-gp] derived external leaves "
            f"built={len(leaf_report['built'])} missing={leaf_report['missing']}",
            flush=True,
        )
    registry = build_operator_registry()
    # Daily leaves first: they must be in the context before any genome that
    # names them is built, and before the engine collects its leaf pool.
    anchor_leaves = install_anchor_leaves(
        args.extra_anchor, context, registry, args.chunk_rows
    )
    anchor_leaves += install_daily_leaves(
        args.daily_leaf, context, meta["instruments"], meta["dates"], device
    )
    for name in args.drop_leaf:
        if name in anchor_leaves:
            parser.error(f"--drop-leaf {name} would remove an installed anchor leaf")
        if context.pop(name, None) is None:
            parser.error(f"--drop-leaf {name} is not in the context")
        print(f"[seed-tree-gp] dropped leaf {name}", flush=True)
    if args.drop_leaf:
        if device == "cuda":
            torch.cuda.empty_cache()
        print(
            "[seed-tree-gp] context now holds "
            f"{sum(v.numel() * v.element_size() for v in context.values()) / 1e9:.2f} GB",
            flush=True,
        )
    organ_library = None
    if args.external_organs:
        organ_library = build_external_organ_library(
            registry,
            min_levels=args.organ_min_levels,
            max_levels=args.organ_max_levels,
            source_names=args.organ_source,
            available_fields=context,
        )
        atomic_json(run_dir / "organ_manifest.json", organ_library.manifest())
        print(
            f"[seed-tree-gp] {len(organ_library.blocks)} transparent organs "
            f"from {len(organ_library.source_names)} external factors; "
            f"graft_probability={args.organ_graft_probability:.2f}",
            flush=True,
        )
    extra_genomes = []
    for sidecar in args.seed_candidate:
        genome, expression, stored = load_exported_genome(sidecar, registry)
        extra_genomes.append(genome)
        print(
            f"[seed-tree-gp] seeded from {Path(sidecar).name} "
            f"(robust_ic={stored.get('robust_ic', float('nan')):.4f}): {expression}",
            flush=True,
        )
    extra_genomes.extend(build_leaf_anchors(anchor_leaves, registry))
    factor_pool = [
        torch.where(
            pool,
            load_factor_parquet(
                path, meta["instruments"], meta["dates"], device=device
            ),
            torch.full(grid, float("nan"), device=device),
        )
        for path in args.pool_parquet
    ]
    if factor_pool:
        print(
            f"[seed-tree-gp] {len(factor_pool)} pooled factors; "
            f"novelty cut={args.max_pool_correlation}",
            flush=True,
        )
    engine = SeedTreeFactorGP(
        context, fwd_ret, pool_mask=pool,
        gp_config=SeedTreeGPConfig(
            population_size=args.pop, generations=args.gens,
            max_depth=args.max_depth, chunk_rows=args.chunk_rows,
            seed=args.seed, anchor_names=None if args.anchor is None else tuple(args.anchor),
            unified_operator_space=args.unified_operator_space,
            random_initialization_fraction=args.random_initialization_fraction,
            intra_run_max_correlation=args.intra_run_max_correlation,
            leaves_from_context=args.leaves_from_context,
            organ_graft_probability=args.organ_graft_probability,
            **{
                name: value for name, value in (
                    ("elite", args.elite), ("tournament", args.tournament),
                ) if value is not None
            },
            organs_replace_catalog_anchors=args.organs_replace_catalog_anchors,
            **{
                name: value for name, value in (
                    ("max_peak_bytes", None if args.max_peak_bytes is None
                     else int(args.max_peak_bytes)),
                    ("max_cost_units", args.max_cost_units),
                    ("max_estimated_seconds", args.max_estimated_seconds),
                ) if value is not None
            },
        ),
        fitness_config=fitness,
        extra_genomes=extra_genomes,
        factor_pool=factor_pool,
        organ_library=organ_library,
    )
    ranked = engine.run()
    destination = args.out or str(run_dir / "seed_tree_gp.jsonl")
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        for score, genome, rank, crowd in ranked:
            handle.write(json.dumps({
                "family": "seed_tree", "pareto_rank": rank,
                "crowding": None if crowd == float("inf") else crowd,
                "fitness": score.__dict__, "genome": genome.to_dict(),
                "expression": str(genome),
                "provenance": {
                    "train_start": args.start, "train_end": args.end,
                    "rebalance_rule": args.rebalance,
                    "signal_average_days": args.signal_average_days,
                    "period": args.period, "direction_mode": args.direction,
                    "seed": args.seed, "anchors": args.anchor or ["all"],
                    "external_organ_sources": (
                        [] if organ_library is None
                        else list(organ_library.source_names)
                    ),
                    "organ_graft_probability": args.organ_graft_probability,
                    "unified_operator_space": args.unified_operator_space,
                },
            }, ensure_ascii=False) + "\n")
    if args.export_rank is not None:
        outputs = export_ranked_factors(
            ranked, engine._factor, meta["instruments"], meta["dates"],
            run_dir / "daily_factors", args.export_rank,
        )
        print(f"[seed-tree-gp] exported {len(outputs)} daily parquet leaves", flush=True)
    print(f"[seed-tree-gp] {len(ranked)} candidates -> {destination}")


if __name__ == "__main__":
    main()
