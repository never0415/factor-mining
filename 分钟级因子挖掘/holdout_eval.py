"""Hold-out evaluation for genomes produced by the typed GP islands.

The walk-forward folds inside `min_gp.evaluation` are the *selection* set: the
GP maximises objectives computed on them, so their numbers stop being unbiased
the moment a genome is chosen. This script scores the survivors on a period the
search never touched, reusing the training-time direction verbatim.

  python -m min_gp.holdout_eval \
      --candidates min_gp/output/dripping_stone_gp.jsonl \
      --start 2025-01-02 --end 2026-07-31
"""

import argparse
import json
from collections import OrderedDict

import numpy as np
import torch

from min_gp.config import (
    ADJUSTED_CLOSE_PARQUET,
    MINUTE_PARQUET,
    ZZ500_PIT_PARQUET,
    output_path,
    require_path,
)
from min_gp.data import load_pit_codes, load_pit_daily_mask, load_pit_dates
from min_gp.evaluation.incremental import (
    DEFAULT_COST_BPS,
    cross_section_residual,
    net_long_short_return,
    trailing_signal_mean,
)
from min_gp.factors import (
    DrippingStoneTemplate, DrippingSkeletonGenome,
    EventSkeletonGenome,
    ModerateRiskTemplate,
    WaitRescueTemplate,
)
from min_gp.factors.event_skeleton import moderate_risk_anchor, wait_rescue_anchor
from min_gp.fitness import daily_spearman_ic, factor_health
from min_gp.label import tensor_rebalance_fwd_ret
from min_gp.spectral_data import build_minute_slice, load_daily_close_tensor


# family -> (template class, required minute fields, paper direction)
TEMPLATES = {
    "dripping_stone": (DrippingStoneTemplate, ("volume",), +1),
    "moderate_risk": (ModerateRiskTemplate, ("close", "volume"), -1),
    "wait_rescue": (WaitRescueTemplate, ("volume",), -1),
    # Required fields are derived from each skeleton genome below.
    "event_skeleton": (wait_rescue_anchor, ("volume",), -1),
}


def detect_family(record):
    """Infer the factor family from an explicit tag or the genome's fields."""
    family = record.get("family")
    if family in TEMPLATES:
        return family
    genome = record.get("genome", {})
    if "detector" in genome and "primary" in genome:
        return "event_skeleton"
    if genome.get("kind") == "dripping_skeleton":
        return "dripping_stone"
    if "volume_transform" in genome:
        return "dripping_stone"
    if "response_window" in genome:
        return "moderate_risk"
    if "top_k" in genome:
        return "wait_rescue"
    raise ValueError(f"cannot infer factor family from genome keys {sorted(genome)}")


def build_genome(family, payload):
    """Rebuild a template, dropping fields the current dataclass no longer has."""
    cls = TEMPLATES[family][0]
    if family == "dripping_stone" and payload.get("kind") == "dripping_skeleton":
        return DrippingSkeletonGenome.from_dict(payload), []
    if family == "event_skeleton":
        return EventSkeletonGenome.from_dict(payload), []
    fields = set(cls.__dataclass_fields__)
    dropped = sorted(set(payload) - fields)
    kept = {key: value for key, value in payload.items() if key in fields}
    return cls(**kept), dropped


def event_baseline_genome(record, available_fields=None):
    """Restore the exact training anchor, with a legacy-file fallback."""
    payload = record.get("provenance", {}).get("baseline_genome")
    if payload is not None:
        return EventSkeletonGenome.from_dict(payload)
    fields = set(available_fields or ())
    return moderate_risk_anchor() if "close" in fields else wait_rescue_anchor()


def required_minute_fields(records):
    """Union of fields used by candidates and their recorded baselines."""
    required = set()
    for record in records:
        family = detect_family(record)
        if family != "event_skeleton":
            required.update(TEMPLATES[family][1])
            continue
        genome, _ = build_genome(family, record["genome"])
        required.update(genome.required_fields)
        provenance_fields = record.get("provenance", {}).get("minute_fields", ())
        baseline = event_baseline_genome(record, provenance_fields)
        required.update(baseline.required_fields)
    return tuple(sorted(required))


def load_candidates(path, max_rank, limit):
    """Read a GP jsonl, keep valid genomes up to `max_rank`, dedupe by expression."""
    seen = OrderedDict()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not record.get("fitness", {}).get("valid", False):
                continue
            if record.get("pareto_rank", 0) > max_rank:
                continue
            key = record.get("expression") or json.dumps(
                record["genome"], sort_keys=True
            )
            if key not in seen:
                seen[key] = record
    records = list(seen.values())
    return records[:limit] if limit else records


def newey_west_icir(series, period):
    """ICIR, Bartlett-corrected when overlapping labels autocorrelate the series."""
    if series.size < 2:
        return float("nan")
    std = series.std()
    if std <= 0:
        return float("nan")
    if period <= 1:
        return float(series.mean() / std)
    lag = period - 1
    weight = 1.0
    for k in range(1, min(lag + 1, series.size // 2)):
        rho = np.corrcoef(series[:-k], series[k:])[0, 1]
        weight += 2 * (1 - k / (lag + 1)) * (rho if np.isfinite(rho) else 0.0)
    variance = std**2 * max(weight, 0.1)
    if variance <= 0:
        return float("nan")
    return float(series.mean() / np.sqrt(variance))


def evaluate_factor(factor, anchor, fwd_ret, direction, args, missing_rate):
    """Full-period hold-out metrics plus a block-wise sign-stability check."""
    factor = trailing_signal_mean(factor, args.signal_average_days)
    anchor = trailing_signal_mean(anchor, args.signal_average_days)
    ic = daily_spearman_ic(factor, fwd_ret, min_n=args.min_cross_section)
    finite = ic[torch.isfinite(ic)]
    if finite.numel() < args.min_ic_days:
        return None
    series = finite.cpu().numpy()
    aligned = float(series.mean()) * direction
    overlap_period = 1 if args.rebalance == "week_end" else args.period
    icir = newey_west_icir(series, overlap_period) * direction

    residual = cross_section_residual(factor, anchor, use_ranks=True)
    inc_ic = daily_spearman_ic(residual, fwd_ret, min_n=args.min_cross_section)
    inc_finite = inc_ic[torch.isfinite(inc_ic)]
    incremental = (
        float(inc_finite.mean().item()) * direction if inc_finite.numel() else 0.0
    )

    signs = []
    min_block_days = max(5, args.min_ic_days // args.blocks)
    for block in np.array_split(np.arange(factor.shape[1]), args.blocks):
        piece = ic[torch.as_tensor(block, device=ic.device)]
        piece = piece[torch.isfinite(piece)]
        if piece.numel() >= min_block_days:
            signs.append(float(piece.mean().item()) * direction > 0)
    consistency = float(np.mean(signs)) if signs else 0.0

    net, _traded = net_long_short_return(
        factor, fwd_ret, direction, args.quantile, args.cost_bps,
        args.min_cross_section, overlap_period,
    )
    ok_health, health = factor_health(factor, missing_rate)
    eligible = torch.isfinite(fwd_ret)
    coverage = float(
        (torch.isfinite(factor) & eligible).sum().float()
        .div(eligible.sum().clamp(min=1)).item()
    )
    passed = (
        aligned >= args.min_ic
        and consistency >= args.min_consistency
        and coverage >= args.min_coverage
        and ok_health
    )
    return dict(
        aligned_ic=aligned,
        icir=icir,
        incremental_ic=incremental,
        net_long_short=float(net) if np.isfinite(net) else float("nan"),
        block_consistency=consistency,
        coverage=coverage,
        n_days=int(finite.numel()),
        health_ok=bool(ok_health),
        zero_frac=health["zero_frac"],
        miss_corr=health["miss_corr_med"],
        low_unique=health["low_unique_frac"],
        passed=bool(passed),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True,
                        help="jsonl written by dripping_gp.py / event_gp.py")
    parser.add_argument("--start", default="2025-01-02")
    parser.add_argument("--end", default="2026-07-31")
    parser.add_argument("--period", type=int, default=None,
                        help="default: the period recorded in the candidates file")
    parser.add_argument(
        "--rebalance", choices=("week_end", "daily"), default=None,
        help="default: the rule recorded in the candidates file",
    )
    parser.add_argument("--signal-average-days", type=int, default=None)
    parser.add_argument("--max-rank", type=int, default=0,
                        help="keep genomes up to this Pareto rank (0 = front only)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-ic", type=float, default=0.02)
    parser.add_argument("--min-consistency", type=float, default=0.66,
                        help="share of blocks that must agree in sign "
                             "(0.66 = 2 of 3 with the default --blocks)")
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--min-ic-days", type=int, default=60)
    parser.add_argument("--min-cross-section", type=int, default=30)
    parser.add_argument("--blocks", type=int, default=3,
                        help="sub-windows used for the sign-stability check")
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS,
                        help="round-trip cost in bp charged on traded "
                             "notional; 30 = 0.15%% per side")
    parser.add_argument("--quantile", type=float, default=0.2)
    parser.add_argument("--minute-parquet", default=str(MINUTE_PARQUET))
    parser.add_argument("--daily-parquet", default=str(ADJUSTED_CLOSE_PARQUET),
                        help="adjusted daily close (ex-dividend safe labels)")
    parser.add_argument("--pit", default=str(ZZ500_PIT_PARQUET))
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--max-stocks", type=int, default=None,
                        help="diagnostic only: restrict the universe")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def resolve_period(records, requested):
    periods = {r.get("provenance", {}).get("period") for r in records} - {None}
    if requested is not None:
        return requested
    if len(periods) > 1:
        raise SystemExit(f"candidates mix periods {sorted(periods)}; pass --period")
    return periods.pop() if periods else 1


def resolve_rebalance(records, requested):
    rules = {
        row.get("provenance", {}).get("rebalance_rule", "daily")
        for row in records
    }
    if requested is not None:
        if len(rules) == 1 and requested != next(iter(rules)):
            raise SystemExit(
                "requested rebalance rule differs from candidate provenance"
            )
        return requested
    if len(rules) != 1:
        raise SystemExit(f"candidates mix rebalance rules {sorted(rules)}")
    return rules.pop()


def resolve_signal_average_days(records, requested):
    windows = {
        int(row.get("provenance", {}).get("signal_average_days", 1))
        for row in records
    }
    if requested is not None:
        if len(windows) == 1 and requested != next(iter(windows)):
            raise SystemExit(
                "requested signal average differs from candidate provenance"
            )
        return requested
    if len(windows) != 1:
        raise SystemExit(f"candidates mix signal windows {sorted(windows)}")
    return windows.pop()


def assert_disjoint(records, start):
    """Refuse to score a window that overlaps the search's own data."""
    train_ends = {r.get("provenance", {}).get("train_end") for r in records} - {None}
    for train_end in sorted(train_ends):
        if train_end >= start:
            raise SystemExit(
                f"hold-out starts {start} but a candidate was trained through "
                f"{train_end}; the result would not be out of sample"
            )
    if not train_ends:
        print("[holdout] WARNING: candidates carry no provenance, cannot verify "
              "the hold-out is disjoint from training", flush=True)


def main():
    args = parse_args()
    minute_path = require_path(args.minute_parquet, "minute parquet")
    daily_path = require_path(args.daily_parquet, "adjusted daily close parquet")
    pit_path = require_path(args.pit, "CSI500 PIT parquet")
    if not args.cpu and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; pass --cpu for a CPU run")
    device = "cpu" if args.cpu else "cuda"

    records = load_candidates(args.candidates, args.max_rank, args.limit)
    if not records:
        raise SystemExit(f"no valid candidates at pareto_rank <= {args.max_rank}")
    families = {detect_family(record) for record in records}
    args.period = resolve_period(records, args.period)
    args.rebalance = resolve_rebalance(records, args.rebalance)
    args.signal_average_days = resolve_signal_average_days(
        records, args.signal_average_days
    )
    assert_disjoint(records, args.start)

    fields = required_minute_fields(records)
    instruments = load_pit_codes(pit_path, args.start, args.end)
    dates = load_pit_dates(pit_path, args.start, args.end)
    if args.max_stocks:
        instruments = instruments[:args.max_stocks]
    print(f"[holdout] {len(records)} candidates, families={sorted(families)}, "
          f"rebalance={args.rebalance}, signal_mean={args.signal_average_days}d, "
          f"window {args.start}..{args.end}", flush=True)

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
    fwd_ret = torch.where(pool_mask, fwd_ret, torch.full_like(fwd_ret, float("nan")))
    reference = minute.get("close", minute["volume"])
    missing_rate = torch.isnan(reference).float().mean(dim=2)
    print(f"[holdout] grid={meta['I']}x{meta['D']}x{meta['NM']}", flush=True)

    def build_factor(family, genome):
        if family == "event_skeleton":
            factor = genome.evaluate(minute, args.chunk_rows)
        elif isinstance(genome, DrippingSkeletonGenome):
            factor = genome.evaluate({"volume": minute["volume"]}, args.chunk_rows)
        elif family == "moderate_risk":
            factor = genome.evaluate(
                minute["close"], minute["volume"], args.chunk_rows
            )
        else:
            factor = genome.evaluate(minute["volume"], args.chunk_rows)
        return torch.where(pool_mask, factor, torch.full_like(factor, float("nan")))

    anchors = {}

    def anchor_for(record, family):
        if family == "event_skeleton":
            genome = event_baseline_genome(record, fields)
        else:
            genome = TEMPLATES[family][0]()
        key = (family, json.dumps(genome.to_dict(), sort_keys=True))
        if key not in anchors:
            anchors[key] = build_factor(family, genome)
            print(f"[holdout] anchor built: {family}", flush=True)
        return anchors[key]

    print(
        f"\n{'#':>3s} {'IC':>8s} {'ICIR':>7s} {'incIC':>8s} {'netLS':>9s} "
        f"{'cons':>5s} {'cov':>5s} {'days':>5s} {'gate':>5s}  genome"
    )
    results = []
    for i, record in enumerate(records):
        family = detect_family(record)
        try:
            genome, dropped = build_genome(family, record["genome"])
        except (TypeError, ValueError) as exc:
            print(f"{i:3d}   SKIP {type(exc).__name__}: {str(exc)[:70]}")
            continue
        if dropped:
            print(f"    note: ignoring stale genome fields {dropped}", flush=True)
        direction = int(
            record.get("fitness", {}).get("direction", TEMPLATES[family][2])
        )
        try:
            factor = build_factor(family, genome)
            metrics = evaluate_factor(
                factor, anchor_for(record, family), fwd_ret, direction, args,
                missing_rate,
            )
            del factor
        except (RuntimeError, ValueError) as exc:
            print(f"{i:3d}   ERR {type(exc).__name__}: {str(exc)[:70]}")
            continue
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if metrics is None:
            print(f"{i:3d}   DROP too few usable IC days")
            continue
        print(
            f"{i:3d} {metrics['aligned_ic']:8.4f} {metrics['icir']:7.2f} "
            f"{metrics['incremental_ic']:8.4f} {metrics['net_long_short']:9.6f} "
            f"{metrics['block_consistency']:5.2f} {metrics['coverage']:5.2f} "
            f"{metrics['n_days']:5d} {'PASS' if metrics['passed'] else 'DROP':>5s}  "
            f"{str(genome)[:70]}"
        )
        results.append(dict(
            family=family,
            direction=direction,
            genome=genome.to_dict(),
            expression=str(genome),
            holdout=metrics,
            train_fitness=record.get("fitness"),
            provenance=record.get("provenance"),
            holdout_window=[args.start, args.end],
            period=args.period,
            rebalance_rule=args.rebalance,
            signal_average_days=args.signal_average_days,
        ))

    destination = args.out or str(output_path("holdout_results.jsonl"))
    passed = [row for row in results if row["holdout"]["passed"]]
    with open(destination, "w", encoding="utf-8") as handle:
        for row in sorted(results, key=lambda r: -r["holdout"]["aligned_ic"]):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n[holdout] {len(passed)}/{len(results)} passed -> {destination}")
    for row in sorted(passed, key=lambda r: -r["holdout"]["aligned_ic"])[:3]:
        train = (row.get("train_fitness") or {}).get("robust_ic")
        train_text = "n/a" if train is None else f"{train:+.4f}"
        print(f"  holdoutIC={row['holdout']['aligned_ic']:+.4f} "
              f"(train robustIC={train_text})  {row['expression'][:70]}")


if __name__ == "__main__":
    main()
