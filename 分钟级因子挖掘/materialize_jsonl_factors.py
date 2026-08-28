"""Evaluate the genomes in a GP run log and export them as daily factors.

A run log records every candidate's genome but only the ``--export-rank``
subset is ever written as a factor panel. Correlation clustering needs the
panels, not the expressions, so this walks a ``.jsonl`` and materialises what
the run only described -- for example the 76 event-skeleton candidates in
``all76.jsonl``, none of which were exported.

Output matches ``factor_export``: one parquet per candidate plus a sidecar
holding the genome and its stored fitness, so the result drops straight into
``factor_combination_report`` and the IC gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from min_gp.config import (
    ADJUSTED_CLOSE_PARQUET, MINUTE_PARQUET, ZZ500_PIT_PARQUET, require_path,
)
from min_gp.data import load_pit_codes, load_pit_daily_mask, load_pit_dates
from min_gp.factor_export import write_factor_parquet
from min_gp.factors.catalog import genome_from_export, infer_genome_kind
from min_gp.factors.seed_tree import SeedTreeGenome
from min_gp.operators import build_operator_registry
from min_gp.spectral_data import build_minute_slice, load_daily_close_tensor


RAW_MINUTE_FIELDS = {"open", "high", "low", "close", "volume"}


def _evaluate(genome, context, registry, chunk_rows):
    if isinstance(genome, SeedTreeGenome):
        return genome.evaluate(context, registry, chunk_rows)
    return genome.evaluate(context, chunk_rows=chunk_rows)


def _records(path: Path, rank_limit, keep_invalid):
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        fitness = record.get("fitness", {})
        if not keep_invalid and not fitness.get("valid", True):
            continue
        if rank_limit is not None and record.get("pareto_rank", 0) > rank_limit:
            continue
        yield record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jsonl", required=True, help="GP run log to materialise")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--start", default="2018-01-02")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--minute-parquet", default=str(MINUTE_PARQUET))
    parser.add_argument("--daily-parquet", default=str(ADJUSTED_CLOSE_PARQUET))
    parser.add_argument("--pit", default=str(ZZ500_PIT_PARQUET))
    parser.add_argument(
        "--leaf-npz", action="append", default=[], metavar="NAME=PATH:KEY",
        help="report-specific leaf for genomes that need one",
    )
    parser.add_argument("--max-rank", type=int, default=None,
                        help="skip candidates above this Pareto rank")
    parser.add_argument("--keep-invalid", action="store_true")
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="leave panels that are already written alone, so an interrupted "
             "run resumes instead of restarting",
    )
    parser.add_argument("--prefix", default="candidate")
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)
    device = "cpu" if args.cpu else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; pass --cpu")

    registry = build_operator_registry()
    records = list(_records(Path(args.jsonl), args.max_rank, args.keep_invalid))
    if not records:
        raise SystemExit(f"no usable records in {args.jsonl}")
    genomes = [genome_from_export(record["genome"], registry) for record in records]
    kinds = {infer_genome_kind(record["genome"]) for record in records}
    required = sorted({field for g in genomes for field in g.required_fields})
    print(
        f"[materialize] {len(records)} candidates, families {sorted(kinds)}, "
        f"leaves {required}",
        flush=True,
    )

    pit = require_path(args.pit, "PIT universe")
    instruments = load_pit_codes(pit, args.start, args.end)
    dates = load_pit_dates(pit, args.start, args.end)
    context = {}
    for value in args.leaf_npz:
        from min_gp.handbook_gp import load_npz_leaves_aligned, parse_leaf
        context.update(load_npz_leaves_aligned(
            [parse_leaf(value)], instruments, dates, device
        ))
    minute_fields = sorted(set(required) & RAW_MINUTE_FIELDS - set(context))
    if minute_fields:
        minute, meta = build_minute_slice(
            require_path(args.minute_parquet, "minute parquet"),
            args.start, args.end, fields=tuple(minute_fields),
            instruments=instruments, dates=dates, device=device,
        )
        context.update(minute)
        instruments = [str(v) for v in meta["instruments"]]
        dates = [str(v) for v in meta["dates"]]
    missing = sorted(set(required) - set(context))
    if missing:
        raise SystemExit(
            f"genomes need leaves {missing}; supply each as --leaf-npz NAME=PATH:KEY"
        )

    close = load_daily_close_tensor(
        args.daily_parquet, dates, instruments, device=device
    )
    pool = load_pit_daily_mask(pit, dates, instruments, device=device)
    pool &= torch.isfinite(close)
    blank = torch.full((len(instruments), len(dates)), float("nan"), device=device)

    out_dir = Path(args.out_dir)
    written, skipped, failed = 0, 0, []
    for index, (record, genome) in enumerate(zip(records, genomes)):
        destination = (
            out_dir
            / f"{args.prefix}_{index:04d}_rank{int(record.get('pareto_rank', 0))}.parquet"
        )
        if args.skip_existing and destination.exists():
            skipped += 1
            continue
        try:
            factor = _evaluate(genome, context, registry, args.chunk_rows).float()
            # The universe mask belongs on the panel, not on the reader: an
            # exported factor is consumed by several scripts and each one
            # re-deriving "was this name investable" is how they drift apart.
            factor = torch.where(pool, factor, blank)
            write_factor_parquet(
                factor, instruments, dates, destination,
                metadata={
                    "pareto_rank": int(record.get("pareto_rank", 0)),
                    "fitness": record.get("fitness", {}),
                    "genome": record["genome"],
                    "expression": record.get("expression", ""),
                    "provenance": record.get("provenance", {}),
                    "materialized_from": str(args.jsonl),
                },
            )
            written += 1
        except (RuntimeError, ValueError, TypeError, KeyError) as error:
            failed.append((index, type(error).__name__, str(error)[:120]))
        if (index + 1) % 10 == 0:
            print(f"[materialize] {index + 1}/{len(records)}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(
        f"[materialize] wrote {written}/{len(records)} panels"
        + (f", skipped {skipped} already present" if skipped else "")
        + f" -> {out_dir}"
    )
    for index, name, message in failed:
        print(f"  candidate {index:04d} failed: {name}: {message}")


if __name__ == "__main__":
    main()
