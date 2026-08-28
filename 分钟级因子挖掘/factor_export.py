"""Persist evaluated typed factors in the daily-leaf interchange format."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch


def factor_frame(
    factor: torch.Tensor, instruments: list[str], dates: list[str],
    column: str = "factor",
) -> pd.DataFrame:
    """Convert an aligned (instrument,date) tensor to the canonical long table."""
    if factor.ndim != 2 or tuple(factor.shape) != (len(instruments), len(dates)):
        raise ValueError(
            f"factor grid {tuple(factor.shape)} does not match "
            f"{len(instruments)} instruments x {len(dates)} dates"
        )
    values = factor.detach().float().cpu().numpy().reshape(-1)
    frame = pd.DataFrame({
        "instrument": np.repeat(np.asarray(instruments), len(dates)),
        "trade_date": np.tile(np.asarray(dates), len(instruments)),
        column: values,
    })
    frame["status"] = np.where(np.isfinite(values), "ok", "invalid")
    return frame


def write_factor_parquet(
    factor: torch.Tensor, instruments: list[str], dates: list[str], path,
    column: str = "factor", metadata: dict | None = None,
) -> Path:
    """Write a typed factor so ``daily_gp --leaf`` can consume it directly."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = factor_frame(factor, instruments, dates, column)
    frame.to_parquet(destination, index=False)
    if metadata is not None:
        sidecar = destination.with_suffix(destination.suffix + ".json")
        sidecar.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return destination


def export_ranked_factors(
    ranked, evaluator, instruments, dates, directory, max_rank=0,
):
    """Export every valid candidate up to a requested Pareto rank."""
    root = Path(directory)
    outputs = []
    for index, (score, genome, rank, _crowding) in enumerate(ranked):
        if rank > max_rank or not score.valid:
            continue
        path = root / f"candidate_{index:04d}_rank{rank}.parquet"
        write_factor_parquet(
            evaluator(genome), instruments, dates, path,
            metadata={
                "pareto_rank": rank,
                "fitness": score.__dict__,
                "genome": genome.to_dict(),
                "expression": str(genome),
            },
        )
        outputs.append(path)
    return outputs


def load_factor_parquet(path, instruments, dates, device="cpu", column="factor"):
    """Read an exported factor back onto a requested (instrument, date) grid.

    The export is a long table, so a pool factor written on one run's grid can
    be realigned to another's. Cells the file does not cover come back NaN and
    are dropped by the usual validity masks rather than silently read as zero.
    """
    table = pq.read_table(path, columns=["instrument", "trade_date", column])
    instrument_index = {str(name): i for i, name in enumerate(instruments)}
    date_index = {str(day): d for d, day in enumerate(dates)}
    rows = np.asarray([
        instrument_index.get(str(name), -1)
        for name in table.column("instrument").combine_chunks().to_pylist()
    ])
    columns = np.asarray([
        date_index.get(str(day), -1)
        for day in table.column("trade_date").combine_chunks().to_pylist()
    ])
    values = table.column(column).combine_chunks().to_numpy(zero_copy_only=False)
    keep = (rows >= 0) & (columns >= 0)
    grid = np.full((len(instruments), len(dates)), np.nan, dtype=np.float32)
    grid[rows[keep], columns[keep]] = values[keep].astype(np.float32)
    return torch.as_tensor(grid, device=device)
