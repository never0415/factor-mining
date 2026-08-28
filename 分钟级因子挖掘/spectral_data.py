"""Lean 240-minute data path for strongly typed GP islands.

Unlike the legacy data builder, this loader reads only minute volume and builds
the actual 240-observation trading sequence. It does not construct S27/S33
features, which keeps spectral experiments within GPU memory.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch


def _minute_index_240(hour: np.ndarray, minute: np.ndarray) -> np.ndarray:
    hm = hour.astype(np.int64) * 60 + minute.astype(np.int64)
    out = np.full(hm.shape, -1, dtype=np.int64)
    morning = (hm >= 570) & (hm <= 689)      # 09:30..11:29
    afternoon = (hm >= 780) & (hm <= 899)    # 13:00..14:59
    out[morning] = hm[morning] - 570
    out[afternoon] = hm[afternoon] - 780 + 120
    return out


def build_minute_slice(
    parquet_path,
    start: str,
    end: str,
    fields=("close", "volume"),
    instruments=None,
    dates=None,
    device: str = "cuda",
) -> tuple[dict[str, torch.Tensor], dict]:
    """Stream selected minute fields into aligned (I,D,240) float32 tensors.

    Passing PIT-derived instruments and dates avoids a preliminary scan. Record
    batches are scattered directly to the output tensor, so multi-year runs do
    not materialize hundreds of millions of Arrow rows in host memory.
    """
    fields = tuple(dict.fromkeys(fields))
    allowed = {"open", "high", "low", "close", "volume"}
    unknown = set(fields) - allowed
    if not fields or unknown:
        raise ValueError(f"invalid minute fields: {sorted(unknown)}")
    filters = [("trade_date", ">=", start), ("trade_date", "<=", end)]
    if instruments is not None:
        filters.append(("instrument", "in", list(instruments)))

    # Discovery is available for diagnostics, but production callers should
    # pass the much smaller PIT-derived grids.
    if instruments is None:
        discovered = pq.read_table(
            parquet_path, columns=["instrument"], filters=filters[:2])
        instruments = pc.unique(discovered.column("instrument")).to_pylist()
    if dates is None:
        discovered = pq.read_table(
            parquet_path, columns=["trade_date"], filters=filters)
        dates = pc.unique(discovered.column("trade_date")).to_pylist()
    insts = sorted(instruments)
    dates = sorted(dates)
    inst_arr, date_arr = pa.array(insts), pa.array(dates)
    I, D, M = len(insts), len(dates), 240
    tensors = {
        field: torch.full(
            (I, D, M), float("nan"), device=device, dtype=torch.float32
        )
        for field in fields
    }
    dataset = ds.dataset(str(parquet_path), format="parquet")
    predicate = (
        (ds.field("trade_date") >= start)
        & (ds.field("trade_date") <= end)
        & ds.field("instrument").isin(insts)
    )
    scanner = dataset.scanner(
        columns=["trade_date", "instrument", "datetime", *fields],
        filter=predicate,
        batch_size=262_144,
        use_threads=True,
    )
    rows_loaded = 0
    for batch in scanner.to_batches():
        inst_col = batch.column("instrument")
        date_col = batch.column("trade_date")
        dt_col = batch.column("datetime")
        i_arr = pc.index_in(inst_col, inst_arr).to_numpy().astype(np.int64)
        d_arr = pc.index_in(date_col, date_arr).to_numpy().astype(np.int64)
        m_arr = _minute_index_240(
            pc.hour(dt_col).to_numpy(), pc.minute(dt_col).to_numpy()
        )
        keep = (m_arr >= 0) & (i_arr >= 0) & (d_arr >= 0)
        if not keep.any():
            continue
        flat = torch.as_tensor(
            i_arr[keep] * (D * M) + d_arr[keep] * M + m_arr[keep],
            device=device, dtype=torch.long,
        )
        for field in fields:
            values = batch.column(field).to_numpy()[keep].astype(np.float32)
            tensors[field].view(-1)[flat] = torch.as_tensor(values, device=device)
        rows_loaded += int(keep.sum())
    if rows_loaded == 0:
        raise ValueError(f"empty volume slice: {start}..{end}")
    meta = {
        "I": I,
        "D": D,
        "NM": M,
        "instruments": insts,
        "dates": dates,
        "start": start,
        "end": end,
        "device": str(device),
        "dtype": "torch.float32",
        "fields": fields,
        "rows_loaded": rows_loaded,
    }
    return tensors, meta


def build_volume_slice(
    parquet_path,
    start: str,
    end: str,
    instruments=None,
    dates=None,
    device: str = "cuda",
) -> tuple[torch.Tensor, dict]:
    """Backward-compatible volume-only wrapper for the spectral island."""
    tensors, meta = build_minute_slice(
        parquet_path, start, end, fields=("volume",),
        instruments=instruments, dates=dates, device=device,
    )
    return tensors["volume"], meta


ADJUSTED_CLOSE_COLUMNS = ("close_badj", "close_hfq", "close_adj")


def _date_bounds(schema, dates):
    """Return (low, high) filter bounds matching the file's trade_date type."""
    field = schema.field("trade_date")
    low, high = min(dates), max(dates)
    if pa.types.is_timestamp(field.type) or pa.types.is_date(field.type):
        return pd.Timestamp(low), pd.Timestamp(high)
    return low, high


def load_daily_close_tensor(
    daily_parquet,
    dates: list[str],
    instruments: list[str],
    device: str = "cuda",
    column: str | None = None,
    require_adjusted: bool = True,
) -> torch.Tensor:
    """Load a daily close aligned to an existing (instrument, date) grid.

    An adjusted column is required by default. An unadjusted close injects a
    fake drop on every ex-dividend day, and because dividends cluster in the
    June-August window that error is seasonal rather than random: a whole
    walk-forward fold can be contaminated at once.
    """
    if not dates or not instruments:
        return torch.empty(
            (len(instruments), len(dates)), device=device, dtype=torch.float32
        )
    schema = pq.ParquetFile(daily_parquet).schema_arrow
    names = set(schema.names)
    if column is None:
        column = next((c for c in ADJUSTED_CLOSE_COLUMNS if c in names), None)
        if column is None:
            if require_adjusted:
                raise ValueError(
                    f"{daily_parquet} exposes no adjusted close column "
                    f"(looked for {ADJUSTED_CLOSE_COLUMNS}). Point at the "
                    "adjusted file, or pass column='close' with "
                    "require_adjusted=False to accept raw prices explicitly."
                )
            column = "close"
    if column not in names:
        raise ValueError(f"column {column!r} not in {sorted(names)}")
    if require_adjusted and column not in ADJUSTED_CLOSE_COLUMNS:
        raise ValueError(
            f"column {column!r} is not an adjusted close; pass "
            "require_adjusted=False to override."
        )

    low, high = _date_bounds(schema, dates)
    table = pq.read_table(
        daily_parquet,
        columns=["trade_date", "instrument", column],
        filters=[
            ("trade_date", ">=", low),
            ("trade_date", "<=", high),
            ("instrument", "in", list(instruments)),
        ],
    )
    frame = table.to_pandas()
    frame["trade_date"] = frame["trade_date"].astype(str).str.slice(0, 10)
    inst_idx = {value: i for i, value in enumerate(instruments)}
    date_idx = {value: i for i, value in enumerate(dates)}
    rows = frame["instrument"].map(inst_idx)
    cols = frame["trade_date"].map(date_idx)
    keep = rows.notna() & cols.notna() & frame[column].notna()
    close = torch.full(
        (len(instruments), len(dates)), float("nan"),
        device=device, dtype=torch.float32,
    )
    if keep.any():
        close[
            torch.as_tensor(rows[keep].to_numpy(np.int64, copy=True), device=device),
            torch.as_tensor(cols[keep].to_numpy(np.int64, copy=True), device=device),
        ] = torch.as_tensor(
            frame.loc[keep, column].to_numpy(np.float32, copy=True), device=device
        )
    return close


def load_daily_factor_leaves(
    parquet_dir,
    dates: list[str],
    instruments: list[str],
    columns: list[str],
    device: str = "cuda",
    require_ok: bool = True,
) -> dict[str, torch.Tensor]:
    """Load cached daily raw factors into aligned (I,D) tensors.

    Quality metadata is used only as a validity mask and is never exposed as a
    GP leaf. This prevents the search from exploiting missingness/liquidity.
    """
    path = Path(parquet_dir)
    if not path.exists():
        raise FileNotFoundError(path)
    sources = [str(path)] if path.is_file() else [
        str(item) for item in sorted(path.rglob("*.parquet"))
    ]
    if not sources:
        raise FileNotFoundError(f"no parquet files under {path}")
    read_columns = ["instrument", "trade_date", *columns]
    if require_ok:
        read_columns.append("status")
    schema = pq.ParquetFile(sources[0]).schema_arrow
    low, high = _date_bounds(schema, dates)
    table = pq.read_table(
        sources, columns=read_columns,
        filters=[
            ("trade_date", ">=", low),
            ("trade_date", "<=", high),
            ("instrument", "in", list(instruments)),
        ],
    )
    frame = table.to_pandas()
    if require_ok:
        frame = frame[frame["status"] == "ok"]
    inst_idx = {v: i for i, v in enumerate(instruments)}
    date_idx = {v: i for i, v in enumerate(dates)}
    ii = frame["instrument"].map(inst_idx)
    dd = frame["trade_date"].astype(str).str.slice(0, 10).map(date_idx)
    valid = ii.notna() & dd.notna()
    i_t = torch.as_tensor(ii[valid].to_numpy(np.int64, copy=True), device=device)
    d_t = torch.as_tensor(dd[valid].to_numpy(np.int64, copy=True), device=device)
    result = {}
    for column in columns:
        value = torch.full(
            (len(instruments), len(dates)), float("nan"),
            device=device, dtype=torch.float32,
        )
        value[i_t, d_t] = torch.as_tensor(
            frame.loc[valid, column].to_numpy(np.float32, copy=True), device=device
        )
        result[column] = value
    return result


def load_daily_exposures(
    parquet_path,
    dates: list[str],
    instruments: list[str],
    continuous_columns=("ln_float_market_cap",),
    industry_column="sw_level1",
    device="cuda",
):
    """Load point-in-time style grids and an integer industry grid."""
    columns = ["instrument", "trade_date", *continuous_columns]
    if industry_column:
        columns.append(industry_column)
    schema = pq.ParquetFile(parquet_path).schema_arrow
    low, high = _date_bounds(schema, dates)
    table = pq.read_table(
        parquet_path, columns=columns,
        filters=[
            ("trade_date", ">=", low),
            ("trade_date", "<=", high),
            ("instrument", "in", list(instruments)),
        ],
    )
    frame = table.to_pandas()
    inst_idx = {value: i for i, value in enumerate(instruments)}
    date_idx = {value: i for i, value in enumerate(dates)}
    ii = frame["instrument"].map(inst_idx)
    dd = frame["trade_date"].astype(str).str.slice(0, 10).map(date_idx)
    located = ii.notna() & dd.notna()
    i_t = torch.as_tensor(ii[located].to_numpy(np.int64, copy=True), device=device)
    d_t = torch.as_tensor(dd[located].to_numpy(np.int64, copy=True), device=device)
    continuous = {}
    for name in continuous_columns:
        grid = torch.full(
            (len(instruments), len(dates)), float("nan"),
            dtype=torch.float32, device=device,
        )
        values = frame.loc[located, name]
        good_np = np.array(values.notna(), dtype=np.bool_, copy=True)
        good = torch.tensor(good_np, dtype=torch.bool, device=device)
        numeric = np.array(values[good_np].tolist(), dtype=np.float32)
        grid[i_t[good], d_t[good]] = torch.tensor(
            numeric, device=device
        )
        continuous[name] = grid
    industry = None
    industry_levels = []
    if industry_column:
        categories = frame[industry_column].astype("category")
        industry_levels = list(categories.cat.categories.astype(str))
        codes = categories.cat.codes.to_numpy(np.int64, copy=True)
        industry = torch.full(
            (len(instruments), len(dates)), -1, dtype=torch.long, device=device
        )
        good = located.to_numpy() & (codes >= 0)
        located_positions = np.flatnonzero(located.to_numpy())
        local_good_np = codes[located_positions] >= 0
        local_good = torch.tensor(local_good_np, dtype=torch.bool, device=device)
        industry[i_t[local_good], d_t[local_good]] = torch.tensor(
            np.array(codes[located_positions][local_good_np], copy=True), device=device
        )
    return continuous, industry, industry_levels
