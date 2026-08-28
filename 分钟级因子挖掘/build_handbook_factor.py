"""Build one strict local-OHLCV handbook factor and save an aligned parquet."""

import argparse

import numpy as np
import pandas as pd
import torch

from min_gp.config import MINUTE_PARQUET, ZZ500_PIT_PARQUET, output_path, require_path
from min_gp.data import load_pit_codes, load_pit_daily_mask, load_pit_dates
from min_gp.factors import (
    HANDBOOK_FACTORS, LOCAL_MINUTE_FACTORS, evaluate_local_minute_factor,
)
from min_gp.spectral_data import build_minute_slice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", required=True, choices=LOCAL_MINUTE_FACTORS)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--minute-parquet", default=str(MINUTE_PARQUET))
    parser.add_argument("--pit", default=str(ZZ500_PIT_PARQUET))
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-stocks", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    device = "cpu" if args.cpu else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; pass --cpu")
    pit = require_path(args.pit, "PIT universe")
    instruments = load_pit_codes(pit, args.start, args.end)
    dates = load_pit_dates(pit, args.start, args.end)
    if args.max_stocks:
        instruments = instruments[:args.max_stocks]
    fields = HANDBOOK_FACTORS[args.factor].required_fields
    minute, meta = build_minute_slice(
        require_path(args.minute_parquet, "minute parquet"),
        args.start, args.end, fields=fields, instruments=instruments,
        dates=dates, device=device,
    )
    factor = evaluate_local_minute_factor(args.factor, minute)
    pool = load_pit_daily_mask(
        pit, meta["dates"], meta["instruments"], device=device
    )
    factor = torch.where(pool, factor, torch.full_like(factor, float("nan")))
    values = factor.detach().cpu().numpy()
    frame = pd.DataFrame({
        "instrument": np.repeat(meta["instruments"], len(meta["dates"])),
        "trade_date": np.tile(meta["dates"], len(meta["instruments"])),
        "factor": values.reshape(-1),
    })
    frame["status"] = np.where(np.isfinite(frame["factor"]), "ok", "invalid")
    destination = args.out or str(output_path(f"{args.factor}.parquet"))
    frame.to_parquet(destination, index=False)
    print(f"[handbook] {args.factor}: {factor.shape} -> {destination}")


if __name__ == "__main__":
    main()

