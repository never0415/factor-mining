"""Build auditable CSI-500 OHLCV leaves for the Rushing-Forward island.

This is explicitly a reproducible reconstruction because the source minute
parquet has no traded-amount column and the handbook report leaves are not
available.  Amount is close * volume; shares are cross-sectional within each
minute over point-in-time CSI-500 members; the event is a within-session
volume increase paired with a price decrease from the immediately preceding
minute.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from min_gp.data import load_pit_codes, load_pit_daily_mask, load_pit_dates
from min_gp.spectral_data import build_minute_slice


def _write_npz(path: Path, key: str, value: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{key: value.detach().cpu().numpy()})
    print(f"[rushing-leaves] {key} -> {path} ({path.stat().st_size} bytes)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minute-parquet", required=True)
    parser.add_argument("--pit", required=True)
    parser.add_argument("--start", default="2018-01-02")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    device = "cpu" if args.cpu else "cuda"

    instruments = load_pit_codes(args.pit, args.start, args.end)
    dates = load_pit_dates(args.pit, args.start, args.end)
    print(
        f"[rushing-leaves] load close/volume I={len(instruments)} D={len(dates)}",
        flush=True,
    )
    context, meta = build_minute_slice(
        args.minute_parquet,
        args.start,
        args.end,
        fields=("close", "volume"),
        instruments=instruments,
        dates=dates,
        device=device,
    )
    instruments = [str(value) for value in meta["instruments"]]
    dates = [str(value) for value in meta["dates"]]
    close = context["close"].float()
    volume = context["volume"].float()
    pool = load_pit_daily_mask(
        args.pit, dates, instruments, device=device
    ).unsqueeze(-1)
    valid = pool & torch.isfinite(close) & torch.isfinite(volume) & volume.ge(0)

    safe_volume = torch.where(valid, volume, torch.zeros_like(volume))
    amount = torch.where(valid, close * volume, torch.zeros_like(volume))
    volume_denominator = safe_volume.sum(dim=0, keepdim=True)
    amount_denominator = amount.sum(dim=0, keepdim=True)
    volume_share = safe_volume / volume_denominator.clamp(min=1e-12)
    amount_share = amount / amount_denominator.clamp(min=1e-12)
    volume_share = torch.where(
        valid & volume_denominator.gt(0),
        volume_share,
        torch.full_like(volume_share, float("nan")),
    )
    amount_share = torch.where(
        valid & amount_denominator.gt(0),
        amount_share,
        torch.full_like(amount_share, float("nan")),
    )

    event = torch.zeros_like(valid)
    consecutive = valid[..., 1:] & valid[..., :-1]
    event[..., 1:] = (
        consecutive
        & volume[..., 1:].gt(volume[..., :-1])
        & close[..., 1:].lt(close[..., :-1])
    )

    finite_volume_share = torch.isfinite(volume_share)
    finite_amount_share = torch.isfinite(amount_share)
    sample_minutes = finite_volume_share.any(dim=0)
    volume_sums = torch.where(
        finite_volume_share, volume_share, torch.zeros_like(volume_share)
    ).sum(dim=0)
    amount_sums = torch.where(
        finite_amount_share, amount_share, torch.zeros_like(amount_share)
    ).sum(dim=0)
    max_volume_sum_error = float(
        (volume_sums[sample_minutes] - 1).abs().max()
    )
    max_amount_sum_error = float(
        (amount_sums[sample_minutes] - 1).abs().max()
    )
    if event[..., 0].any():
        raise ValueError("first session minute must never be an event")
    if max(max_volume_sum_error, max_amount_sum_error) > 2e-5:
        raise ValueError("cross-sectional shares do not sum to one")

    out = Path(args.out_dir)
    _write_npz(out / "amount_share.npz", "amount_share", amount_share)
    _write_npz(out / "volume_share.npz", "volume_share", volume_share)
    _write_npz(
        out / "up_volume_down_price_mask.npz",
        "up_volume_down_price_mask",
        event,
    )
    metadata = {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "variant": "CSI500_OHLCV_reconstruction_v1",
        "source_minute_parquet": str(args.minute_parquet),
        "source_pit": str(args.pit),
        "start": args.start,
        "end": args.end,
        "shape": list(amount_share.shape),
        "instruments": instruments,
        "dates": dates,
        "definitions": {
            "amount": "minute_close * minute_volume",
            "amount_share": "amount_i,d,m / sum_PIT_CSI500_i amount_i,d,m",
            "volume_share": "volume_i,d,m / sum_PIT_CSI500_i volume_i,d,m",
            "up_volume_down_price_mask": "m>0 and volume[m]>volume[m-1] and close[m]<close[m-1]; reset each session",
            "information_set": "minute m and immediately preceding minute only; no forward data",
        },
        "validation": {
            "max_volume_share_sum_error": max_volume_sum_error,
            "max_amount_share_sum_error": max_amount_sum_error,
            "share_finite_ratio": float(
                finite_volume_share.sum() / pool.expand_as(valid).sum().clamp(min=1)
            ),
            "event_ratio_on_valid_minutes": float(
                event.sum() / valid.sum().clamp(min=1)
            ),
            "first_minute_event_count": int(event[..., 0].sum()),
        },
    }
    (out / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "[rushing-leaves] validation "
        f"share_error={max(max_volume_sum_error, max_amount_sum_error):.2e} "
        f"event_ratio={metadata['validation']['event_ratio_on_valid_minutes']:.4%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
