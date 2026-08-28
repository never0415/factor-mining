"""Build point-in-time daily style exposures used by neutralisation reports.

Daily valuation history and historical financial statements come from
AkShare/Eastmoney.  Financial values become usable on the first trading day
after their actual notice date.  Old-report restatements never displace a more
recent report period, while revisions to the latest report do take effect on
their own notice date.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd
import requests

from min_gp.config import ZZ500_PIT_PARQUET


EASTMONEY_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
BALANCE_REPORT = "RPT_DMSK_FN_BALANCE"
INCOME_REPORT = "RPT_DMSK_FN_INCOME"
BALANCE_COLUMNS = (
    "SECURITY_CODE", "REPORT_DATE", "NOTICE_DATE", "DATA_STATE",
    "TOTAL_ASSETS", "TOTAL_LIABILITIES", "TOTAL_EQUITY_RATIO",
)
INCOME_COLUMNS = (
    "SECURITY_CODE", "REPORT_DATE", "NOTICE_DATE", "DATA_STATE",
    "TOTAL_OPERATE_INCOME", "TOI_RATIO", "PARENT_NETPROFIT",
    "PARENT_NETPROFIT_RATIO",
)


def _code(instrument: str) -> str:
    return "".join(ch for ch in str(instrument) if ch.isdigit())[-6:]


def _instrument(code: str) -> str:
    value = str(code).zfill(6)
    if value.startswith(("4", "8", "9")):
        return f"bj{value}"
    return f"sh{value}" if value.startswith(("5", "6")) else f"sz{value}"


def _quarter_dates(start: str, end: str) -> list[str]:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    values = []
    for year in range(start_ts.year, end_ts.year + 1):
        for month, day in ((3, 31), (6, 30), (9, 30), (12, 31)):
            value = pd.Timestamp(year, month, day)
            if start_ts <= value <= end_ts:
                values.append(value.strftime("%Y%m%d"))
    return values


def _request_json(session: requests.Session, params: dict, attempts: int = 5) -> dict:
    error = None
    for attempt in range(attempts):
        try:
            response = session.get(EASTMONEY_URL, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            if payload.get("success") and payload.get("result") is not None:
                return payload
            error = RuntimeError(f"Eastmoney response: {payload.get('message')}")
        except Exception as exc:  # network retries are intentionally broad
            error = exc
        time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Eastmoney request failed after {attempts} attempts: {error}")


def _fetch_statement(
    report_name: str,
    report_date: str,
    destination: Path,
    universe_codes: set[str],
) -> tuple[str, int, bool]:
    if destination.exists() and destination.stat().st_size > 0:
        return destination.name, -1, True
    date_text = f"{report_date[:4]}-{report_date[4:6]}-{report_date[6:]}"
    params = {
        "sortColumns": "NOTICE_DATE,SECURITY_CODE",
        "sortTypes": "-1,-1",
        "pageSize": "500",
        "pageNumber": "1",
        "reportName": report_name,
        "columns": "ALL",
        "filter": (
            '(SECURITY_TYPE_CODE in ("058001001","058001008"))'
            '(TRADE_MARKET_CODE!="069001017")'
            f"(REPORT_DATE='{date_text}')"
        ),
    }
    session = requests.Session()
    first = _request_json(session, params)
    pages = int(first["result"].get("pages") or 0)
    rows = []
    for page in range(1, pages + 1):
        payload = first if page == 1 else _request_json(
            session, {**params, "pageNumber": str(page)}
        )
        for row in payload["result"].get("data") or []:
            if str(row.get("SECURITY_CODE", "")).zfill(6) in universe_codes:
                rows.append(row)
    columns = BALANCE_COLUMNS if report_name == BALANCE_REPORT else INCOME_COLUMNS
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame:
            frame[column] = np.nan
    frame = frame.loc[:, columns]
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
    return destination.name, len(frame), False


def _fetch_valuation(symbol: str, destination: Path) -> tuple[str, str, bool]:
    if destination.exists() and destination.stat().st_size > 0:
        return symbol, "cached", True
    error = None
    for attempt in range(4):
        try:
            frame = ak.stock_value_em(symbol=symbol)
            if frame is None or frame.empty:
                raise RuntimeError("empty stock_value_em response")
            pe = pd.to_numeric(frame["PE(TTM)"], errors="coerce")
            result = pd.DataFrame({
                "instrument": _instrument(symbol),
                "trade_date": pd.to_datetime(frame["数据日期"], errors="coerce"),
                "earnings_yield": np.where(pe.ne(0), 1.0 / pe, np.nan),
            }).dropna(subset=["trade_date"])
            result["trade_date"] = result["trade_date"].dt.strftime("%Y-%m-%d")
            destination.parent.mkdir(parents=True, exist_ok=True)
            result.to_parquet(destination, index=False)
            return symbol, "ok", False
        except Exception as exc:  # AkShare wraps several network backends
            error = exc
            time.sleep(min(2 ** attempt, 8))
    return symbol, repr(error), False


def _load_statement_events(paths: list[Path], value_columns: tuple[str, ...]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if path.exists() and path.stat().st_size > 0:
            frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame(columns=("instrument", "REPORT_DATE", "NOTICE_DATE", *value_columns))
    frame = pd.concat(frames, ignore_index=True)
    frame["instrument"] = frame["SECURITY_CODE"].map(_instrument)
    frame["REPORT_DATE"] = pd.to_datetime(frame["REPORT_DATE"], errors="coerce")
    frame["NOTICE_DATE"] = pd.to_datetime(frame["NOTICE_DATE"], errors="coerce")
    for column in value_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["REPORT_DATE", "NOTICE_DATE"])
    return frame[["instrument", "REPORT_DATE", "NOTICE_DATE", *value_columns]]


def _latest_report_events(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Keep only announcements that are current as of their notice date."""
    result = {}
    for instrument, group in frame.groupby("instrument", sort=False):
        group = group.sort_values(["NOTICE_DATE", "REPORT_DATE"], kind="stable")
        keep = []
        latest_report = pd.Timestamp.min
        for row in group.itertuples(index=False):
            report_date = row.REPORT_DATE
            if report_date >= latest_report:
                keep.append(True)
                latest_report = max(latest_report, report_date)
            else:
                keep.append(False)
        group = group.loc[keep].copy()
        group["effective_date"] = group["NOTICE_DATE"] + pd.Timedelta(days=1)
        # If several statements are released together, consume the newest period.
        group = group.sort_values(["effective_date", "REPORT_DATE"], kind="stable")
        group = group.drop_duplicates("effective_date", keep="last")
        result[instrument] = group
    return result


def _asof_values(
    dates: pd.Series,
    events: pd.DataFrame | None,
    value_columns: tuple[str, ...],
    max_age_days: int = 550,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    count = len(dates)
    missing = {column: np.full(count, np.nan, dtype=np.float64) for column in value_columns}
    missing_source = np.full(count, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    if events is None or events.empty:
        return missing, missing_source
    event_dates = events["effective_date"].to_numpy(dtype="datetime64[ns]")
    target_dates = pd.to_datetime(dates).to_numpy(dtype="datetime64[ns]")
    positions = np.searchsorted(event_dates, target_dates, side="right") - 1
    valid = positions >= 0
    notices = events["NOTICE_DATE"].to_numpy(dtype="datetime64[ns]")
    located = np.flatnonzero(valid)
    if located.size:
        ages = (
            target_dates[located] - notices[positions[located]]
        ) / np.timedelta64(1, "D")
        valid[located[ages > max_age_days]] = False
    values = {}
    for column in value_columns:
        output = np.full(count, np.nan, dtype=np.float64)
        source = events[column].to_numpy(dtype=np.float64)
        output[valid] = source[positions[valid]]
        values[column] = output
    source_dates = np.full(count, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    source_dates[valid] = notices[positions[valid]]
    return values, source_dates


def _assemble_daily(
    pit: pd.DataFrame,
    valuation_dir: Path,
    balance_events: dict[str, pd.DataFrame],
    income_events: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    pieces = []
    balance_values = ("TOTAL_ASSETS", "TOTAL_LIABILITIES", "TOTAL_EQUITY_RATIO")
    income_values = ("TOI_RATIO", "PARENT_NETPROFIT_RATIO")
    for instrument, group in pit.groupby("instrument", sort=False):
        group = group.sort_values("trade_date").copy()
        bvalues, bsource = _asof_values(
            group["trade_date"], balance_events.get(instrument), balance_values
        )
        ivalues, isource = _asof_values(
            group["trade_date"], income_events.get(instrument), income_values
        )
        group["leverage"] = bvalues["TOTAL_LIABILITIES"] / bvalues["TOTAL_ASSETS"]
        group["equity_growth_yoy"] = bvalues["TOTAL_EQUITY_RATIO"]
        group["revenue_growth_yoy"] = ivalues["TOI_RATIO"]
        group["profit_growth_yoy"] = ivalues["PARENT_NETPROFIT_RATIO"]
        group["balance_source_date"] = bsource
        group["income_source_date"] = isource
        value_path = valuation_dir / f"{_code(instrument)}.parquet"
        group["earnings_yield"] = np.nan
        if value_path.exists():
            value = pd.read_parquet(value_path)
            value["trade_date"] = value["trade_date"].astype(str).str.slice(0, 10)
            lookup = value.drop_duplicates("trade_date", keep="last").set_index("trade_date")["earnings_yield"]
            group["earnings_yield"] = group["trade_date"].map(lookup)
        pieces.append(group)
    result = pd.concat(pieces, ignore_index=True)
    numeric = [
        "earnings_yield", "leverage", "equity_growth_yoy",
        "revenue_growth_yoy", "profit_growth_yoy",
    ]
    finite_count = np.isfinite(result[numeric]).sum(axis=1)
    result["status"] = np.where(finite_count > 0, "ok", "missing")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pit", default=str(ZZ500_PIT_PARQUET))
    parser.add_argument("--start", default="2018-01-02")
    parser.add_argument("--end", default="2026-07-31")
    parser.add_argument("--raw-dir", default=r"F:\fangzheng research\data\raw\akshare_style_exposures")
    parser.add_argument("--out", default=r"F:\fangzheng research\data\interim\zz500_fundamental_style_exposures.parquet")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--assemble-only", action="store_true")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    valuation_dir = raw_dir / "stock_value_em"
    statement_dir = raw_dir / "statements"
    pit = pd.read_parquet(args.pit)
    pit["trade_date"] = pit["trade_date"].astype(str).str.slice(0, 10)
    pit = pit[(pit["trade_date"] >= args.start) & (pit["trade_date"] <= args.end)]
    codes = sorted({_code(value) for value in pit["instrument"].unique()})
    failures = []
    if args.assemble_only:
        failures = [
            {"symbol": code, "error": "no cached stock_value_em response"}
            for code in codes if not (valuation_dir / f"{code}.parquet").exists()
        ]
    if not args.assemble_only:
        print(f"[style-data] {len(codes)} PIT instruments; download daily PE(TTM)", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_fetch_valuation, code, valuation_dir / f"{code}.parquet"): code
                for code in codes
            }
            for done, future in enumerate(as_completed(futures), 1):
                code, status, cached = future.result()
                if status not in {"ok", "cached"}:
                    failures.append({"symbol": code, "error": status})
                if done % 50 == 0 or done == len(futures):
                    print(f"[style-data] valuation {done}/{len(futures)}; failures={len(failures)}", flush=True)

    report_dates = _quarter_dates("2017-09-30", "2026-06-30")
    tasks = []
    for date in report_dates:
        tasks.extend([
            (BALANCE_REPORT, date, statement_dir / "balance" / f"{date}.parquet"),
            (INCOME_REPORT, date, statement_dir / "income" / f"{date}.parquet"),
        ])
    statement_failures = []
    if args.assemble_only:
        statement_failures = [
            {"report": report, "date": date, "error": "missing cached statement panel"}
            for report, date, path in tasks if not path.exists()
        ]
    if not args.assemble_only:
        print(f"[style-data] download/cache {len(tasks)} quarterly statement panels", flush=True)
        with ThreadPoolExecutor(max_workers=min(args.workers, 4)) as executor:
            futures = {
                executor.submit(_fetch_statement, report, date, path, set(codes)): (report, date)
                for report, date, path in tasks
            }
            for done, future in enumerate(as_completed(futures), 1):
                report, date = futures[future]
                try:
                    name, rows, cached = future.result()
                except Exception as exc:
                    statement_failures.append({"report": report, "date": date, "error": repr(exc)})
                if done % 6 == 0 or done == len(futures):
                    print(f"[style-data] statements {done}/{len(futures)}; failures={len(statement_failures)}", flush=True)

    balance_paths = [statement_dir / "balance" / f"{date}.parquet" for date in report_dates]
    income_paths = [statement_dir / "income" / f"{date}.parquet" for date in report_dates]
    balance = _load_statement_events(
        balance_paths, ("TOTAL_ASSETS", "TOTAL_LIABILITIES", "TOTAL_EQUITY_RATIO")
    )
    income = _load_statement_events(
        income_paths, ("TOI_RATIO", "PARENT_NETPROFIT_RATIO")
    )
    end = pd.Timestamp(args.end)
    balance = balance[balance["NOTICE_DATE"] <= end]
    income = income[income["NOTICE_DATE"] <= end]
    print(f"[style-data] assemble PIT daily grids: balance={len(balance)}, income={len(income)}", flush=True)
    daily = _assemble_daily(
        pit, valuation_dir, _latest_report_events(balance), _latest_report_events(income)
    )
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(destination, index=False)
    manifest = {
        "generated": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "source": "AkShare stock_value_em and Eastmoney historical financial statements",
        "financial_effective_rule": "notice date + 1 calendar day, then next available trading date",
        "rows": len(daily),
        "instruments": int(daily["instrument"].nunique()),
        "start": args.start,
        "end": args.end,
        "valuation_failures": failures,
        "statement_failures": statement_failures,
        "coverage": {
            column: float(daily[column].notna().mean())
            for column in (
                "earnings_yield", "leverage", "equity_growth_yoy",
                "revenue_growth_yoy", "profit_growth_yoy",
            )
        },
    }
    destination.with_suffix(".json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[style-data] written -> {destination}", flush=True)


if __name__ == "__main__":
    main()
