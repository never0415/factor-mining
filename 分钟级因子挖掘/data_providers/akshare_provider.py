"""Small, normalized adapter around AKShare's public A-share history API."""

import pandas as pd


class AkShareProvider:
    def __init__(self):
        try:
            import akshare as ak
        except ImportError as exc:
            raise RuntimeError("install the 'data' extra to use AkShare") from exc
        self.ak = ak

    def daily_stock(self, symbol, start, end, adjust="hfq"):
        frame = self.ak.stock_zh_a_hist(
            symbol=symbol, period="daily",
            start_date=start.replace("-", ""), end_date=end.replace("-", ""),
            adjust=adjust,
        )
        rename = {
            "日期": "trade_date", "开盘": "open", "最高": "high",
            "最低": "low", "收盘": "close", "成交量": "volume",
            "成交额": "amount",
        }
        frame = frame.rename(columns=rename)
        required = ["trade_date", "open", "high", "low", "close", "volume", "amount"]
        missing = set(required) - set(frame)
        if missing:
            raise ValueError(f"AKShare response missing {sorted(missing)}")
        result = frame[required].copy()
        result["trade_date"] = pd.to_datetime(result["trade_date"]).dt.strftime("%Y-%m-%d")
        result["instrument"] = symbol
        result["adjustment"] = adjust
        return result

