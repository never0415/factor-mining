"""Authenticated TqSdk adapter with credentials sourced only from environment."""

import os


class TqSdkProvider:
    def __init__(self, user=None, password=None):
        try:
            from dotenv import load_dotenv
            load_dotenv()
            from tqsdk import TqApi, TqAuth
        except ImportError as exc:
            raise RuntimeError("install the 'data' extra to use TqSdk") from exc
        user = user or os.environ.get("TQ_USER")
        password = password or os.environ.get("TQ_PASS")
        if not user or not password:
            raise RuntimeError("TQ_USER and TQ_PASS must be set in the environment")
        self._api = TqApi(auth=TqAuth(user, password))

    def recent_klines(self, symbol, duration_seconds=60, data_length=8964):
        """Return a detached copy; TqSdk's live serial mutates on updates."""
        serial = self._api.get_kline_serial(
            symbol, int(duration_seconds), data_length=int(data_length)
        )
        result = serial.copy()
        result["instrument"] = symbol
        return result

    def close(self):
        if self._api is not None:
            self._api.close()
            self._api = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

