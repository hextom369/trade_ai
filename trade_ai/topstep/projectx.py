"""Minimal client for the ProjectX Gateway API used by TopstepX.

Only the standard library is used. Endpoints (POST, JSON):
  /api/Auth/loginKey, /api/Account/search, /api/Contract/search,
  /api/History/retrieveBars, /api/Order/place, /api/Order/cancel,
  /api/Order/searchOpen, /api/Position/searchOpen, /api/Position/closeContract

API access needs a ProjectX API subscription and an API key from the
TopstepX settings. Credentials are read from the environment
(PROJECTX_USERNAME / PROJECTX_API_KEY) so they never end up in the repo.
"""

from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

DEFAULT_BASE_URL = "https://api.topstepx.com"

# enums
ORDER_LIMIT, ORDER_MARKET, ORDER_STOP = 1, 2, 4
SIDE_BUY, SIDE_SELL = 0, 1
POSITION_LONG, POSITION_SHORT = 1, 2
UNIT_SECOND, UNIT_MINUTE, UNIT_HOUR, UNIT_DAY = 1, 2, 3, 4

Transport = Callable[[str, dict, dict], dict]


class ProjectXError(RuntimeError):
    pass


def _urllib_transport(url: str, body: dict, headers: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Accept": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - fixed https host
        return json.loads(resp.read().decode() or "{}")


def _iso(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


class ProjectXClient:
    def __init__(self, username: str | None = None, api_key: str | None = None,
                 base_url: str | None = None, transport: Transport | None = None):
        self.username = username or os.environ.get("PROJECTX_USERNAME", "")
        self.api_key = api_key or os.environ.get("PROJECTX_API_KEY", "")
        self.base_url = (base_url or os.environ.get("PROJECTX_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.transport = transport or _urllib_transport
        self.token: str | None = None

    # -- plumbing ----------------------------------------------------------
    def _post(self, path: str, body: dict, auth: bool = True) -> dict:
        headers = {}
        if auth:
            if self.token is None:
                self.login()
            headers["Authorization"] = f"Bearer {self.token}"
        data = self.transport(self.base_url + path, body, headers)
        if not data.get("success", False):
            raise ProjectXError(f"{path} failed: errorCode={data.get('errorCode')} "
                                f"{data.get('errorMessage') or ''}".strip())
        return data

    def login(self) -> str:
        if not self.username or not self.api_key:
            raise ProjectXError("set PROJECTX_USERNAME and PROJECTX_API_KEY")
        data = self._post("/api/Auth/loginKey",
                          {"userName": self.username, "apiKey": self.api_key}, auth=False)
        self.token = data["token"]
        return self.token

    # -- account / contracts ----------------------------------------------
    def accounts(self, only_active: bool = True) -> list[dict]:
        return self._post("/api/Account/search", {"onlyActiveAccounts": only_active})["accounts"]

    def search_contracts(self, text: str, live: bool = False) -> list[dict]:
        return self._post("/api/Contract/search", {"searchText": text, "live": live})["contracts"]

    def front_contract_id(self, symbol: str) -> str:
        contracts = self.search_contracts(symbol)
        active = [c for c in contracts if c.get("activeContract")] or contracts
        if not active:
            raise ProjectXError(f"no contract found for {symbol}")
        return active[0]["id"]

    # -- market data -------------------------------------------------------
    def bars(self, contract_id: str, start: datetime, end: datetime, unit: int = UNIT_MINUTE,
             unit_number: int = 1, limit: int = 20_000, live: bool = False,
             include_partial: bool = False) -> pd.DataFrame:
        data = self._post("/api/History/retrieveBars", {
            "contractId": contract_id, "live": live, "startTime": _iso(start),
            "endTime": _iso(end), "unit": unit, "unitNumber": unit_number, "limit": limit,
            "includePartialBar": include_partial,
        })
        rows = data.get("bars") or []
        df = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"]).rename(
            columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        df.index = pd.to_datetime(df.pop("t"), utc=True)
        df.index.name = "date"
        return df.astype(float).sort_index()

    def bars_range(self, contract_id: str, start: datetime, end: datetime,
                   unit_number: int = 1, chunk_days: int = 10) -> pd.DataFrame:
        """Download long ranges in chunks (the API caps bars per request)."""
        parts = []
        t = pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tz is None else pd.Timestamp(start)
        end_ts = pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tz is None else pd.Timestamp(end)
        while t < end_ts:
            nxt = min(t + pd.Timedelta(days=chunk_days), end_ts)
            parts.append(self.bars(contract_id, t.to_pydatetime(), nxt.to_pydatetime(),
                                   unit_number=unit_number))
            t = nxt
        df = pd.concat(parts) if parts else pd.DataFrame()
        return df[~df.index.duplicated(keep="last")].sort_index()

    # -- orders / positions ------------------------------------------------
    def place_order(self, account_id: int, contract_id: str, order_type: int, side: int,
                    size: int, limit_price: float | None = None,
                    stop_price: float | None = None, tag: str | None = None) -> int:
        body: dict[str, Any] = {"accountId": account_id, "contractId": contract_id,
                                "type": order_type, "side": side, "size": int(size)}
        if limit_price is not None:
            body["limitPrice"] = limit_price
        if stop_price is not None:
            body["stopPrice"] = stop_price
        if tag:
            body["customTag"] = tag
        return int(self._post("/api/Order/place", body)["orderId"])

    def cancel_order(self, account_id: int, order_id: int) -> None:
        self._post("/api/Order/cancel", {"accountId": account_id, "orderId": order_id})

    def open_orders(self, account_id: int) -> list[dict]:
        return self._post("/api/Order/searchOpen", {"accountId": account_id})["orders"]

    def open_positions(self, account_id: int) -> list[dict]:
        return self._post("/api/Position/searchOpen", {"accountId": account_id})["positions"]

    def close_position(self, account_id: int, contract_id: str) -> None:
        self._post("/api/Position/closeContract",
                   {"accountId": account_id, "contractId": contract_id})
