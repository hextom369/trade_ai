"""Variational Omni adapter.

Status (checked 2026-10): Variational publishes a read-only REST API with market
statistics. The *trading* API (orders / RFQs, positions, margin) is announced
but not available to users yet, so ``VariationalBroker`` can price and paper
trade against Omni's live mark prices, but refuses to send real orders.
When the trading API ships, implement ``equity``, ``position`` and
``market_order`` here; nothing else in the bot has to change.

The listing field names below follow the public docs / third-party indexers
(``ticker``, ``mark_price``, ``funding_rate``, ``quotes``). The parser is
deliberately tolerant; verify against https://docs.variational.io if it breaks.
"""

from __future__ import annotations

from typing import Any

from .brokers import Broker, Fill
from .crypto_data import HttpGet, http_get_json

VARIATIONAL_API = "https://omni-client-api.prod.ap-northeast-1.variational.io"

TRADING_API_UNAVAILABLE = (
    "Variational's trading API is not publicly available yet, so real orders cannot be "
    "sent. Use --broker paper (optionally priced from Variational) until it ships."
)


def _num(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


class VariationalPublicClient:
    """Read-only market data from Variational Omni."""

    def __init__(self, base_url: str = VARIATIONAL_API, http_get: HttpGet | None = None):
        self.base_url = base_url.rstrip("/")
        self.http_get = http_get or http_get_json

    def stats(self) -> dict:
        """GET /metadata/stats: platform totals plus per-listing statistics."""
        return self.http_get(f"{self.base_url}/metadata/stats", {})

    def listings(self) -> list[dict]:
        listings = self.stats().get("listings", [])
        if isinstance(listings, dict):  # tolerate {ticker: {...}} as well as [{...}]
            listings = [{"ticker": k, **v} for k, v in listings.items()]
        return listings

    def listing(self, ticker: str) -> dict:
        want = ticker.upper()
        for item in self.listings():
            if str(item.get("ticker", "")).upper() == want:
                return item
        raise KeyError(f"Variational has no listing for {ticker!r}")

    def mark_price(self, ticker: str) -> float:
        item = self.listing(ticker)
        for key in ("mark_price", "price", "index_price"):
            val = _num(item.get(key))
            if val:
                return val
        raise KeyError(f"no mark price in Variational listing for {ticker!r}: {sorted(item)}")

    def funding_rate(self, ticker: str) -> float | None:
        return _num(self.listing(ticker).get("funding_rate"))


class VariationalBroker(Broker):
    """Live Variational venue. Prices work today; orders wait for the trading API."""

    def __init__(self, client: VariationalPublicClient | None = None):
        self.client = client or VariationalPublicClient()

    def price(self, symbol: str) -> float:
        return self.client.mark_price(symbol)

    def equity(self) -> float:
        raise NotImplementedError(TRADING_API_UNAVAILABLE)

    def position(self, symbol: str) -> float:
        raise NotImplementedError(TRADING_API_UNAVAILABLE)

    def market_order(self, symbol: str, qty: float) -> Fill:
        raise NotImplementedError(TRADING_API_UNAVAILABLE)
