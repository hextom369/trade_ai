"""Execution venues behind one small interface.

A strategy only produces a target position as a fraction of equity; the broker
turns that into orders. Quantities are signed base-asset units (+ long, - short)
of a linear, USD(C)-margined perpetual.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable


@dataclass
class Fill:
    symbol: str
    qty: float          # signed base quantity
    price: float        # average fill price
    fee: float          # fee paid, in quote currency
    time: str


class Broker(ABC):
    """Minimal venue interface used by the live runner."""

    @abstractmethod
    def equity(self) -> float:
        """Account value (collateral + unrealised PnL) in quote currency."""

    @abstractmethod
    def position(self, symbol: str) -> float:
        """Current signed position in base units."""

    @abstractmethod
    def price(self, symbol: str) -> float:
        """Reference (mark) price used for sizing."""

    @abstractmethod
    def market_order(self, symbol: str, qty: float) -> Fill:
        """Trade ``qty`` base units at market (+ buy, - sell)."""


@dataclass
class PaperState:
    cash: float
    positions: dict = field(default_factory=dict)   # symbol -> signed qty
    fills: list = field(default_factory=list)


class PaperBroker(Broker):
    """Simulated venue. Fills at the reference price plus slippage, charges fees.

    Prices come from ``price_source(symbol)`` (e.g. Variational's public mark price
    or the latest candle close) or are pushed with ``set_price``. With
    ``state_path`` the account survives restarts.
    """

    def __init__(self, initial_equity: float = 10_000.0, fee_bps: float = 0.0,
                 slippage_bps: float = 2.0, price_source: Callable[[str], float] | None = None,
                 state_path: str | None = None):
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self.price_source = price_source
        self.state_path = state_path
        self._prices: dict[str, float] = {}
        if state_path and os.path.exists(state_path):
            with open(state_path) as f:
                self.state = PaperState(**json.load(f))
        else:
            self.state = PaperState(cash=float(initial_equity))

    def set_price(self, symbol: str, price: float) -> None:
        self._prices[symbol] = float(price)

    def price(self, symbol: str) -> float:
        if self.price_source is not None:
            self._prices[symbol] = float(self.price_source(symbol))
        if symbol not in self._prices:
            raise KeyError(f"no price for {symbol}; call set_price or pass price_source")
        return self._prices[symbol]

    def position(self, symbol: str) -> float:
        return float(self.state.positions.get(symbol, 0.0))

    def equity(self) -> float:
        value = self.state.cash
        for sym, qty in self.state.positions.items():
            if qty:
                value += qty * (self._prices[sym] if sym in self._prices else self.price(sym))
        return float(value)

    def market_order(self, symbol: str, qty: float) -> Fill:
        ref = self.price(symbol)
        px = ref * (1 + (1 if qty > 0 else -1) * self.slippage_bps / 1e4)
        fee = abs(qty) * px * self.fee_bps / 1e4
        self.state.cash -= qty * px + fee
        new = self.position(symbol) + qty
        self.state.positions[symbol] = 0.0 if abs(new) < 1e-12 else new
        fill = Fill(symbol, float(qty), float(px), float(fee),
                    datetime.now(timezone.utc).isoformat(timespec="seconds"))
        self.state.fills.append(asdict(fill))
        self._save()
        return fill

    def apply_funding(self, symbol: str, rate: float) -> float:
        """Settle one funding payment (longs pay positive ``rate``). Returns the amount paid."""
        paid = self.position(symbol) * self.price(symbol) * rate
        self.state.cash -= paid
        self._save()
        return paid

    def _save(self) -> None:
        if self.state_path:
            tmp = f"{self.state_path}.tmp"
            with open(tmp, "w") as f:
                json.dump(asdict(self.state), f, indent=1)
            os.replace(tmp, self.state_path)
