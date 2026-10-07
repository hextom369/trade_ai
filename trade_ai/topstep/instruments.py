"""Futures contract specifications.

Fees are approximate per-side (one fill) commission + exchange fees on
TopstepX and are deliberately on the conservative side. Check the current
fee schedule and override ``fee_per_side`` if needed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Instrument:
    symbol: str
    tick_size: float
    tick_value: float          # USD per tick per contract
    fee_per_side: float        # USD per contract per fill
    micro: bool = False

    @property
    def point_value(self) -> float:
        return self.tick_value / self.tick_size

    @property
    def mini_equivalent(self) -> float:
        """Contract weight for Topstep's position limit (10 micros = 1 mini)."""
        return 0.1 if self.micro else 1.0

    def round_price(self, price: float) -> float:
        return round(round(price / self.tick_size) * self.tick_size, 10)

    def with_fee(self, fee_per_side: float) -> "Instrument":
        return replace(self, fee_per_side=fee_per_side)


INSTRUMENTS: dict[str, Instrument] = {
    "MES": Instrument("MES", 0.25, 1.25, 0.50, micro=True),
    "MNQ": Instrument("MNQ", 0.25, 0.50, 0.50, micro=True),
    "MYM": Instrument("MYM", 1.0, 0.50, 0.50, micro=True),
    "M2K": Instrument("M2K", 0.10, 0.50, 0.50, micro=True),
    "MGC": Instrument("MGC", 0.10, 1.00, 0.70, micro=True),
    "MCL": Instrument("MCL", 0.01, 1.00, 0.70, micro=True),
    "ES": Instrument("ES", 0.25, 12.50, 2.00),
    "NQ": Instrument("NQ", 0.25, 5.00, 2.00),
    "YM": Instrument("YM", 1.0, 5.00, 2.00),
    "RTY": Instrument("RTY", 0.10, 5.00, 2.00),
    "GC": Instrument("GC", 0.10, 10.00, 2.20),
    "CL": Instrument("CL", 0.01, 10.00, 2.20),
}


def get_instrument(symbol: str) -> Instrument:
    try:
        return INSTRUMENTS[symbol.upper()]
    except KeyError:
        raise ValueError(f"unknown instrument {symbol!r}; known: {sorted(INSTRUMENTS)}") from None
