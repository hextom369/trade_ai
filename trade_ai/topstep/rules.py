"""Topstep account rules (Trading Combine / Express Funded style).

Defaults are the 50K Trading Combine: $3,000 profit target, $2,000 trailing
Maximum Loss Limit (trails the end-of-day balance high and stops trailing at
the starting balance), $1,000 Daily Loss Limit, 5 minis / 50 micros and a 50%
consistency target (best day < 50% of total profit). Topstep changes its rules
from time to time, so verify them and override the fields when they differ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, time


@dataclass
class TopstepRules:
    account_size: float = 50_000.0
    profit_target: float = 3_000.0
    max_loss_limit: float = 2_000.0
    daily_loss_limit: float | None = 1_000.0
    max_minis: float = 5.0
    consistency: float | None = 0.5
    # Positions must be flat before this time (3:10 PM CT = 4:10 PM ET).
    flat_by: time = time(16, 10)

    @classmethod
    def combine(cls, size: str = "50k") -> "TopstepRules":
        presets = {
            "50k": dict(account_size=50_000, profit_target=3_000, max_loss_limit=2_000,
                        daily_loss_limit=1_000, max_minis=5),
            "100k": dict(account_size=100_000, profit_target=6_000, max_loss_limit=3_000,
                         daily_loss_limit=2_000, max_minis=10),
            "150k": dict(account_size=150_000, profit_target=9_000, max_loss_limit=4_500,
                         daily_loss_limit=3_000, max_minis=15),
        }
        return cls(**presets[size.lower()])


@dataclass
class AccountState:
    """Balance bookkeeping plus the Topstep loss limits."""

    rules: TopstepRules
    balance: float = 0.0
    eod_high: float = 0.0
    day_start_balance: float = 0.0
    daily_pnl: dict[date, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.balance:
            self.balance = self.rules.account_size
        self.eod_high = max(self.eod_high, self.balance)
        self.day_start_balance = self.balance

    @property
    def mll_floor(self) -> float:
        """Account fails when equity (incl. unrealized) touches this level."""
        return min(self.eod_high - self.rules.max_loss_limit, self.rules.account_size)

    @property
    def day_pnl(self) -> float:
        return self.balance - self.day_start_balance

    @property
    def total_profit(self) -> float:
        return self.balance - self.rules.account_size

    def dll_level(self) -> float | None:
        if self.rules.daily_loss_limit is None:
            return None
        return self.day_start_balance - self.rules.daily_loss_limit

    def breach(self, equity: float) -> str | None:
        """Return 'MLL' or 'DLL' if ``equity`` violates a limit, else None."""
        if equity <= self.mll_floor:
            return "MLL"
        dll = self.dll_level()
        if dll is not None and equity <= dll:
            return "DLL"
        return None

    def apply_pnl(self, pnl: float) -> None:
        self.balance += pnl

    def end_of_day(self, day: date) -> None:
        self.daily_pnl[day] = self.daily_pnl.get(day, 0.0) + self.day_pnl
        self.eod_high = max(self.eod_high, self.balance)
        self.day_start_balance = self.balance

    def best_day_ratio(self) -> float:
        profit = self.total_profit
        if profit <= 0 or not self.daily_pnl:
            return float("nan")
        return max(self.daily_pnl.values()) / profit

    def consistency_ok(self) -> bool:
        if self.rules.consistency is None:
            return True
        ratio = self.best_day_ratio()
        return ratio == ratio and ratio < self.rules.consistency

    def target_reached(self) -> bool:
        return self.total_profit >= self.rules.profit_target and self.consistency_ok()
