import json

import pandas as pd
import pytest

from trade_ai import cli
from trade_ai.brokers import PaperBroker
from trade_ai.live import Trader, TraderConfig
from trade_ai.pipeline import PipelineConfig
from trade_ai.portfolio import run_portfolio_backtest
from trade_ai.rules import RuleConfig
from trade_ai.universe import (binance_24h_quote_volume, point_in_time_universe, prefilter,
                               rank_universe, venue_ticker, volume_scores)
from trade_ai.data import synthetic_ohlcv


def market(n=400, seed=0, volume_scale=1.0, end=None):
    df = synthetic_ohlcv(n=n, seed=seed, regime_strength=0.3)
    if end is None:
        df.index = pd.date_range("2026-01-01", periods=n, freq="h", name="date")
    else:
        df.index = pd.date_range(end=end, periods=n, freq="h", name="date")
    df["volume"] *= volume_scale
    df["taker_buy_volume"] = df["volume"] * 0.5
    return df


def test_venue_ticker():
    assert venue_ticker("BTCUSDT") == "BTC"
    assert venue_ticker("1000PEPEUSDT") == "PEPE"
    assert venue_ticker("1INCHUSDT") == "1INCH"


def test_binance_24h_and_prefilter():
    rows = [{"symbol": "BTCUSDT", "quoteVolume": "9e9"},
            {"symbol": "USDCUSDT", "quoteVolume": "8e9"},
            {"symbol": "ETHUSDT", "quoteVolume": "5e9"},
            {"symbol": "BTCUSDT_260327", "quoteVolume": "7e9"},   # dated future
            {"symbol": "ETHBTC", "quoteVolume": "1e9"},
            {"symbol": "TINYUSDT", "quoteVolume": "1e5"},
            {"symbol": "1000PEPEUSDT", "quoteVolume": "1e9"}]
    vol = binance_24h_quote_volume(http_get=lambda url, params: rows)
    assert list(vol.index) == ["BTCUSDT", "USDCUSDT", "ETHUSDT", "1000PEPEUSDT", "TINYUSDT"]
    assert prefilter(vol, 10, min_quote_volume=1e6) == ["BTCUSDT", "ETHUSDT", "1000PEPEUSDT"]
    assert prefilter(vol, 1) == ["BTCUSDT"]
    assert prefilter(vol, 10, allowed={"ETH", "PEPE"}) == ["ETHUSDT", "1000PEPEUSDT"]


def test_ranks_by_traded_value_and_surge():
    frames = {"BIG": market(seed=1, volume_scale=10), "MID": market(seed=2, volume_scale=3),
              "SMALL": market(seed=3)}
    frames["SMALL"].iloc[-24:, frames["SMALL"].columns.get_loc("volume")] *= 4  # volume spike
    liq, surge = volume_scores(frames, window=100, surge_window=24)
    assert rank_universe(liq.iloc[-1], surge.iloc[-1], 2) == ["BIG", "MID"]
    assert rank_universe(liq.iloc[-1], surge.iloc[-1], 1, "surge") == ["SMALL"]
    # The liquidity floor removes thin markets even when they surge.
    floor = liq.iloc[-1]["MID"]
    assert rank_universe(liq.iloc[-1], surge.iloc[-1], 1, "surge", floor) != ["SMALL"]


def test_point_in_time_universe_ignores_future_volume():
    frames = {"A": market(seed=1, volume_scale=5), "B": market(seed=2)}
    base = point_in_time_universe(frames, top_n=1, window=50, rebalance_every=10)
    assert base.iloc[:49].sum().sum() == 0            # nothing before the window fills
    assert base["A"].iloc[60:].all() and not base["B"].any()
    cut = 250
    tampered = {k: v.copy() for k, v in frames.items()}
    tampered["B"].iloc[cut + 1:, tampered["B"].columns.get_loc("volume")] *= 100
    alt = point_in_time_universe(tampered, top_n=1, window=50, rebalance_every=10)
    pd.testing.assert_frame_equal(base.iloc[:cut + 1], alt.iloc[:cut + 1])
    assert alt["B"].iloc[-1]                          # B takes over once its volume is known


def test_portfolio_only_trades_selected_symbols_with_equal_slices():
    frames = {"A": market(seed=1, volume_scale=5), "B": market(seed=2, volume_scale=4),
              "C": market(seed=3)}
    members = point_in_time_universe(frames, top_n=2, window=50, rebalance_every=24)
    cfg = PipelineConfig()
    cfg.strategy.max_leverage = 1.0
    res = run_portfolio_backtest(frames, members, 2, "ma_cross", cfg, RuleConfig(fast=5, slow=20))
    assert "C" not in res.positions or (res.positions["C"] == 0).all()
    assert res.positions.abs().max().max() <= 0.5 + 1e-12
    assert res.backtest.positions.max() <= 1.0 + 1e-12
    assert res.backtest.returns.index[0] > members.index[50]


def test_step_universe_weights_and_exits(tmp_path):
    end = pd.Timestamp.now("UTC").tz_localize(None).floor("h") - pd.Timedelta(hours=1)
    frames = {"A": market(seed=1, end=end), "B": market(seed=2, end=end)}
    broker = PaperBroker(10_000, slippage_bps=0)
    for s in ("A", "B", "OLD"):
        broker.set_price(s, 100.0)
    broker.market_order("OLD", 10.0)
    cfg = TraderConfig("", "", "1h", strategy="ma_cross", rules=RuleConfig(fast=5, slow=20),
                       dry_run=False, state_dir=str(tmp_path), top_n=4)
    t = Trader(broker, cfg)
    t.risk_state["universe"] = ["OLD", "A"]
    out = t.step_universe(frames)
    assert broker.position("OLD") == 0 and out[0]["reason"] == "left universe"
    assert all(d["weight"] == 0.25 for d in out[1:])
    assert all(abs(d["target_fraction"]) <= 0.25 for d in out[1:])
    assert t.risk_state["universe"] == ["A", "B"]


@pytest.fixture
def fake_binance(monkeypatch):
    end = pd.Timestamp.now("UTC").tz_localize(None).floor("h") - pd.Timedelta(hours=1)
    data = {"BTCUSDT": market(seed=1, volume_scale=10, end=end),
            "ETHUSDT": market(seed=2, volume_scale=5, end=end),
            "DOGEUSDT": market(seed=3, end=end)}
    monkeypatch.setattr(cli, "load_binance_klines", lambda sym, *a, **k: data[sym])
    monkeypatch.setattr(cli, "binance_24h_quote_volume",
                        lambda: pd.Series({"BTCUSDT": 9e9, "ETHUSDT": 5e9, "DOGEUSDT": 1e9}))
    monkeypatch.setattr(cli, "load_binance_funding",
                        lambda *a, **k: pd.Series(dtype=float, name="funding"))
    return data


def test_cli_screen_backtest_signal_trade(fake_binance, tmp_path, capsys):
    common = ["--top-n", "2", "--volume-window", "48", "--rebalance-every", "24",
              "--min-quote-volume", "0"]
    assert cli.main(["screen", *common]) == 0
    assert "selected: BTCUSDT, ETHUSDT" in capsys.readouterr().out
    assert cli.main(["backtest", "--universe", "--strategy", "breakout", *common]) == 0
    out = capsys.readouterr().out
    assert "Universe: top 2 of 3" in out and "BTCUSDT" in out
    assert cli.main(["signal", "--universe", "--strategy", "vwap_obv", *common]) == 0
    sig = json.loads(capsys.readouterr().out)
    assert [s["symbol"] for s in sig] == ["BTCUSDT", "ETHUSDT"]
    state = tmp_path / "st"
    assert cli.main(["trade", "--universe", "--strategy", "ma_cross", "--once", "--execute",
                     "--state-dir", str(state), *common]) == 0
    sel = json.loads((state / "universe_selection.json").read_text())
    assert sel["symbols"] == ["BTCUSDT", "ETHUSDT"]
    recs = [json.loads(x) for x in (state / "decisions.jsonl").read_text().splitlines()]
    assert {r["symbol"] for r in recs} == {"BTC", "ETH"}
