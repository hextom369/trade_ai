import json

import numpy as np
import pandas as pd
import pytest

from trade_ai import cli
from trade_ai.backtest import run_backtest
from trade_ai.brokers import PaperBroker
from trade_ai.crypto_data import (funding_per_bar, load_binance_funding, load_binance_klines,
                                  periods_per_year)
from trade_ai.data import synthetic_ohlcv
from trade_ai.live import RiskLimits, Trader, TraderConfig
from trade_ai.pipeline import PipelineConfig, run_rule_pipeline
from trade_ai.features import build_features
from trade_ai.rules import RULES, RuleConfig, breakout, rule_signal, taker_flow, volume_breakout, vwap_obv
from trade_ai.variational import VariationalBroker, VariationalPublicClient

HOUR_MS = 3_600_000


def hourly(n=600, seed=2):
    df = synthetic_ohlcv(n=n, seed=seed, regime_strength=0.3)
    df.index = pd.date_range("2026-01-01", periods=n, freq="h", name="date")
    # Takers lean toward the bar's direction, like real order flow.
    share = 0.5 + 0.3 * np.tanh(np.log(df["close"] / df["open"]) * 50)
    df["taker_buy_volume"] = df["volume"] * share
    return df


@pytest.mark.parametrize("rule", RULES)
def test_rules_do_not_look_ahead(rule):
    df = hourly()
    cut = 400
    tampered = df.copy()
    tampered.iloc[cut + 1:] *= 1.5
    a = rule_signal(df, rule).iloc[:cut + 1]
    b = rule_signal(tampered, rule).iloc[:cut + 1]
    pd.testing.assert_series_equal(a, b)
    assert set(np.unique(rule_signal(df, rule))) <= {-1.0, 0.0, 1.0}


def test_breakout_enters_and_exits():
    close = [10.0] * 6 + [12.0, 12.5, 11.0, 9.0]
    df = pd.DataFrame({"close": close, "high": close, "low": close})
    sig = breakout(df, window=5, exit_window=2)
    assert list(sig) == [0, 0, 0, 0, 0, 0, 1, 1, 0, -1]


def test_volume_breakout_needs_heavy_volume():
    close = [10.0] * 6 + [12.0, 12.5]
    base = pd.DataFrame({"close": close, "high": close, "low": close})
    thin = base.assign(volume=[100.0] * 8)
    heavy = base.assign(volume=[100.0] * 6 + [300.0, 100.0])
    assert volume_breakout(thin, window=5, exit_window=2, vol_window=3).max() == 0
    assert list(volume_breakout(heavy, window=5, exit_window=2, vol_window=3)) == \
        [0, 0, 0, 0, 0, 0, 1, 1]


def test_vwap_obv_needs_price_and_volume_to_agree():
    n = 60
    close = pd.Series(np.linspace(100, 130, n))
    df = pd.DataFrame({"close": close, "high": close + 1, "low": close - 1,
                       "volume": 1000.0})
    assert vwap_obv(df, window=10).iloc[-1] == 1.0          # rising price, buying volume
    # Same price path, but heavy volume only on the (few) down bars -> OBV falls.
    choppy = close.copy()
    choppy.iloc[1::5] -= 1.5
    df2 = pd.DataFrame({"close": choppy, "high": choppy + 1, "low": choppy - 1})
    df2["volume"] = np.where(choppy.diff() < 0, 10_000.0, 100.0)
    assert vwap_obv(df2, window=10).iloc[-1] == 0.0


def test_taker_flow_follows_aggressive_buyers():
    df = hourly(200)
    df["taker_buy_volume"] = df["volume"] * 0.6
    assert (taker_flow(df).iloc[20:] == 1).all()
    df["taker_buy_volume"] = df["volume"] * 0.4
    assert (taker_flow(df).iloc[20:] == -1).all()
    with pytest.raises(ValueError):
        taker_flow(df.drop(columns="taker_buy_volume"))


def test_volume_features_do_not_look_ahead():
    df = hourly()
    cut = 400
    tampered = df.copy()
    tampered.iloc[cut + 1:] *= 2.0
    tampered.iloc[cut + 1:, tampered.columns.get_loc("taker_buy_volume")] *= 0.3
    a, b = build_features(df), build_features(tampered)
    assert {"obv_slope_10", "vwap_dist_20", "cmf_20", "mfi_14", "taker_buy_5"} <= set(a.columns)
    pd.testing.assert_frame_equal(a.iloc[:cut + 1], b.iloc[:cut + 1])
    base = build_features(df, volume_features=False)
    assert "obv_slope_10" not in base.columns and "ret_1" in base.columns


def test_rule_pipeline_runs_after_warmup():
    df = hourly()
    cfg = PipelineConfig()
    res = run_rule_pipeline(df, "ma_cross", RuleConfig(fast=10, slow=50), cfg)
    assert res.backtest.returns.index[0] == df.index[50]
    assert res.backtest.stats["bars"] == len(df) - 50


def test_funding_assigned_to_bar_and_paid_by_longs():
    idx = pd.date_range("2026-01-01", periods=10, freq="h")
    events = pd.Series([0.001, 0.002, 0.5],
                       index=pd.to_datetime(["2026-01-01 00:00", "2026-01-01 08:00",
                                             "2026-01-01 10:00"]))
    per_bar = funding_per_bar(events, idx, pd.Timedelta(hours=1))
    assert per_bar.iloc[0] == 0.001 and per_bar.iloc[8] == 0.002
    assert per_bar.sum() == pytest.approx(0.003)  # 10:00 is after the last bar ends
    close = pd.Series(100.0, index=idx)
    long = run_backtest(close, pd.Series(1.0, index=idx), 0, 0, funding=per_bar)
    short = run_backtest(close, pd.Series(-1.0, index=idx), 0, 0, funding=per_bar)
    assert long.returns.iloc[8] == pytest.approx(-0.002)
    assert short.returns.iloc[8] == pytest.approx(0.002)
    assert long.returns.iloc[0] == 0.0  # no position held yet at the first event


def test_periods_per_year():
    assert periods_per_year("1h") == 8760
    assert periods_per_year("15m") == 35040


def _kline(open_ms, price=100.0):
    return [open_ms, str(price), str(price + 1), str(price - 1), str(price), "5",
            open_ms + HOUR_MS - 1, "0", 0, "3", "0", "0"]


def test_binance_klines_paginate_and_drop_open_candle():
    start = 1_767_225_600_000  # 2026-01-01
    all_rows = [_kline(start + i * HOUR_MS, 100 + i) for i in range(7)]
    calls = []

    def fake_get(url, params):
        calls.append(params)
        rows = [r for r in all_rows if r[0] >= params["startTime"]]
        return rows[:params["limit"]]

    now = start + 6 * HOUR_MS + 10  # the 7th candle is still forming
    df = load_binance_klines("btcusdt", "1h", start="2026-01-01", limit=3,
                             http_get=fake_get, now_ms=now)
    assert len(df) == 6 and len(calls) == 3
    assert calls[0]["symbol"] == "BTCUSDT"
    assert df.index[0] == pd.Timestamp("2026-01-01") and df["close"].iloc[-1] == 105.0
    assert "taker_buy_volume" in df.columns


def test_binance_funding_parses():
    def fake_get(url, params):
        return [{"fundingTime": 1_767_225_600_000, "fundingRate": "0.0001"}]
    s = load_binance_funding("BTCUSDT", http_get=fake_get)
    assert s.iloc[0] == 0.0001 and s.index[0] == pd.Timestamp("2026-01-01")


def test_variational_client_and_broker():
    stats = {"total_volume_24h": "1", "listings": [
        {"ticker": "BTC", "mark_price": "65000.5", "funding_rate": "0.0001"},
        {"ticker": "ETH", "mark_price": "3000"}]}
    client = VariationalPublicClient(http_get=lambda url, params: stats)
    assert client.mark_price("btc") == 65000.5
    assert client.funding_rate("BTC") == 0.0001
    with pytest.raises(KeyError):
        client.mark_price("DOGE")
    broker = VariationalBroker(client)
    assert broker.price("ETH") == 3000.0
    with pytest.raises(NotImplementedError):
        broker.market_order("BTC", 0.1)


def test_paper_broker_accounting_and_persistence(tmp_path):
    path = str(tmp_path / "acct.json")
    b = PaperBroker(1000.0, fee_bps=10, slippage_bps=0, state_path=path)
    b.set_price("BTC", 100.0)
    b.market_order("BTC", 2.0)                 # notional 200, fee 0.2
    assert b.equity() == pytest.approx(999.8)
    b.set_price("BTC", 110.0)
    assert b.equity() == pytest.approx(1019.8)
    b.apply_funding("BTC", 0.001)             # long pays 2 * 110 * 0.001
    again = PaperBroker(state_path=path)
    again.set_price("BTC", 110.0)
    assert again.position("BTC") == 2.0
    assert again.equity() == pytest.approx(1019.58)


def _trader(tmp_path, broker, df, **risk):
    now = df.index[-1] + pd.Timedelta(hours=1, minutes=1)
    cfg = TraderConfig("BTCUSDT", "BTC", "1h", strategy="ma_cross",
                       rules=RuleConfig(fast=5, slow=20), risk=RiskLimits(**risk),
                       dry_run=False, state_dir=str(tmp_path))
    return Trader(broker, cfg, now=lambda: now)


def test_trader_moves_to_target_and_logs(tmp_path):
    df = hourly()
    broker = PaperBroker(10_000, slippage_bps=0)
    broker.set_price("BTC", float(df["close"].iloc[-1]))
    t = _trader(tmp_path, broker, df, max_abs_position=0.5)
    d = t.step(df)
    assert d["action"] == "order"
    assert abs(d["target_fraction"]) <= 0.5
    assert broker.position("BTC") == pytest.approx(d["target_qty"])
    assert t.step(df)["action"] == "hold"   # already at target
    lines = (tmp_path / "decisions.jsonl").read_text().splitlines()
    assert [json.loads(x)["action"] for x in lines] == ["order", "hold"]


def test_trader_skips_stale_data_and_dry_run(tmp_path):
    df = hourly()
    broker = PaperBroker(10_000)
    broker.set_price("BTC", 100.0)
    t = _trader(tmp_path, broker, df)
    assert t.step(df.iloc[:-3])["action"] == "skip"
    t.cfg.dry_run = True
    d = t.step(df)
    assert d["action"] in ("dry_run", "hold") and broker.position("BTC") == 0


def test_kill_switch_flattens_after_daily_loss(tmp_path):
    df = hourly()
    broker = PaperBroker(10_000, slippage_bps=0)
    broker.set_price("BTC", 100.0)
    t = _trader(tmp_path, broker, df, max_daily_loss=0.05, rebalance_band=0.0)
    broker.market_order("BTC", 50.0)          # 50% long
    t.risk_state = {"day": str(t.now().date()), "start_equity": 10_000, "halted": False}
    broker.set_price("BTC", 80.0)             # equity 9,000 -> -10% on the day
    d = t.step(df)
    assert d["halted"] and d["target_qty"] == 0 and broker.position("BTC") == 0
    assert t.step(df)["action"] == "hold"     # stays flat for the rest of the day


def test_cli_trade_once_paper(tmp_path, monkeypatch, capsys):
    df = hourly()
    df.index = pd.date_range(end=pd.Timestamp.now("UTC").tz_localize(None).floor("h") - pd.Timedelta(hours=1),
                             periods=len(df), freq="h", name="date")
    monkeypatch.setattr(cli, "load_binance_klines", lambda *a, **k: df)
    state = tmp_path / "state"
    assert cli.main(["trade", "--binance", "BTCUSDT", "--strategy", "breakout", "--once",
                     "--execute", "--state-dir", str(state)]) == 0
    rec = json.loads((state / "decisions.jsonl").read_text().splitlines()[-1])
    assert rec["action"] in ("order", "hold") and rec["strategy"] == "breakout"
    assert (state / "paper_account.json").exists() or rec["action"] == "hold"
