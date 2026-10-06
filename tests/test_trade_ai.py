import numpy as np
import pandas as pd
import pytest

from trade_ai import (
    PipelineConfig, StrategyConfig, WalkForwardConfig, build_features, make_labels,
    positions_from_proba, run_backtest, run_pipeline, synthetic_ohlcv,
)
from trade_ai.cli import main
from trade_ai.data import load_csv
from trade_ai.model import walk_forward_predict
from trade_ai.strategy import quality_gate


@pytest.fixture(scope="module")
def df():
    return synthetic_ohlcv(n=900, seed=1, regime_strength=0.3)


def fast_cfg(**kw):
    return PipelineConfig(walk_forward=WalkForwardConfig(horizon=1, min_train=300,
                                                         retrain_every=100, model="logreg"), **kw)


def test_features_do_not_look_ahead(df):
    cut = 600
    full = build_features(df)
    # Corrupt everything after the cut; features up to the cut must not change.
    tampered = df.copy()
    tampered.iloc[cut + 1:] *= 3.0
    part = build_features(tampered)
    pd.testing.assert_frame_equal(full.iloc[:cut + 1], part.iloc[:cut + 1])


def test_labels_are_forward_and_nan_at_end(df):
    lab = make_labels(df, horizon=5)
    assert lab.iloc[-5:].isna().all()
    expected = float(df["close"].iloc[10 + 5] > df["close"].iloc[10])
    assert lab.iloc[10] == expected


def test_walk_forward_predictions_ignore_future_labels(df):
    feats = build_features(df)
    labels = make_labels(df, horizon=1)
    cfg = WalkForwardConfig(horizon=1, min_train=300, retrain_every=100, model="logreg")
    base = walk_forward_predict(feats, labels, cfg)
    # Flipping labels from row 500 on must not change predictions made before
    # any model could have been trained on them (first refit at/after 500 + horizon).
    flipped = labels.copy()
    flipped.iloc[500:] = 1 - flipped.iloc[500:]
    alt = walk_forward_predict(feats, flipped, cfg)
    # Blocks start at 301, 401, 501, 601. Block 501 trains on rows < 500
    # (purged), so the first prediction that may change is row 601.
    np.testing.assert_allclose(base.iloc[:601].dropna(), alt.iloc[:601].dropna())
    assert not np.allclose(base.iloc[601:], alt.iloc[601:])
    assert base.iloc[:301].isna().all()


def test_backtest_uses_previous_position_and_charges_costs():
    idx = pd.bdate_range("2020-01-01", periods=4)
    close = pd.Series([100.0, 110.0, 99.0, 99.0], index=idx)
    pos = pd.Series([1.0, 1.0, 0.0, 0.0], index=idx)
    res = run_backtest(close, pos, cost_bps=10, slippage_bps=0)
    # bar0: no position. bar1: entered at close0 -> +10% minus entry cost.
    assert res.returns.iloc[0] == 0.0
    assert res.returns.iloc[1] == pytest.approx(0.10 - 0.001)
    assert res.returns.iloc[2] == pytest.approx(-0.10)
    assert res.returns.iloc[3] == pytest.approx(-0.001)  # exit cost


def test_positions_respect_band_leverage_and_long_only(df):
    proba = pd.Series(np.linspace(0, 1, len(df)), index=df.index)
    cfg = StrategyConfig(target_vol=0, smoothing=1, entry_band=0.05, max_leverage=0.5)
    pos = positions_from_proba(proba, df["close"], cfg)
    assert pos.abs().max() <= 0.5 + 1e-12
    assert (pos[(proba - 0.5).abs() < 0.05] == 0).all()
    lo = positions_from_proba(proba, df["close"], StrategyConfig(long_only=True))
    assert (lo >= 0).all()


def test_quality_gate_waits_for_labels_to_resolve():
    idx = pd.RangeIndex(300)
    proba = pd.Series(0.9, index=idx)
    labels = pd.Series(1.0, index=idx)
    g = quality_gate(proba, labels, horizon=5, window=40)
    assert (g.iloc[:5 + 20 - 1] == 0).all()  # needs horizon shift + min_periods
    assert g.iloc[-1] == 1.0
    bad = quality_gate(proba, 1 - labels, horizon=5, window=40)
    assert (bad == 0).all()


def test_pipeline_finds_edge_in_predictable_series():
    df = synthetic_ohlcv(n=1200, seed=3, regime_strength=0.4)
    res = run_pipeline(df, fast_cfg())
    stats = res.backtest.stats
    assert stats["bars"] > 0
    assert stats["sharpe"] > 0.5
    assert res.backtest.returns.index[0] == res.proba.first_valid_index()


def test_pipeline_does_not_profit_on_random_walk():
    df = synthetic_ohlcv(n=1200, seed=3, regime_strength=0.0, drift=0.0)
    stats = run_pipeline(df, fast_cfg()).backtest.stats
    assert stats["sharpe"] < 1.0  # no edge to find; a high Sharpe would signal leakage


def test_cli_roundtrip(tmp_path, df, capsys):
    path = tmp_path / "prices.csv"
    df.reset_index().rename(columns={"date": "Date", "close": "Close"}).to_csv(path, index=False)
    assert len(load_csv(str(path))) == len(df)
    out = tmp_path / "out.csv"
    args = ["--csv", str(path), "--model", "logreg", "--min-train", "300",
            "--retrain-every", "100", "--horizon", "1"]
    assert main(["backtest", *args, "--output", str(out)]) == 0
    assert out.exists()
    assert main(["signal", *args]) == 0
    assert "target_position" in capsys.readouterr().out
