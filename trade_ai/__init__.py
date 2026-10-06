"""trade_ai: ML trading signal research with walk-forward validation."""

from .backtest import BacktestResult, run_backtest
from .data import load_csv, load_yfinance, synthetic_ohlcv
from .features import build_features, make_labels
from .model import WalkForwardConfig, walk_forward_predict
from .pipeline import PipelineConfig, latest_signal, run_pipeline
from .strategy import StrategyConfig, positions_from_proba

__all__ = [
    "BacktestResult", "PipelineConfig", "StrategyConfig", "WalkForwardConfig",
    "build_features", "latest_signal", "load_csv", "load_yfinance", "make_labels",
    "positions_from_proba", "run_backtest", "run_pipeline", "synthetic_ohlcv",
    "walk_forward_predict",
]
__version__ = "0.1.0"
