"""trade_ai: ML trading signal research with walk-forward validation."""

from .backtest import BacktestResult, run_backtest
from .data import load_csv, load_yfinance, synthetic_ohlcv
from .features import build_features, make_labels
from .model import WalkForwardConfig, walk_forward_predict
from .pipeline import (PipelineConfig, latest_signal, latest_target, run_pipeline,
                       run_rule_pipeline)
from .rules import RULES, RuleConfig, rule_positions, rule_signal
from .strategy import StrategyConfig, positions_from_proba, size_positions

__all__ = [
    "RULES", "BacktestResult", "PipelineConfig", "RuleConfig", "StrategyConfig",
    "WalkForwardConfig", "build_features", "latest_signal", "latest_target", "load_csv",
    "load_yfinance", "make_labels", "positions_from_proba", "rule_positions", "rule_signal",
    "run_backtest", "run_pipeline", "run_rule_pipeline", "size_positions", "synthetic_ohlcv",
    "walk_forward_predict",
]
__version__ = "0.1.0"
