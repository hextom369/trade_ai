# trade_ai

機械学習でトレードシグナルを作り、**未来の情報を使わない（リークのない）検証**と、**コスト込みの現実的なバックテスト**で評価するためのフレームワークです。

> ⚠️ 研究・学習用です。投資助言ではありません。バックテストの成績は将来の利益を保証しません。

## 主な改善ポイント

よくある「トレードAI」がバックテストでは勝つのに実運用で負ける原因をつぶす設計にしています。

| よくある問題 | このリポジトリでの対策 |
|---|---|
| 学習データとテストデータをランダムに分割（未来で過去を予測） | **ウォークフォワード学習**：過去データだけで定期的に再学習し、次の区間だけを予測 (`model.py`) |
| 予測期間が重なるラベルによるリーク | 学習末尾と予測開始の間に `horizon` 本分の**パージ（隙間）**を入れる |
| シグナルが当日のリターンで計算されている | 足 t の終値で決めたポジションは **t+1 のリターン**にだけ適用 (`backtest.py`) |
| 手数料・スリッページを無視 | ポジション変化量に対して **手数料+スリッページ (bps)** を課金 |
| 価格そのものを特徴量にしている（非定常） | リターン・比率・Zスコアなど**スケールに依存しない特徴量** (`features.py`) |
| 確率に関係なく常にフルポジション | **確信度に応じたサイズ**＋ノートレード帯＋EMAで売買回数を抑制 |
| 相場のボラティリティでリスクが大きく変動 | **ボラティリティ・ターゲティング**（年率目標ボラに合わせてサイズ調整、上限レバレッジあり） |
| モデルが効かなくなっても売買を続ける | **品質ゲート**：直近の実際の的中率（ラベル確定後のみで計算）が悪ければポジションを自動で縮小 |
| 単一モデルへの過学習 | 勾配ブースティング＋正則化ロジスティック回帰の**アンサンブル** |

テストでは「将来のデータを書き換えても過去の特徴量・予測が変わらないこと」「ランダムウォークでは儲からないこと」（＝リークがないこと）を確認しています。

## インストール

```bash
pip install -e ".[dev]"          # 本体 + pytest
pip install -e ".[data]"         # yfinance で株価を取得する場合
```

## 使い方

```bash
# 合成データでデモ（ネット不要）
python -m trade_ai backtest --synthetic --horizon 1

# CSV（date, open, high, low, close, volume 列）でバックテスト
python -m trade_ai backtest --csv prices.csv --output result.csv

# yfinance から取得（例: トヨタ、買いのみ）
python -m trade_ai backtest --ticker 7203.T --start 2012-01-01 --long-only

# 次の足の目標ポジションを出力
python -m trade_ai signal --ticker SPY
```

`signal` の出力例:

```json
{
  "date": "2022-08-31",
  "close": 83.38,
  "proba_up": 0.36,
  "quality_gate": 1.0,
  "target_position": -0.54
}
```

`target_position` は資金に対する比率（+1 = 100% 買い、-0.5 = 50% 売り）です。

### 主なオプション

| オプション | 既定値 | 説明 |
|---|---|---|
| `--horizon` | 5 | 何本先のリターンを予測するか（パージ幅も兼ねる） |
| `--model` | ensemble | `ensemble` / `hgb` / `logreg` |
| `--min-train` / `--retrain-every` | 500 / 60 | 初回学習に必要な本数 / 再学習の間隔 |
| `--train-window` | なし | 指定すると直近 N 本のみで学習（なしは拡張ウィンドウ） |
| `--entry-band` / `--full-edge` | 0.04 / 0.15 | \|p−0.5\| がこれ未満なら取引しない / これ以上でフルサイズ |
| `--target-vol` / `--max-leverage` | 0.15 / 1.0 | 年率目標ボラ / ポジション上限 |
| `--long-only` | off | 売りポジションを取らない |
| `--no-gate` / `--gate-window` | on / 120 | 品質ゲートの無効化 / 的中率の計算期間 |
| `--cost-bps` / `--slippage-bps` | 5 / 2 | 片道コスト (bps) |
| `--periods-per-year` | 252 | 日足以外（例: 時間足）の年率換算に使用 |

## Python から使う

```python
from trade_ai import PipelineConfig, WalkForwardConfig, load_csv, run_pipeline

df = load_csv("prices.csv")
cfg = PipelineConfig(walk_forward=WalkForwardConfig(horizon=5, model="ensemble"))
res = run_pipeline(df, cfg)
print(res.backtest.stats)          # sharpe, max_drawdown, cagr など
res.backtest.equity.plot()
```

## 構成

```
trade_ai/
  data.py       CSV / yfinance / 合成データの読み込み
  features.py   特徴量（モメンタム, ボラ, RSI, MACD, ボリンジャー, ATR, 出来高, 自己相関レジーム…）とラベル
  model.py      モデルとパージ付きウォークフォワード学習
  strategy.py   確率→ポジション変換、ボラターゲット、品質ゲート
  backtest.py   コスト込みバックテスト
  metrics.py    シャープ, ソルティノ, 最大DD, カルマー, 回転率など
  pipeline.py   一連の処理と最新シグナル
  cli.py        コマンドライン
tests/          リーク検出を含むテスト
```

## テスト

```bash
pytest -q
```
