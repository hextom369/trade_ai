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
| `--periods-per-year` | 252（`--binance` は自動） | 日足以外（例: 時間足）の年率換算に使用 |

## 暗号資産パーペチュアル / Variational での自動売買

AI（`--strategy ml`）とルールベース戦略を同じ枠組みでバックテスト→ペーパー取引→実運用できます。

> ⚠️ **Variational の現状（2026-10 確認）**：公開 API は読み取り専用（`/metadata/stats` のマーク価格・ファンディング等）で、
> **注文用のトレード API はまだ一般公開されていません**。そのため現在は「Variational のマーク価格でのペーパー取引」まで対応し、
> 実注文は `trade_ai/variational.py` の `VariationalBroker` が明示的にエラーにします。API 公開後はこのクラスの
> `equity` / `position` / `market_order` を実装するだけで、戦略・リスク管理・ループはそのまま使えます。

### 戦略

| `--strategy` | 内容 |
|---|---|
| `ml` | 既存の ML（ウォークフォワード学習のアンサンブル） |
| `ma_cross` | 短期MA > 長期MA で買い、逆で売り（`--fast` / `--slow`） |
| `breakout` | ドンチャン・ブレイクアウト（`--breakout-window` で新値エントリー、`--exit-window` で手仕舞い） |
| `rsi_reversion` | RSI 逆張り（`--rsi-low` 以下で買い / `--rsi-high` 以上で売り、50 で手仕舞い） |

| `volume_breakout` | **出来高を伴うブレイクアウトだけ**エントリー（ブレイク足の出来高が直前 `--vol-window` 本平均の `--vol-mult` 倍以上） |
| `vwap_obv` | **価格と出来高が一致した時だけ**取引：VWAP より上かつ OBV（出来高の買い越し）が増加でロング、逆でショート、食い違えばノーポジ |
| `taker_flow` | **テイカー買い比率**（成行買い ÷ 出来高）の EMA が 0.5±`--taker-band` を超えた方向に追随。`--binance` データが必要 |

どの戦略も同じサイジング（ボラ・ターゲティング、上限レバレッジ、`--long-only`、平滑化）を通ります。

### 出来高の活用

- **ML の特徴量**に出来高系を追加しています：相対出来高、OBV の傾き、VWAP 乖離、CMF（チャイキン・マネーフロー）、MFI、価格変化と出来高変化の相関、出来高で重み付けしたリターン、テイカー買い比率（1/5/20 本）。
- 出来高は Binance 先物の出来高（市場全体の流動性）を使います。Variational は RFQ 方式なので足ごとの自前の出来高はありません。
- **出来高が本当に効いているか**は銘柄・時間足で変わるので、必ず比較してください：

```bash
python -m trade_ai backtest --binance BTCUSDT --interval 1h --start 2024-01-01 --strategy ml --horizon 3
python -m trade_ai backtest --binance BTCUSDT --interval 1h --start 2024-01-01 --strategy ml --horizon 3 --no-volume-features
```

  （合成データでは出来高がノイズなので、出来高特徴量を外した方が成績が良くなります。効かない特徴量はむしろ害になる例です。）

### データとバックテスト

価格データは Variational が参照する大手取引所のうち、Binance USDⓈ-M 先物の公開ローソク足を使います（API キー不要）。
バックテストでは**ファンディング（ロングが支払い、ショートが受け取り）**も損益に反映し、年率換算は 24 時間 365 日で自動計算します。

```bash
python -m trade_ai backtest --binance BTCUSDT --interval 1h --start 2024-01-01 --strategy breakout
python -m trade_ai backtest --binance ETHUSDT --interval 4h --start 2023-01-01 --strategy ml --horizon 3
python -m trade_ai signal   --binance BTCUSDT --interval 1h --strategy ma_cross   # 次の足の目標ポジション
```

Variational は取引手数料ゼロですが RFQ のスプレッドがかかるため、`--cost-bps 0 --slippage-bps 3` のようにスリッページ側で見積もってください。

### ペーパー取引 / 実運用ループ

```bash
# 1回だけ判断して記録（注文はしない＝ドライラン）
python -m trade_ai trade --binance BTCUSDT --strategy ma_cross --once

# ペーパー口座で継続運用（足が確定するたびに判断・約定）。価格は Variational のマーク価格
python -m trade_ai trade --binance BTCUSDT --venue-symbol BTC --strategy breakout \
    --execute --paper-prices variational --paper-equity 10000 --cost-bps 0 --slippage-bps 3
```

- 判断はすべて `trade_state/decisions.jsonl` に、ペーパー口座は `trade_state/paper_account.json` に保存（再起動しても継続）
- `--execute` を付けない限り注文しません（ドライラン）

安全装置:

| オプション | 既定値 | 内容 |
|---|---|---|
| `--max-position` | 1.0 | 目標ポジションの絶対上限（資金比） |
| `--max-notional` | なし | 建玉の名目金額の上限 |
| `--max-daily-loss` | 0.05 | UTC の1日で資金が 5% 減ったら全決済し、その日は停止 |
| `--rebalance-band` / `--min-trade-notional` | 0.05 / 10 | 小さな調整は発注しない（無駄な売買を防止） |
| （自動） | | 最新の確定足が古すぎる場合は取引しない／通信エラーは記録して次の足で再試行、5 連続で停止 |

Python からは `trade_ai.live.Trader` に任意の `Broker`（`trade_ai.brokers.Broker` を継承）を渡せます。
Variational 以外の API がある DEX/CEX も同じ形でアダプタを追加すれば動きます。

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
  pipeline.py   一連の処理と最新シグナル（ML / ルール共通）
  rules.py      ルールベース戦略（MAクロス, ブレイクアウト, RSI逆張り）
  crypto_data.py Binance 先物のローソク足・ファンディング取得
  brokers.py    取引所インターフェースとペーパー取引
  variational.py Variational Omni アダプタ（公開データ／トレードAPI待ち）
  live.py       リスク管理付きの売買ループ
  cli.py        コマンドライン
tests/          リーク検出を含むテスト
```

## テスト

```bash
pytest -q
```
