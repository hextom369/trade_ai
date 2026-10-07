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

---

# Topstep 50K 用 先物Bot (`trade_ai.topstep`)

Topstep 50K 口座で **月 +$1,000 前後を目標**に、ルール違反（MLL / DLL）を絶対に起こさないことを最優先にした日中先物Botです。
バックテストとライブ（TopstepX / ProjectX API）で**同じ戦略コード・同じリスク管理コード**を使います。

> ⚠️ **利益は保証できません。** 月$1,000を「コンスタントに」稼げるかは、実データでの検証結果次第です。
> 同梱の合成データでの成績は動作確認用で、実際の相場について何も意味しません。必ず実データでバックテスト →
> TopstepX の練習口座でドライラン/ペーパー運用 → Combine の順に進めてください。

## 目標から逆算した設計

| 項目 | 既定値 | 理由 |
|---|---|---|
| 月間目標 | $1,000 (`--monthly-target`) | 約21営業日 → 1日平均 ≈ $50 |
| 1トレードのリスク | $150 (`--risk-per-trade`) | 手数料・スリッページ込み。1日の損失上限の半分 |
| 1日の損失ストップ | $300 (`--daily-loss-stop`) | Topstep DLL $1,000 の 30%。3日連続で負けても MLL まで余裕 |
| 1日の利益確定 | $300 (`--daily-profit-target`) | 1日の大勝ちを防ぎ **50% コンシステンシー**を満たしやすく |
| 1日の最大トレード数 | 2 | オーバートレード防止 |
| 月間目標達成後 | リスク半分 (`--after-target reduce` / `stop`) | 達成した利益を守る |
| MLL バッファ | $300 | 残り余力が少なくなったら自動停止（キルスイッチ） |
| 最大枚数 | 50 micro / 5 mini | Topstep 50K の上限 |

計算例：$150 リスクで平均 +0.33R の期待値なら 1トレード ≈ +$50、月20トレードで ≈ +$1,000。
**つまり月$1,000は「勝率」ではなく「期待値（R）× 回数」で決まります。** バックテストの `avg_r` と `trades` を見てください。

## 戦略：オープニングレンジ・ブレイクアウト (ORB)

- 9:30–9:45 ET の高値・安値をレンジとする（`--or-minutes`）
- 過去14日の平均日足レンジに対してレンジが狭すぎ/広すぎる日は見送り（`--min-range-adr` / `--max-range-adr`）
- 1分足終値がレンジ上抜け＋VWAP より上 → 買い（下は売り）。エントリーは 11:30 ET まで
- 損切りはレンジ中央（`--stop-mode mid`）または反対側、利確は 1.5R（`--target-r`）、1R で建値ストップ
- 15:50 ET に強制決済（Topstep の 3:10 PM CT クローズ前）

## Topstep ルールの再現

`rules.py` は 50K Combine 既定値：利益目標 $3,000、**MLL $2,000（日末残高の最高値にトレール、開始残高でロック）**、
**DLL $1,000**、50% コンシステンシー、最大 5 mini。バックテストは**含み損も含めてバー内で**判定し、
MLL に触れたら「口座リセット」として回数を集計します。Topstep のルールは変更されることがあるので、最新の規約を確認して
`TopstepRules` や `--no-dll` で合わせてください。手数料（`instruments.py`）も概算なので `--fee` で調整してください。

## 使い方

```bash
pip install -e ".[dev]"

# 1) 動作確認（合成データ）
python -m trade_ai topstep backtest --synthetic-trend 1

# 2) 実データを取得（TopstepX の API キーが必要。ProjectX API サブスクリプション）
export PROJECTX_USERNAME=あなたのユーザー名
export PROJECTX_API_KEY=あなたのAPIキー
python -m trade_ai topstep download --symbol MES --days 365 --output mes_1m.csv

#    他社データ（Databento, FirstRate 等）の CSV も可。タイムスタンプがUTCでなければ --source-tz を指定
# 3) 実データでバックテスト（月別損益・月$1,000達成率・MLL違反・モンテカルロ）
python -m trade_ai topstep backtest --csv mes_1m.csv --symbol MES --trades-out trades.csv

# 4) ドライラン（注文は送らずログだけ）。TopstepX に表示されている現在の MLL を渡すと安全
python -m trade_ai topstep live --symbol MES --account-name 練習口座名 --mll-floor 48000

# 5) 練習口座で実注文 → 問題なければ Combine 口座で
python -m trade_ai topstep live --symbol MES --account-name 口座名 --mll-floor 48000 --live
```

### バックテスト出力の見方

- `months_>=_1000` … 月 $1,000 以上を達成した月の割合（これがあなたの目標の直接の指標）
- `account_resets (MLL breaches)` … 0 であるべき
- `avg_r` / `profit_factor` … 0 / 1.0 を明確に上回らないなら、その設定では稼げません
- Monte Carlo … 日次損益を再サンプルし、新しい口座で 21 営業日運用したときの「月$1,000達成確率」「MLL 抵触確率」

**過剰最適化に注意：** パラメータを調整するときは、期間の前半で決めて後半（`--start`）で確認してください。
後半でも `avg_r > 0` が維持されない設定は使わないでください。

### ライブBotの仕組み (`live.py`)

1分足をポーリング → 確定足だけを戦略に渡す → シグナルが出たら成行エントリー → 逆指値（損切り）と指値（利確）を発注 →
ポジションが無くなったら残りの注文を取消（OCO 代替）→ 15:50 ET または日次損失ストップで全決済。
月間損益と日末残高の最高値は `topstep_state.json` に保存され、再起動しても引き継がれます。
停電・回線断に備え、損切り注文は常にサーバー側に置かれます。Bot 停止中はポジションを持ち越さないよう注意してください。

```
trade_ai/topstep/
  rules.py        Topstep の MLL(トレール) / DLL / 目標 / コンシステンシー
  risk.py         枚数計算・日次/月次の損益ストップ・キルスイッチ
  strategy.py     ORB 戦略（1本ずつ処理、未来参照なし）
  backtest.py     バー内の損切り優先・スリッページ・手数料込みのバックテスト
  report.py       月別集計・モンテカルロ
  projectx.py     TopstepX (ProjectX Gateway API) クライアント
  live.py         ライブ/ドライラン Bot
  cli.py          python -m trade_ai topstep ...
```
