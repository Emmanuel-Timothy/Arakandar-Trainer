# Arakandar Model Guidelines

## Model roles

Arakandar has two different model layers:

1. **Signal model:** LightGBM predicts `BUY`, `HOLD`, or `SELL` from market features. It is authoritative for the deterministic signal and probabilities.
2. **Analyst model:** the local Qwen-based Arakandar 3B model, optionally loaded with `models/adapters/arakandar-3b-lora`, explains the supplied market, signal, news, workflow, and memory evidence.

The analyst model must never invent prices, news, probabilities, trades, or data freshness. It must not change the deterministic signal.

## Current model chain

The chat runtime selects the first available path:

1. `models/adapters/arakandar-3b-lora`
2. `models/base/arakandar-3b`
3. `models/base/arakandar-1.5b`

The adapter is a LoRA specialization on top of the 3B base. It does not replace or modify the base weights.

## Data requirements

Use real, validated OHLCV data with:

- `date`
- `open`
- `high`
- `low`
- `close`
- `volume`

Record source and freshness in `data/raw/market_data_manifest.json`. Yahoo data can contain zero-volume or malformed rows; the fetcher quarantines them and records the count. Do not treat a downloaded file as trustworthy without checking the manifest.

## Signal-model training

Train candidates from real historical data:

```powershell
.\.venv-arakandar\Scripts\python.exe run_training.py `
  --mode scratch --n-estimators 500 `
  --out models/champion_lgbm.real.candidate.pkl `
  --register-candidate
```

Evaluate more than accuracy:

- Macro-F1 for class balance
- Log-loss and probability calibration
- Net P&L after fees and slippage
- Drawdown
- Turnover
- Walk-forward performance
- Stability across tickers and market regimes

A candidate with negative net P&L must not be promoted. Accuracy alone is not evidence of a profitable strategy.

## Analyst-model specialization

Generate examples from the deterministic workflow:

```powershell
.\.venv-arakandar\Scripts\python.exe run_specialization.py --refresh-real-data
```

Training examples should contain:

- A realistic analyst question
- Structured market evidence
- Deterministic signal and probabilities
- News sentiment and article metadata
- Tool workflow plan
- Memory/history context
- A reviewed answer separating view, evidence, risks, and bottom line

Never use unreviewed generated answers as labels. Include BUY, HOLD, and SELL cases, comparisons, missing-news cases, uncertainty, and risk questions.

## Adapter acceptance

Before enabling an adapter:

1. Confirm the base model path and adapter manifest.
2. Check that the dataset was generated from real data and reviewed.
3. Hold out evaluation examples that were not used in training.
4. Test that the answer remains grounded and does not invent evidence.
5. Verify the deterministic signal is identical with and without the adapter.
6. Test CPU/GPU loading and restart behavior.
7. Keep the previous adapter available for rollback.

## Safety boundary

This project is a research and paper-trading system. The disclaimer must remain visible. No live broker execution is implemented. Do not enable live trading based only on a better-looking language-model answer or classification accuracy.
