# Arakandar AI Trainer (Standalone Sandbox)

This is a **standalone training sandbox** for the Arakandar Gradient Boosting
Decision Engine, built from `implementation_plan.md` (Phases 2 & 3: Dual
Token System + ML Engine). It is intentionally **decoupled from the
`RexCrowSS1/Arakandar` GitHub repo** — train and validate the model here
first; once you're happy with results, copy `app/agent/` into
`Backend/app/agent/` in the real repo.

## What's inside

```
arakandar_ai_training/
├── app/agent/
│   ├── tokens/
│   │   ├── market_tokenizer.py   # Symbolic market-state tokens (RSI, MACD, EMA, ATR, OBV, Bollinger)
│   │   ├── nlp_tokenizer.py      # Financial headline sentiment tokenizer (offline lexicon)
│   │   └── session_tokens.py     # Bloomberg-style terminal token ledger
│   └── ml/
│       ├── feature_pipeline.py   # Multi-factor feature matrix + BUY/HOLD/SELL labeling
│       ├── trainer.py            # train_from_scratch / continue_training / fine_tune
│       └── predictor.py          # <2ms inference + explainability
├── data/raw/                     # Drop your own OHLCV CSVs here
├── models/                       # champion_lgbm.pkl + .report.json land here
├── tests/                        # pytest unit tests (8/8 passing)
├── run_training.py               # CLI entry point (see below)
└── requirements.txt
```

This sandbox contains the data pipeline, tokenizers, and model trainer only;
it now includes a Streamlit analyst workspace, local automation, model
governance, and paper trading. Live broker execution is intentionally absent.

## Guides

- [Run guide](docs/RUN_GUIDE.md): setup, chat, real data, training, API, and tests.
- [Project guidelines](docs/PROJECT_GUIDELINES.md): structure, workflow contract, and engineering rules.
- [Model guidelines](docs/MODEL_GUIDELINES.md): signal model, 3B adapter, evaluation, and safety boundaries.

## ML backend note (important)

The plan specifies **LightGBM**. `trainer.py` tries `import lightgbm` first
and uses it automatically if installed — this gives you true native
`init_model` warm-starts exactly as the plan describes.

If LightGBM isn't installed (e.g. fully offline machine), it **silently
falls back** to scikit-learn's `HistGradientBoostingClassifier`, which is
the same histogram-based GBDT family. Nothing else in the code changes.
To get LightGBM:

```bash
pip install -r requirements.txt

# Verify you are using the same interpreter for install and execution
python -c "import sys; print(sys.executable)"
python -m pytest -q
```

## Quickstart

```bash
cd arakandar_ai_training
pip install -r requirements.txt

# 1. Train a brand-new model from scratch on everything in data/raw/
python run_training.py --mode scratch --n-estimators 300

# 2. Later, after you append fresh bars to data/raw/*.csv, extend the
#    existing model with more boosting rounds (true incremental training,
#    old trees are kept):
python run_training.py --mode continue --rounds 100

# 3. Specialize the champion model onto one ticker/regime without
#    forgetting the general patterns (low LR, few rounds):
python run_training.py --mode finetune --ticker BBCA.JK --rounds 50 --lr 0.01

# 4. Ingest and persist market bars (repeat --ticker for more symbols)
python run_ingestion.py --ticker BBCA.JK --ticker BBRI.JK --provider csv

# 5. Run one complete market-data -> agent -> report cycle
python run_automation.py --ticker BBCA.JK

# 6. Keep polling until interrupted
python run_automation.py --ticker BBCA.JK --watch --interval-seconds 300

# 7. Train an approval-gated candidate without replacing champion
python run_training.py --mode scratch --register-candidate

# 8. Start the analyst workspace
python run_ui.py

# 9. Chat directly with the local Qwen-based Arakandar model
python run_chat.py --ticker BBCA.JK

# 10. Start the authenticated HTTP service
copy .env.example .env
python run_api.py

# 11. Automatically generate grounded data and train the 3B adapter
python run_specialization.py --refresh-real-data

# Use the fast deterministic summary instead of loading the local LLM
python run_chat.py --ticker BBCA.JK --heuristic

# Ask one question and exit
python run_chat.py --ticker BBCA.JK --question "What is the latest catalyst?"

# Include a configured local RSS refresh for the ticker
python run_chat.py --ticker BBCA.JK --rss-feed "BBCA.JK=https://example.test/feed.xml"

# Chat uses live Yahoo Finance market data and Google News RSS by default.
# Use cached SQLite/CSV data only when offline.
python run_chat.py --ticker BBCA.JK --offline

# Interactive commands: /help, /listticker, /ticker SYMBOL, /quit
```

Each run prints a `TrainingReport` JSON (also saved next to the model as
`champion_lgbm.report.json`) with train/val accuracy, macro F1, log-loss,
the custom **net_pnl_score** (asymmetric financial P&L metric from the
plan), and top-15 feature importances for explainability.

Candidate training writes a separate artifact and records it in the local model
registry. Approve candidates from the Streamlit Model registry panel; training
never promotes a candidate automatically.

The Streamlit Paper portfolio panel records simulated BUY/SELL fills and
positions in SQLite. It does not connect to a broker or place live orders.

## Private API deployment

The HTTP service exposes `/healthz`, authenticated `/readyz`, `/v1/chat`, and
`/v1/paper-orders`. Configure it with the variables in `.env.example` and send
the configured value as the `X-API-Key` header. `/healthz` is unauthenticated
for process probes; application endpoints require authentication.

Live broker execution is not implemented and the service rejects attempts to
enable it. For an internet-connected deployment, bind Uvicorn to localhost
behind an HTTPS reverse proxy or private VPN. Do not expose Uvicorn directly,
commit `.env`, or use the included synthetic CSV files as evidence for real
trading.

### Docker

The included `Dockerfile` runs as a non-root user. Copy `.env.example` to `.env`,
replace the API key, and keep the model and data directories mounted as shown in
`docker-compose.yml`:

```bash
docker compose up --build -d
curl http://127.0.0.1:8000/healthz
```

The compose file binds only to localhost and persists `data/`; put an HTTPS
reverse proxy or private VPN in front of it before allowing remote access.

## Train the 3B conversation adapter

The 3B base model is selected automatically for chat. `run_specialization.py`
builds grounded examples from the deterministic analyst workflow, including
market evidence, signal output, tool plan, and memory fields, then holds out
20% and trains a LoRA adapter without changing the base weights. Chat
automatically uses the adapter when it exists. Review the generated dataset and
use `--refresh-real-data` to download and validate Yahoo Finance history first.
The previous raw files are backed up under `data/backups/synthetic/` and a
provenance manifest is written to `data/raw/market_data_manifest.json`.

On Windows, select the same interpreter in VS Code that you use for
`python -m pip install -r requirements.txt`. If `python -m pytest` reports
that pytest is missing, `python` is resolving to a different installation;
run `where python` and switch the VS Code interpreter before installing again.

## Bring your own data (search source online)

The pipeline only needs 6 columns per ticker: `date, open, high, low, close,
volume`. To plug in real historical data you source yourself:

1. Save a CSV as `data/raw/<TICKER>_ohlcv.csv` (e.g. `data/raw/AAPL_ohlcv.csv`,
   `data/raw/BBCA_JK_ohlcv.csv`). Underscores replace dots in the filename.
2. Good free sources: `yfinance` (already in requirements.txt), Stooq,
   IDX/KSEI historical exports, or Perplexity Finance exports.
3. Re-run `--mode scratch` (full retrain) or `--mode continue` (extend).

The included `data/raw/*.csv` files are **synthetic placeholder data**
(random-walk OHLCV calibrated to real BBCA.JK/BBRI.JK/BMRI.JK/TLKM.JK price
ranges) so you can verify the whole pipeline runs before spending time
sourcing real history. Replace them with real data for a model you'd
actually trust.

## Design choices carried over from the plan

- **Directional multi-class target**: BUY if forward return over 5 bars
  > +2%, SELL if < -2%, else HOLD (configurable via `--horizon` and
  thresholds in `feature_pipeline.make_labels`).
- **Dual Token System**: market-state tokens are one-hot categorical
  features; news sentiment collapses to a single scalar factor
  (`news_sentiment_factor`), both feed the same feature matrix as the
  numeric technical indicators.
- **Asymmetric financial P&L scoring** (`trainer.financial_pnl_score`):
  wrong directional calls are penalized quadratically harder than a missed
  HOLD, matching the plan's loss formula.

## Next steps to wire into the real repo

1. Validate `net_pnl_score` and `val_f1_macro` improve as you add real data
   and tune `--horizon` / thresholds.
2. Once satisfied, copy `app/agent/tokens/` and `app/agent/ml/` into
   `Backend/app/agent/` in `RexCrowSS1/Arakandar`.
3. Continue with Phase 4 (risk_manager.py, decision_engine.py) and Phase 5
   (FastAPI routes) from `implementation_plan.md`, which are not part of
   this training-only sandbox.
