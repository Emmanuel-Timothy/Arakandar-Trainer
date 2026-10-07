# Arakandar Run Guide

## 1. Set up Windows

From the repository root:

```powershell
uv venv .venv-arakandar --python 3.14
uv pip install --python .venv-arakandar\Scripts\python.exe -r requirements.txt -r requirements-local-llm.txt
```

Use `.venv-arakandar\Scripts\python.exe` for every command. The local model files are already expected under `models/base/`.

## 2. Run chat

Use the trained 3B adapter automatically:

```powershell
.\.venv-arakandar\Scripts\python.exe run_chat.py --ticker BBCA.JK
```

The model selection order is:

1. `models/adapters/arakandar-3b-lora`
2. `models/base/arakandar-3b`
3. `models/base/arakandar-1.5b`

Useful modes:

```powershell
# One question
.\.venv-arakandar\Scripts\python.exe run_chat.py --ticker BBCA.JK --question "Explain the latest signal and risks"

# Offline/cached data only
.\.venv-arakandar\Scripts\python.exe run_chat.py --ticker BBCA.JK --offline

# Deterministic signal summary without the language model
.\.venv-arakandar\Scripts\python.exe run_chat.py --ticker BBCA.JK --heuristic
```

Interactive commands:

- `/help`
- `/listticker`
- `/ticker BBCA.JK`
- `/quit` or `/exit`

## 3. Refresh real market data

This backs up the current raw files, downloads Yahoo Finance history, removes unusable provider rows, and writes a provenance manifest:

```powershell
.\.venv-arakandar\Scripts\python.exe scripts/fetch_real_market_data.py `
  --ticker BBCA.JK --ticker BBRI.JK --ticker BMRI.JK --ticker TLKM.JK
```

Check `data/raw/market_data_manifest.json` before training. The previous files are in `data/backups/synthetic/`.

## 4. Retrain the signal model

Always create a candidate first:

```powershell
.\.venv-arakandar\Scripts\python.exe run_training.py `
  --mode scratch --n-estimators 500 `
  --out models/champion_lgbm.real.candidate.pkl `
  --register-candidate
```

Do not promote a candidate only because accuracy improved. Check macro-F1, log-loss, net P&L after costs, drawdown, and walk-forward stability.

## 5. Retrain the 3B analyst adapter

The one-command workflow refreshes data, generates grounded examples, and trains the adapter:

```powershell
.\.venv-arakandar\Scripts\python.exe run_specialization.py --refresh-real-data
```

It writes the dataset to `data/llm/train.jsonl` and the adapter to `models/adapters/arakandar-3b-lora/`. Review the dataset before using the adapter for serious analysis.

## 6. Run the private API

Copy `.env.example` to `.env`, set a long random `ARAKANDAR_API_KEY`, and keep live trading disabled:

```powershell
Copy-Item .env.example .env
.\.venv-arakandar\Scripts\python.exe run_api.py
```

The API binds to localhost by default. Use HTTPS/private VPN before remote access. Endpoints:

- `GET /healthz`
- authenticated `GET /readyz`
- authenticated `POST /v1/chat`
- authenticated `POST /v1/paper-orders`

There is no live broker endpoint.

## 7. Verify the installation

```powershell
.\.venv-arakandar\Scripts\python.exe -m pytest -q
```

The project is operational only when the tests pass and the model/data manifest is present.
