# Arakandar Project Guidelines

## Purpose

Arakandar is a purpose-built market research assistant. Its workflow combines market data, technical features, news sentiment, a deterministic signal model, tool execution, persistent task history, and a local language model that explains supplied evidence.

## Repository structure

- `app/agent/`: analyst orchestration, memory/state, tokens, ML prediction, and local model integration.
- `app/data/`: market and news providers plus validation.
- `app/api.py`: private authenticated HTTP boundary.
- `app/ui.py`: local Streamlit analyst workspace.
- `run_chat.py`: interactive CLI.
- `run_specialization.py`: real-data refresh, example generation, and 3B adapter training.
- `scripts/`: operational data and training utilities.
- `data/raw/`: validated OHLCV input files.
- `data/llm/`: reviewed or generated training examples.
- `data/agent_state.sqlite3`: local audit state, task history, market cache, and paper orders.
- `models/base/`: immutable base model files.
- `models/adapters/`: generated LoRA adapters; do not commit intermediate checkpoints.
- `models/*.pkl`: signal-model candidates and champion artifacts.
- `tests/`: required regression and contract tests.
- `docs/`: operator, project, and model guidance.

## Engineering rules

1. Keep the deterministic LightGBM signal authoritative. The language model explains evidence; it does not override the signal.
2. Keep live trading disabled. Paper trading is allowed; broker execution requires a separate reviewed safety implementation.
3. Never silently use stale, malformed, or synthetic data for production claims.
4. Preserve provenance: record provider, fetch time, date range, dropped rows, and model version.
5. Use candidate artifacts first. Never overwrite the champion during experiments.
6. Do not train on unreviewed model-generated answers.
7. Do not expose filesystem paths, arbitrary tools, model internals, or database access through the API.
8. Validate all external market/news input before it reaches features or prompts.
9. Keep changes small and add a focused test for changed behavior.
10. Do not commit `.env`, model weights, raw private data, caches, or intermediate checkpoints.

## Workflow contract

The analyst workflow is:

```text
market_snapshot -> news_sentiment -> signal_prediction -> analyst_report
```

Each step must be auditable in SQLite. A response must distinguish:

- Observed evidence
- Deterministic model output
- Interpretation
- Missing information and risks

## Change checklist

Before merging a change:

- Run `python -m pytest -q` in `.venv-arakandar`.
- Run `python -m py_compile` for changed Python files.
- Verify no secrets, caches, or generated checkpoints are included.
- Check that offline mode still works.
- Check that the deterministic signal is unchanged by language-model wording changes.
- Update the relevant guide when commands, paths, model behavior, or deployment requirements change.
