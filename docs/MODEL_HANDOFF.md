# Arakandar Model Handoff

This package owns local model inference and model evidence contracts. The
backend teammate owns FastAPI, RSS scheduling, persistence, and frontend wiring.

## Validated base model

- Assistant name: `Arakandar`
- Local path: `models/base/arakandar-1.5b`
- Higher-capacity local path: `models/base/arakandar-3b`
- Runtime: `requirements-local-llm.txt`
- Validated device: CUDA on an RTX 5070 with 12 GB VRAM
- Weights: share through Git LFS, a release artifact, or a model registry; do
  not commit multi-gigabyte weights to ordinary Git history.

The chat CLI prefers the trained `models/adapters/arakandar-3b-lora` adapter
when present, then `arakandar-3b`, and finally falls back to `arakandar-1.5b`.
The 3B artifact is compatible with the existing Transformers loader and is
suitable for inference on the validated NVIDIA GPU. The adapter is only as
good as the reviewed examples used to train it.

## Install and validate

```powershell
uv venv .venv-arakandar --python 3.14
uv pip install --python .venv-arakandar\Scripts\python.exe -r requirements.txt -r requirements-local-llm.txt
.\.venv-arakandar\Scripts\python.exe -m scripts.validate_local_model
```

The model can run on CPU by constructing `TransformersLocalModel(...,
device="cpu", torch_dtype="float32")`, but CUDA is recommended.

## Contract

The backend should pass structured evidence with these fields:

```json
{
  "question": "What is the latest catalyst?",
  "market": {
    "ticker": "BBCA.JK",
    "as_of": "2026-09-13T09:00:00+00:00",
    "close": 9875,
    "rsi14": 54,
    "macd_diff": 1.2,
    "active_tokens": ["TK_PRICE_ABOVE_EMA50"]
  },
  "signal": {
    "signal": "HOLD",
    "confidence": 0.62,
    "probabilities": {"BUY": 0.2, "HOLD": 0.62, "SELL": 0.18},
    "top_drivers": ["rsi14", "macd_diff"],
    "source": "champion_model"
  },
  "news": [],
  "news_sentiment": 0.2,
  "history": []
}
```

LightGBM remains authoritative for `signal` and probabilities. Arakandar
explains the supplied evidence and must not invent prices, news, or trades.

The response should expose:

```json
{
  "assistant": "Grounded response text",
  "model": "Arakandar",
  "latency_ms": 14580,
  "sources": [],
  "deterministic_signal": "HOLD",
  "disclaimer": "Research signal only. No live order was placed."
}
```

Use `ModelInput.validate()` and `ModelOutput.validate()` at the integration
boundary. Realtime RSS/Atom fetching belongs to the backend; pass article title,
source, publication timestamp, URL, and sentiment into `news`.

## Fine-tuning status

The base model is validated but not fine-tuned. Do not train on generated
answers without human review. Future LoRA/QLoRA adapters will be stored
separately from the base model and evaluated for grounding independently of
LightGBM trading metrics.