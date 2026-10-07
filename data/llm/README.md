# Arakandar 3B Training Data

You can generate a starting dataset from the deterministic analyst workflow:

```powershell
.\.venv-arakandar\Scripts\python.exe scripts/generate_analyst_training_data.py
```

Fetch real historical data before generating examples:

```powershell
.\.venv-arakandar\Scripts\python.exe scripts/fetch_real_market_data.py `
	--ticker BBCA.JK --ticker BBRI.JK --ticker BMRI.JK --ticker TLKM.JK
```

This creates grounded examples from the current market files, technical signal,
news sentiment, workflow plan, and memory fields. Review the generated file
before training. Do not generate answers from the language model and feed them
back as truth. Each line must contain a question, the
structured evidence available to the assistant, and a human-reviewed answer:

```json
{"question":"Should I buy BBCA?","evidence":{"ticker":"BBCA.JK","close":6300,"rsi14":43.7,"macd_diff":-39.47,"signal":"HOLD","confidence":0.67,"news_sentiment":0.0},"answer":"The deterministic signal is HOLD. RSI is neutral, while the negative MACD difference suggests weakening momentum. The evidence does not justify a BUY conclusion, and position sizing or risk tolerance is unavailable."}
```

Use diverse questions, BUY/HOLD/SELL cases, missing-news cases, comparisons,
and examples where the safe answer is that evidence is insufficient. Review for
invented facts, unsupported certainty, and accidental financial advice.

Run training with:

```powershell
.\.venv-arakandar\Scripts\python.exe -m pip install -r requirements-local-llm.txt
.\.venv-arakandar\Scripts\python.exe scripts/train_arakandar_lora.py
```

The script requires at least eight reviewed examples, holds out 20% for
evaluation, and writes an adapter under `models/adapters/`. The original 3B
weights are not modified.
