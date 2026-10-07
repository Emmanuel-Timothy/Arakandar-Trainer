"""Generate deterministic, evidence-grounded examples for the Arakandar adapter."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.orchestrator import ArakandarAgent
from app.agent.storage import AgentStore


QUESTIONS = (
    "Explain the latest signal for {ticker}.",
    "Should I buy {ticker} right now? Explain the evidence and risks.",
    "What are the main technical drivers for {ticker} and what could invalidate the view?",
    "Give me a cautious analyst brief for {ticker} using the latest market and news evidence.",
    "What information is missing before making a decision about {ticker}?",
)


def _answer(question: str, ticker: str, snapshot: dict[str, Any], news: dict[str, Any], signal: dict[str, Any]) -> str:
    rsi = float(snapshot["rsi14"])
    macd = float(snapshot["macd_diff"])
    signal_name = str(signal["signal"])
    confidence = float(signal["confidence"])
    sentiment = float(news["sentiment"])
    rsi_view = "oversold" if rsi <= 30 else "overbought" if rsi >= 70 else "neutral"
    momentum = "positive" if macd > 0 else "negative" if macd < 0 else "flat"
    risk = (
        "The view could be invalidated by a reversal in momentum, new adverse news, or a change in the model inputs. "
        "Position size, time horizon, and risk tolerance are not available."
    )
    return (
        f"View: {ticker} has a deterministic {signal_name} signal with {confidence:.0%} confidence.\n\n"
        f"Evidence: The latest close is {float(snapshot['close']):.2f}. RSI14 is {rsi:.2f}, which is {rsi_view}; "
        f"MACD difference is {macd:.2f}, indicating {momentum} momentum. News sentiment is {sentiment:+.2f} "
        f"from {int(news['headline_count'])} headline(s). The signal is produced by {signal.get('source', 'the analyst model')}.\n\n"
        f"Risks: {risk}\n\n"
        f"Bottom line: Treat this as a research signal, not a command to trade. The evidence supports {signal_name}, "
        "but it does not establish future returns or suitability."
    )


def generate(output: Path, data_dir: Path, state: Path, model: Path, tickers: list[str]) -> int:
    store = AgentStore(state)
    agent = ArakandarAgent(data_dir, model, store, language_model=None)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for ticker in tickers:
            try:
                result = agent.run(
                    "Explain the latest signal",
                    ticker,
                    headlines=[f"Latest market update for {ticker}"],
                )
            except (FileNotFoundError, ValueError, KeyError) as exc:
                print(f"Skipping {ticker}: {exc}")
                continue
            snapshot = result["report"]["evidence"]
            signal = {
                "signal": result["report"]["headline"].split(": ", 1)[-1],
                "confidence": snapshot["confidence"],
                "probabilities": snapshot.get("probabilities", {}),
                "top_drivers": snapshot.get("top_drivers", []),
                "source": "champion_model_or_heuristic",
            }
            news = {
                "headline_count": 1,
                "sentiment": snapshot.get("news_sentiment", 0.0),
            }
            for question_template in QUESTIONS:
                question = question_template.format(ticker=ticker)
                handle.write(json.dumps({
                    "question": question,
                    "evidence": {
                        "ticker": ticker,
                        "market": snapshot,
                        "signal": signal,
                        "news": news,
                        "workflow": result["plan"],
                        "memory": {"previous_turns": [], "task_id": result["task_id"]},
                    },
                    "answer": _answer(question, ticker, snapshot, news, signal),
                }, ensure_ascii=True) + "\n")
                count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate grounded Arakandar analyst training examples")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--state", type=Path, default=Path("data/agent_state.sqlite3"))
    parser.add_argument("--signal-model", type=Path, default=Path("models/champion_lgbm.pkl"))
    parser.add_argument("--output", type=Path, default=Path("data/llm/train.jsonl"))
    parser.add_argument("--ticker", action="append")
    args = parser.parse_args()
    tickers = args.ticker or ["BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK"]
    count = generate(args.output, args.data_dir, args.state, args.signal_model, tickers)
    print(f"Generated {count} grounded examples at {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
