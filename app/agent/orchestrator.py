"""Custom multi-step analyst agent for the Arakandar training workbench."""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List

import pandas as pd

from app.agent.ml.predictor import Predictor
from app.agent.tokens.market_tokenizer import compute_technical_features, tokenize_dataframe
from app.agent.tokens.nlp_tokenizer import batch_sentiment_factor
from app.agent.storage import AgentStore
from app.agent.llm.context import build_market_messages
from app.agent.llm.local_model import LocalLanguageModel
from app.data.news import NewsIngestor, RSSProvider, headlines_from_articles


def _answer_is_ticker_grounded(
    answer: str,
    question: str,
    ticker: str,
    known_tickers: set[str] | None = None,
) -> bool:
    def aliases(symbol: str) -> set[str]:
        normalized = symbol.upper()
        return {normalized, normalized.split(".", 1)[0]}

    def mentions(text: str, symbols: set[str]) -> bool:
        return any(
            re.search(rf"(?<![A-Z0-9]){re.escape(symbol)}(?![A-Z0-9])", text, re.IGNORECASE)
            for symbol in symbols
        )

    ticker_symbols = aliases(ticker)
    if not mentions(answer, ticker_symbols):
        return False

    allowed_tickers = {ticker.upper()}
    for known_ticker in known_tickers or set():
        if mentions(question, aliases(known_ticker)):
            allowed_tickers.add(known_ticker.upper())
    for known_ticker in known_tickers or set():
        if known_ticker.upper() not in allowed_tickers and mentions(answer, aliases(known_ticker)):
            return False

    index_pattern = r"\b(?:IHSG|IDX\s+Composite|Jakarta\s+Composite(?:\s+Index)?)\b"
    mentions_ihsg = re.search(index_pattern, answer, re.IGNORECASE)
    asks_about_ihsg = re.search(index_pattern, question, re.IGNORECASE)
    return not mentions_ihsg or bool(asks_about_ihsg)


def _question_is_market_related(question: str) -> bool:
    market_terms = (
        r"\b(?:stock|stocks|share|shares|ticker|market|trading|trade|buy|sell|hold|"
        r"price|chart|signal|technical|indicator|rsi|macd|volume|dividend|earnings|"
        r"revenue|profit|valuation|catalyst|pullback|portfolio|invest(?:ment|ing)?|"
        r"risks?|support|resistance|bullish|bearish|ihsg|idx|saham|pasar|perdagangan|"
        r"beli|jual|tahan|harga|grafik|sinyal|teknikal|indikator|dividen|laba|"
        r"pendapatan|valuasi|katalis|koreksi|portofolio|investasi|risiko|berita|"
        r"analisis|analisa)\b"
    )
    return re.search(market_terms, question, re.IGNORECASE) is not None


@dataclass
class Tool:
    name: str
    description: str
    handler: Callable[..., Dict[str, Any]]


class ArakandarAgent:
    """Plans and executes an explain-signal workflow through custom tools."""

    def __init__(
        self,
        data_dir: Path,
        model_path: Path,
        store: AgentStore,
        language_model: LocalLanguageModel | None = None,
        rss_feeds: Dict[str, str] | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.model_path = Path(model_path)
        self.store = store
        self.language_model = language_model
        self.news_ingestor = NewsIngestor(RSSProvider(rss_feeds or {}), store)
        self.tools: Dict[str, Tool] = {}
        self.register(Tool("market_snapshot", "Load and tokenize the latest OHLCV state.", self._market_snapshot))
        self.register(Tool("news_sentiment", "Score supplied headlines for the selected ticker.", self._news_sentiment))
        self.register(Tool("signal_prediction", "Predict a signal from the latest feature row.", self._signal_prediction))
        self.register(Tool("analyst_report", "Synthesize evidence into an analyst-facing report.", self._analyst_report))

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def chat(
        self,
        question: str,
        ticker: str,
        headlines: List[str] | None = None,
        news_articles: List[Dict[str, Any]] | None = None,
        history: List[Dict[str, str]] | None = None,
    ) -> Dict[str, Any]:
        """Answer one ticker-focused question from fresh local market evidence."""
        if not question.strip():
            raise ValueError("Chat question cannot be empty")
        if news_articles is None:
            self.news_ingestor.sync(ticker)
            news_articles = self.store.latest_news(ticker, limit=10)
        if headlines is None:
            headlines = headlines_from_articles(news_articles)
        result = self.run(
            question.strip(),
            ticker,
            headlines=headlines,
            news_articles=news_articles,
            history=history,
        )
        report = result["report"]
        return {
            "task_id": result["task_id"],
            "ticker": ticker,
            "question": question.strip(),
            "answer": report["summary"],
            "headline": report["headline"],
            "evidence": report["evidence"],
            "sources": [
                {
                    "title": article.get("title"),
                    "source": article.get("source"),
                    "published_at": article.get("published_at"),
                    "url": article.get("url"),
                }
                for article in (news_articles or [])
            ],
            "signal": result["report"]["headline"].split(": ", 1)[-1],
            "generation": report.get("generation", {"model": "deterministic"}),
            "disclaimer": report["disclaimer"],
        }

    def run(
        self,
        task: str,
        ticker: str,
        headlines: List[str] | None = None,
        news_articles: List[Dict[str, Any]] | None = None,
        history: List[Dict[str, str]] | None = None,
    ) -> Dict[str, Any]:
        task_id = uuid.uuid4().hex
        self.store.start_task(task_id, task)
        context: Dict[str, Any] = {
            "task_id": task_id,
            "task": task,
            "ticker": ticker,
            "headlines": headlines or [],
            "news_articles": news_articles or [],
            "history": history or [],
        }
        plan = ["market_snapshot", "news_sentiment", "signal_prediction", "analyst_report"]
        context["plan"] = plan

        try:
            for tool_name in plan:
                tool = self.tools[tool_name]
                inputs = {"ticker": ticker}
                if tool_name == "news_sentiment":
                    inputs["headline_count"] = len(context["headlines"])
                started_at = datetime.now(timezone.utc).isoformat()
                try:
                    output = tool.handler(context)
                except Exception as exc:
                    self.store.record_tool_event(task_id, tool_name, "failed", inputs, error=str(exc), started_at=started_at)
                    raise
                context[tool_name] = output
                self.store.record_tool_event(task_id, tool_name, "completed", inputs, output=output, started_at=started_at)

            result = {
                "task_id": task_id,
                "task": task,
                "ticker": ticker,
                "plan": plan,
                "report": context["analyst_report"],
                "route": {name: "local_csv" if name == "market_snapshot" else "local_model_or_heuristic" for name in plan},
            }
            self.store.finish_task(task_id, result)
            return result
        except Exception as exc:
            self.store.fail_task(task_id, str(exc))
            raise

    def _market_snapshot(self, context: Dict[str, Any]) -> Dict[str, Any]:
        ticker = context["ticker"]
        bars = self.store.latest_market_bars(ticker, limit=200)
        if bars:
            raw = pd.DataFrame(bars)
            raw = raw.rename(columns={"timestamp": "date"})
            raw["date"] = pd.to_datetime(raw["date"])
        else:
            path = self.data_dir / f"{ticker.replace('.', '_')}_ohlcv.csv"
            if not path.exists():
                raise FileNotFoundError(f"No market data found in store or CSV for {ticker}: {path}")
            raw = pd.read_csv(path, parse_dates=["date"])
        features = tokenize_dataframe(compute_technical_features(raw.sort_values("date").reset_index(drop=True)))
        row = features.iloc[-1]
        return {
            "ticker": ticker,
            "as_of": row["date"].isoformat() if hasattr(row["date"], "isoformat") else str(row["date"]),
            "close": float(row["close"]),
            "rsi14": float(row["rsi14"]),
            "macd_diff": float(row["macd_diff"]),
            "active_tokens": str(row["active_token_names"]).split(","),
            "feature_row": {key: row[key] for key in row.index},
        }

    def _news_sentiment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        headlines = context["headlines"]
        return {"headline_count": len(headlines), "sentiment": round(batch_sentiment_factor(headlines), 4), "headlines": headlines}

    def _signal_prediction(self, context: Dict[str, Any]) -> Dict[str, Any]:
        snapshot = context["market_snapshot"]
        feature_row = pd.Series(snapshot["feature_row"])
        feature_row["news_sentiment_factor"] = context["news_sentiment"]["sentiment"]
        if self.model_path.exists():
            try:
                prediction = Predictor(self.model_path).predict(feature_row)
                prediction["source"] = "champion_model"
                return prediction
            except (ImportError, ModuleNotFoundError, OSError, ValueError) as exc:
                context["model_fallback_reason"] = str(exc)

        rsi = float(feature_row["rsi14"])
        macd_diff = float(feature_row["macd_diff"])
        if rsi <= 30 and macd_diff >= 0:
            signal = "BUY"
        elif rsi >= 70 and macd_diff <= 0:
            signal = "SELL"
        else:
            signal = "HOLD"
        return {
            "signal": signal,
            "confidence": 0.34,
            "probabilities": {"BUY": 0.34, "HOLD": 0.34, "SELL": 0.32},
            "top_drivers": ["rsi14", "macd_diff"],
            "source": "technical_heuristic",
            "fallback_reason": context.get("model_fallback_reason"),
        }

    def _analyst_report(self, context: Dict[str, Any]) -> Dict[str, Any]:
        snapshot = context["market_snapshot"]
        prediction = context["signal_prediction"]
        news = context["news_sentiment"]
        news_with_articles = {**news, "articles": context.get("news_articles", [])}
        deterministic_report = {
            "headline": f"{context['ticker']} signal: {prediction['signal']}",
            "summary": (
                f"{context['ticker']}: latest close is {snapshot['close']:.2f} "
                f"with RSI {snapshot['rsi14']:.1f}. "
                f"The selected signal is {prediction['signal']} from {prediction['source']}; "
                f"news sentiment is {news['sentiment']:+.2f} across {news['headline_count']} headlines."
            ),
            "evidence": {
                "as_of": snapshot["as_of"],
                "close": snapshot["close"],
                "rsi14": snapshot["rsi14"],
                "macd_diff": snapshot["macd_diff"],
                "active_tokens": snapshot["active_tokens"],
                "top_drivers": prediction["top_drivers"],
                "confidence": prediction["confidence"],
                "probabilities": prediction.get("probabilities", {}),
                "news_sentiment": news["sentiment"],
            },
            "disclaimer": "Research signal only. No live order was placed.",
        }
        if not _question_is_market_related(context["task"]):
            deterministic_report["summary"] = (
                "I'm Arakandar, a market research assistant. I can only help with "
                "stock-market analysis, market news, and trading signals."
            )
            deterministic_report["generation"] = {"model": "scope_guard"}
            return deterministic_report
        if self.language_model is None:
            return deterministic_report

        messages = build_market_messages(
            context["task"],
            context["ticker"],
            snapshot,
            news_with_articles,
            prediction,
            history=context.get("history"),
        )
        try:
            answer = self.language_model.generate(
                messages,
                max_new_tokens=768,
                do_sample=True,
                temperature=0.55,
                top_p=0.85,
                repetition_penalty=1.05,
            )
        except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError) as exc:
            context["language_model_fallback_reason"] = str(exc)
            deterministic_report["language_model_fallback_reason"] = str(exc)
            return deterministic_report

        if not answer.strip():
            deterministic_report["language_model_fallback_reason"] = "empty local-model response"
            return deterministic_report
        answer = answer.strip()
        known_tickers = set(self.store.market_tickers())
        known_tickers.update(
            path.stem.removesuffix("_ohlcv").replace("_", ".").upper()
            for path in self.data_dir.glob("*_ohlcv.csv")
        )
        if not _answer_is_ticker_grounded(
            answer, context["task"], context["ticker"], known_tickers
        ):
            deterministic_report["language_model_fallback_reason"] = (
                "local-model response was not grounded in the requested ticker"
            )
            return deterministic_report
        return {
            **deterministic_report,
            "summary": answer,
            "generation": {"model": getattr(self.language_model, "model_name", "Arakandar")},
        }


def build_default_agent(project_root: Path | None = None) -> ArakandarAgent:
    root = Path(project_root or Path(__file__).resolve().parents[2])
    return ArakandarAgent(
        data_dir=root / "data" / "raw",
        model_path=root / "models" / "champion_lgbm.pkl",
        store=AgentStore(root / "data" / "agent_state.sqlite3"),
    )


if __name__ == "__main__":
    result = build_default_agent().run(
        "Explain the latest signal", "BBCA.JK", ["BBCA reports record profit and dividend hike"]
    )
    print(json.dumps(result, indent=2, default=str))
