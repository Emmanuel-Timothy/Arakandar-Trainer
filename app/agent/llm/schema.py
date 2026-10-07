"""Structured model contract shared with the backend integration."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class NewsEvidence:
    title: str
    source: str = ""
    published_at: str = ""
    url: str = ""
    sentiment: float | None = None


@dataclass(frozen=True)
class MarketEvidence:
    ticker: str
    as_of: str
    close: float
    rsi14: float | None = None
    macd_diff: float | None = None
    active_tokens: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SignalEvidence:
    signal: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    top_drivers: list[str] = field(default_factory=list)
    source: str = ""


@dataclass(frozen=True)
class ModelInput:
    """Evidence the backend passes to Arakandar; no live API dependency."""

    question: str
    market: MarketEvidence
    signal: SignalEvidence
    news: list[NewsEvidence] = field(default_factory=list)
    news_sentiment: float | None = None
    history: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if not self.question.strip():
            raise ValueError("question cannot be empty")
        if not self.market.ticker.strip():
            raise ValueError("market.ticker cannot be empty")
        if self.signal.signal not in {"BUY", "HOLD", "SELL"}:
            raise ValueError("signal must be BUY, HOLD, or SELL")
        if not 0 <= self.signal.confidence <= 1:
            raise ValueError("signal.confidence must be between 0 and 1")
        for article in self.news:
            if not article.title.strip():
                raise ValueError("news article title cannot be empty")


@dataclass(frozen=True)
class ModelOutput:
    """Result returned by a model wrapper to the backend."""

    assistant: str
    model: str = "Arakandar"
    latency_ms: float | None = None
    sources: list[dict[str, str]] = field(default_factory=list)
    deterministic_signal: str | None = None
    disclaimer: str = "Research signal only. No live order was placed."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if not self.assistant.strip():
            raise ValueError("assistant response cannot be empty")
        if self.deterministic_signal and self.deterministic_signal not in {"BUY", "HOLD", "SELL"}:
            raise ValueError("deterministic_signal must be BUY, HOLD, or SELL")


def model_input_from_evidence(
    question: str,
    ticker: str,
    market: Mapping[str, Any],
    signal: Mapping[str, Any],
    articles: list[Mapping[str, Any]] | None = None,
    news_sentiment: float | None = None,
    history: list[Mapping[str, str]] | None = None,
) -> ModelInput:
    """Convert existing agent dictionaries into the public model contract."""
    result = ModelInput(
        question=question,
        market=MarketEvidence(
            ticker=ticker,
            as_of=str(market.get("as_of", "")),
            close=float(market.get("close", 0)),
            rsi14=market.get("rsi14"),
            macd_diff=market.get("macd_diff"),
            active_tokens=list(market.get("active_tokens", [])),
        ),
        signal=SignalEvidence(
            signal=str(signal.get("signal", "HOLD")),
            confidence=float(signal.get("confidence", 0)),
            probabilities=dict(signal.get("probabilities", {})),
            top_drivers=list(signal.get("top_drivers", [])),
            source=str(signal.get("source", "")),
        ),
        news=[
            NewsEvidence(
                title=str(article.get("title", "")),
                source=str(article.get("source", "")),
                published_at=str(article.get("published_at", "")),
                url=str(article.get("url", "")),
                sentiment=article.get("sentiment"),
            )
            for article in (articles or [])
        ],
        news_sentiment=news_sentiment,
        history=[dict(message) for message in (history or [])],
    )
    result.validate()
    return result