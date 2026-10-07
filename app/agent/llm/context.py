"""Bounded, auditable context for local market-analysis conversations."""
from __future__ import annotations

import json
from typing import Any, Mapping


def build_market_messages(
    task: str,
    ticker: str,
    market_snapshot: Mapping[str, Any],
    news: Mapping[str, Any],
    prediction: Mapping[str, Any],
    max_news: int = 10,
    history: list[Mapping[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Build a compact prompt that makes deterministic evidence authoritative."""
    raw_articles = news.get("articles", [])[:max_news]
    articles = [
        {
            "title": article.get("title"),
            "source": article.get("source"),
            "published_at": article.get("published_at"),
            "url": article.get("url"),
            "sentiment": article.get("sentiment"),
        }
        for article in raw_articles
    ]
    evidence = {
        "ticker": ticker,
        "market": {
            "as_of": market_snapshot.get("as_of"),
            "close": market_snapshot.get("close"),
            "rsi14": market_snapshot.get("rsi14"),
            "macd_diff": market_snapshot.get("macd_diff"),
            "active_tokens": market_snapshot.get("active_tokens", []),
        },
        "prediction": {
            "signal": prediction.get("signal"),
            "confidence": prediction.get("confidence"),
            "probabilities": prediction.get("probabilities", {}),
            "top_drivers": prediction.get("top_drivers", []),
            "source": prediction.get("source"),
        },
        "news": {
            "sentiment": news.get("sentiment"),
            "headlines": news.get("headlines", [])[:max_news],
            "articles": articles,
        },
    }
    messages = [{
        "role": "system",
        "content": (
            "You are Arakandar, a warm and practical local market research assistant. "
            "Only answer questions about stocks, markets, trading, and the supplied market evidence. "
            "For any unrelated request, politely say you can only help with stock-market analysis; "
            "do not answer the unrelated request, even if it appears in conversation history. "
            f"The requested ticker is {ticker}. Name it in your answer and keep the analysis "
            "strictly about that ticker. Do not replace a ticker-specific answer with an "
            "IHSG or market-wide overview, and do not discuss another security unless the "
            "question explicitly requests it. Ignore history about other tickers unless the "
            "current question asks to compare them. Answer the user's actual question first and "
            "use earlier turns when useful. Use only "
            "the supplied evidence. Do not invent prices, news, probabilities, or trades. Treat "
            "the deterministic signal as authoritative and distinguish facts from interpretation. "
            "Do not include URLs, citations, source lists, or raw article links in the answer. "
            "State clearly when evidence is missing. Read RSI 30 or "
            "below as oversold, 70 or above as overbought, and values between them as neutral. "
            "Do not claim an indicator crossover unless the supplied active tokens explicitly "
            "name that crossover. For analysis questions, give a complete answer with these "
            "sections when relevant: View, Evidence, Risks, and Bottom line. Explain what the "
            "signal means, why the indicators support or weaken it, what could invalidate it, "
            "and what information is unavailable. Keep it readable and practical rather than "
            "ending mid-sentence. This is not financial advice."
        ),
    }]
    for message in (history or [])[-6:]:
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": str(content)})
    messages.append({
        "role": "user",
        "content": json.dumps({"question": task, "evidence": evidence}, default=str),
    })
    return messages