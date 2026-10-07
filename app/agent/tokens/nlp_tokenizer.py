"""
nlp_tokenizer.py
Financial NLP Subword Tokenizer & Token Impact Attributor
(Arakandar Dual Token System - Phase 2).

100% local / offline: no external API calls. Uses a hand-curated financial
sentiment lexicon (Loughran-McDonald style categories + market slang) to
score headline tokens with an impact weight in [-1.0, 1.0], then aggregates
them into a single composite News Sentiment Factor.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

FINANCIAL_LEXICON: Dict[str, float] = {
    "surge": 0.85, "soar": 0.88, "rally": 0.75, "beat": 0.78, "record": 0.82,
    "record_profit": 0.94, "upgrade": 0.7, "outperform": 0.68, "dividend_hike": 0.76,
    "buyback": 0.6, "expansion": 0.55, "breakthrough": 0.72, "accumulation": 0.5,
    "bullish": 0.65, "profit": 0.5, "growth": 0.5, "beat_estimates": 0.8,
    "plunge": -0.89, "crash": -0.93, "default": -0.92, "scandal": -0.95,
    "downgrade": -0.72, "miss": -0.7, "loss": -0.6, "bearish": -0.65,
    "layoff": -0.68, "lawsuit": -0.6, "fraud": -0.96, "bankruptcy": -0.97,
    "sell_off": -0.75, "distribution": -0.5, "warning": -0.55, "delay": -0.4,
    "correction": -0.45, "recession": -0.8, "inflation_fear": -0.5,
    "cut": -0.5, "slump": -0.7, "decline": -0.5, "underperform": -0.6,
}

_PHRASE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"record\s+profit", re.I), "record_profit"),
    (re.compile(r"beat(s)?\s+estimates?", re.I), "beat_estimates"),
    (re.compile(r"dividend\s+hike", re.I), "dividend_hike"),
    (re.compile(r"sell[- ]off", re.I), "sell_off"),
    (re.compile(r"inflation\s+fear", re.I), "inflation_fear"),
]

_TOKEN_RE = re.compile(r"[A-Za-z\u00c0-\u00ff]+")


@dataclass
class TokenImpact:
    token: str
    weight: float


@dataclass
class NewsTokenizationResult:
    tokens: List[TokenImpact] = field(default_factory=list)
    composite_sentiment: float = 0.0

    def top_catalysts(self, n: int = 3) -> List[TokenImpact]:
        return sorted(self.tokens, key=lambda t: abs(t.weight), reverse=True)[:n]

    def render_badges(self) -> str:
        return " ".join(f"[{t.token.upper()}: {t.weight:+.2f}]" for t in self.top_catalysts())


def _extract_phrases(text: str) -> Tuple[str, List[str]]:
    found = []
    working = text
    for pattern, canonical in _PHRASE_PATTERNS:
        if pattern.search(working):
            found.append(canonical)
            working = pattern.sub(" ", working)
    return working, found


def tokenize_headline(headline: str) -> NewsTokenizationResult:
    """Tokenizes a financial headline into subwords, matches against the
    financial sentiment lexicon, and computes a composite sentiment score."""
    remaining, phrase_hits = _extract_phrases(headline.lower())
    words = _TOKEN_RE.findall(remaining)

    impacts: List[TokenImpact] = []
    for canonical in phrase_hits:
        impacts.append(TokenImpact(token=canonical, weight=FINANCIAL_LEXICON[canonical]))
    for w in words:
        if w in FINANCIAL_LEXICON:
            impacts.append(TokenImpact(token=w, weight=FINANCIAL_LEXICON[w]))

    if impacts:
        composite = sum(t.weight for t in impacts) / len(impacts)
    else:
        composite = 0.0
    composite = max(-1.0, min(1.0, composite))
    return NewsTokenizationResult(tokens=impacts, composite_sentiment=composite)


def batch_sentiment_factor(headlines: List[str]) -> float:
    """Aggregates composite sentiment across multiple recent headlines into
    a single News Sentiment Factor feature for the ML pipeline."""
    if not headlines:
        return 0.0
    scores = [tokenize_headline(h).composite_sentiment for h in headlines]
    return sum(scores) / len(scores)
