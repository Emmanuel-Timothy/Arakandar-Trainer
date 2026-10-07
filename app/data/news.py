"""RSS news ingestion with normalization, deduplication, and sentiment."""
from __future__ import annotations

import hashlib
import re
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, Iterable, List
from urllib.parse import quote_plus

from app.agent.storage import AgentStore
from app.agent.tokens.nlp_tokenizer import tokenize_headline


@dataclass
class NewsArticle:
    article_id: str
    ticker: str
    published_at: str
    title: str
    url: str
    source: str
    sentiment: float


def _text(element: ET.Element | None) -> str:
    return " ".join("".join(element.itertext()).split()) if element is not None else ""


def _child(element: ET.Element, name: str) -> ET.Element | None:
    child = element.find(name)
    return child if child is not None else element.find(f".//{{*}}{name}")


def _published(value: str) -> str:
    try:
        parsed = parsedate_to_datetime(value).astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
        except (TypeError, ValueError, OverflowError):
            parsed = datetime.now(timezone.utc)
    return parsed.isoformat()


def parse_rss(xml: bytes, ticker: str, source: str) -> List[NewsArticle]:
    root = ET.fromstring(xml)
    articles = []
    entries = root.findall(".//item") or root.findall(".//{*}entry")
    for item in entries:
        title = _text(_child(item, "title"))
        link = _child(item, "link")
        url = (link.get("href", "") if link is not None else "") or _text(link)
        if not title or not url:
            continue
        published = (
            _text(_child(item, "pubDate"))
            or _text(_child(item, "published"))
            or _text(_child(item, "updated"))
        )
        published_at = _published(published)
        article_id = hashlib.sha256(f"{url}|{title}".encode("utf-8")).hexdigest()
        articles.append(NewsArticle(
            article_id=article_id,
            ticker=ticker,
            published_at=published_at,
            title=title,
            url=url,
            source=source,
            sentiment=round(tokenize_headline(title).composite_sentiment, 4),
        ))
    return articles


@dataclass
class RSSProvider:
    feeds: Dict[str, str]
    source: str = "rss"
    timeout_seconds: int = 10

    def fetch(self, ticker: str) -> List[NewsArticle]:
        url = self.feeds.get(ticker)
        if not url:
            return []
        request = urllib.request.Request(url, headers={"User-Agent": "Arakandar/1.0"})
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return parse_rss(response.read(), ticker, self.source)


def google_news_feed(ticker: str) -> str:
    """Build a public Google News RSS search URL for one market symbol."""
    query = quote_plus(f'"{ticker}" stock')
    return f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"


class NewsIngestor:
    def __init__(self, provider: RSSProvider, store: AgentStore):
        self.provider = provider
        self.store = store

    def sync(self, ticker: str) -> int:
        try:
            articles = self.provider.fetch(ticker)
        except (OSError, ET.ParseError, ValueError):
            # Existing SQLite articles remain available to the workflow as a cache.
            return 0
        return self.store.upsert_news_articles([article.__dict__ for article in articles])


def headlines_from_articles(articles: Iterable[Dict[str, str]]) -> List[str]:
    return [article["title"] for article in articles if article.get("title")]