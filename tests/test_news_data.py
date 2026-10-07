"""Tests for RSS parsing and idempotent news storage."""
from pathlib import Path

from app.agent.storage import AgentStore
from app.data.news import NewsArticle, NewsIngestor, parse_rss


RSS = b"""<rss><channel>
<item><title>BBCA reports record profit</title><link>https://example.test/1</link><pubDate>Sat, 05 Sep 2026 10:00:00 GMT</pubDate></item>
<item><title>BBCA shares plunge after warning</title><link>https://example.test/2</link><pubDate>Sat, 05 Sep 2026 09:00:00 GMT</pubDate></item>
</channel></rss>"""

ATOM = b"""<feed xmlns=\"http://www.w3.org/2005/Atom\">
<entry><title>BBCA launches new service</title><link href=\"https://example.test/atom\"/>
<updated>2026-09-05T11:00:00Z</updated></entry>
</feed>"""


def test_parse_rss_normalizes_and_scores_articles():
    articles = parse_rss(RSS, "BBCA.JK", "test")

    assert len(articles) == 2
    assert articles[0].ticker == "BBCA.JK"
    assert articles[0].sentiment > 0
    assert articles[1].sentiment < 0


def test_parse_atom_normalizes_link_and_iso_timestamp():
    articles = parse_rss(ATOM, "BBCA.JK", "atom")

    assert len(articles) == 1
    assert articles[0].url == "https://example.test/atom"
    assert articles[0].published_at == "2026-09-05T11:00:00+00:00"


def test_news_storage_is_idempotent(tmp_path: Path):
    store = AgentStore(tmp_path / "state.sqlite3")
    articles = [article.__dict__ for article in parse_rss(RSS, "BBCA.JK", "test")]

    assert store.upsert_news_articles(articles) == 2
    assert store.upsert_news_articles(articles) == 2
    assert len(store.latest_news("BBCA.JK")) == 2


class FailingProvider:
    def fetch(self, ticker: str):
        raise OSError("RSS temporarily unavailable")


def test_failed_refresh_preserves_cached_news(tmp_path: Path):
    store = AgentStore(tmp_path / "state.sqlite3")
    cached = NewsArticle(
        article_id="cached",
        ticker="BBCA.JK",
        published_at="2026-09-05T10:00:00+00:00",
        title="Cached market update",
        url="https://example.test/cached",
        source="cache",
        sentiment=0.2,
    )
    store.upsert_news_articles([cached.__dict__])

    assert NewsIngestor(FailingProvider(), store).sync("BBCA.JK") == 0
    assert store.latest_news("BBCA.JK")[0]["title"] == "Cached market update"