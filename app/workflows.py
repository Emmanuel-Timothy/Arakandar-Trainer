"""End-to-end automation workflows for the local workbench."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

from app.agent.orchestrator import ArakandarAgent
from app.agent.llm.local_model import LocalLanguageModel
from app.agent.storage import AgentStore
from app.data.market import LocalCsvProvider, MarketIngestor, MarketDataProvider
from app.data.news import NewsIngestor, RSSProvider, headlines_from_articles
import pandas as pd
from app.agent.ml.feature_pipeline import build_multi_ticker_dataset
from app.agent.ml.trainer import train_from_scratch


def run_market_intelligence_cycle(
    tickers: Iterable[str],
    data_dir: Path,
    model_path: Path,
    store: AgentStore,
    market_provider: MarketDataProvider | None = None,
    rss_feeds: Dict[str, str] | None = None,
    language_model: LocalLanguageModel | None = None,
    auto_train: bool = False,
) -> List[dict]:
    """Sync configured sources and run one explain-signal task per ticker."""
    provider = market_provider or LocalCsvProvider(data_dir)
    market_ingestor = MarketIngestor(provider, store)
    news_ingestor = NewsIngestor(RSSProvider(rss_feeds or {}), store)
    
    for ticker in tickers:
        period = "2y" if auto_train else "5d"
        market_ingestor.sync(ticker, period=period)
        news_ingestor.sync(ticker)

    if auto_train:
        ticker_to_df = {}
        for ticker in tickers:
            bars = store.latest_market_bars(ticker, limit=100000)
            if bars:
                df = pd.DataFrame(bars)
                df = df.rename(columns={"timestamp": "date"})
                df["date"] = pd.to_datetime(df["date"])
                ticker_to_df[ticker] = df
        
        if ticker_to_df:
            X_all, y_all, fwd_all, meta_all = build_multi_ticker_dataset(ticker_to_df)
            if len(X_all) > 100:
                print(f"Auto-training champion model on {len(X_all)} samples...")
                train_from_scratch(X_all, y_all, fwd_all, save_path=model_path)

    agent = ArakandarAgent(
        data_dir,
        model_path,
        store,
        language_model=language_model,
        rss_feeds=rss_feeds,
    )
    results = []

    for ticker in tickers:
        try:
            articles = store.latest_news(ticker, limit=10)
            headlines = headlines_from_articles(articles)
            results.append(agent.run("Explain the latest signal", ticker, headlines, news_articles=articles))
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Skipping analysis for %s: %s", ticker, exc)
    return results
