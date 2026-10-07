"""Market data adapters and validated, idempotent ingestion."""
from __future__ import annotations

import contextlib
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from app.agent.storage import AgentStore

REQUIRED_COLUMNS = ("date", "open", "high", "low", "close", "volume")


class MarketDataProvider(Protocol):
    name: str

    def fetch(self, ticker: str, period: str = "5d", interval: str = "1d") -> pd.DataFrame:
        ...


@dataclass
class LocalCsvProvider:
    data_dir: Path
    name: str = "local_csv"

    def fetch(self, ticker: str, period: str = "5d", interval: str = "1d") -> pd.DataFrame:
        path = Path(self.data_dir) / f"{ticker.replace('.', '_')}_ohlcv.csv"
        if not path.exists():
            raise FileNotFoundError(f"No local OHLCV file for {ticker}: {path}")
        return pd.read_csv(path, parse_dates=["date"])


class YFinanceProvider:
    name = "yfinance"

    def fetch(self, ticker: str, period: str = "5d", interval: str = "1d") -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("yfinance is not installed; install requirements.txt") from exc
        # yfinance can print provider warnings directly to stderr. Keep the CLI
        # readable and let MarketIngestor report one concise failure instead.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            frame = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
        if frame is None or frame.empty:
            return pd.DataFrame()
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        frame = frame.reset_index()
        frame.columns = [str(c).lower() for c in frame.columns]
        return frame


def normalize_ohlcv(frame: pd.DataFrame, ticker: str, source: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise ValueError(f"{ticker} data is empty")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{ticker} data missing required columns: {missing}")

    normalized = frame.loc[:, REQUIRED_COLUMNS].copy()
    normalized["date"] = pd.to_datetime(normalized["date"], utc=True, errors="coerce")
    for column in REQUIRED_COLUMNS[1:]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized.dropna(subset=list(REQUIRED_COLUMNS))
    if normalized.empty:
        raise ValueError(f"{ticker} data contains no valid OHLCV rows")
    price_columns = ["open", "high", "low", "close"]
    if (normalized[price_columns] <= 0).any().any():
        raise ValueError(f"{ticker} data contains non-positive prices")
    normalized = normalized[normalized["volume"] > 0].copy()
    if normalized.empty:
        raise ValueError(f"{ticker} data contains no rows with positive volume")
    if (normalized["high"] < normalized[["open", "low", "close"]].max(axis=1)).any():
        raise ValueError(f"{ticker} data contains highs below an OHLC value")
    if (normalized["low"] > normalized[["open", "high", "close"]].min(axis=1)).any():
        raise ValueError(f"{ticker} data contains lows above an OHLC value")
    normalized = normalized.drop_duplicates(subset=["date"]).sort_values("date")
    normalized["ticker"] = ticker
    normalized["source"] = source
    return normalized.reset_index(drop=True)


class MarketIngestor:
    def __init__(self, provider: MarketDataProvider, store: AgentStore):
        self.provider = provider
        self.store = store

    def sync(self, ticker: str, period: str = "5d", interval: str = "1d") -> int:
        try:
            raw = self.provider.fetch(ticker, period=period, interval=interval)
            normalized = normalize_ohlcv(raw, ticker, self.provider.name)
            bars = normalized.rename(columns={"date": "timestamp"}).to_dict("records")
            for bar in bars:
                bar["timestamp"] = bar["timestamp"].isoformat()
            return self.store.upsert_market_bars(bars)
        except Exception as exc:
            logging.getLogger(__name__).warning("Failed to sync market data for %s: %s", ticker, exc)
            return 0