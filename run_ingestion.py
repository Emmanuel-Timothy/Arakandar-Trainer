"""Poll and persist market data for configured tickers."""
from __future__ import annotations

import argparse
from pathlib import Path

from app.agent.storage import AgentStore
from app.data.market import LocalCsvProvider, MarketIngestor, YFinanceProvider


def main() -> int:
    parser = argparse.ArgumentParser(description="Arakandar market data ingestion")
    parser.add_argument("--ticker", action="append", required=True, help="Ticker symbol, repeatable")
    parser.add_argument("--provider", choices=["csv", "yfinance"], default="csv")
    parser.add_argument("--period", default="5d")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--state", default="data/agent_state.sqlite3")
    args = parser.parse_args()

    provider = LocalCsvProvider(Path(args.data_dir)) if args.provider == "csv" else YFinanceProvider()
    ingestor = MarketIngestor(provider, AgentStore(Path(args.state)))
    for ticker in args.ticker:
        count = ingestor.sync(ticker, period=args.period, interval=args.interval)
        print(f"{ticker}: stored {count} bars from {provider.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
