"""Run one complete market intelligence automation cycle."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from app.agent.storage import AgentStore
from app.agent.llm.local_model import TransformersLocalModel
from app.scheduler import LocalScheduler
from app.workflows import run_market_intelligence_cycle
from app.data.market import YFinanceProvider, LocalCsvProvider


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Arakandar market intelligence automation")
    parser.add_argument("--ticker", action="append", default=[], help="Ticker symbol, repeatable")
    parser.add_argument("--watchlist", type=str, help="Path to a text file containing tickers (one per line)")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--state", default="data/agent_state.sqlite3")
    parser.add_argument("--model", default="models/champion_lgbm.pkl")
    parser.add_argument(
        "--rss-feed",
        action="append",
        default=[],
        metavar="TICKER=URL",
        help="RSS feed for a ticker, repeatable (for example BBCA.JK=https://example/feed.xml)",
    )
    parser.add_argument(
        "--llm-model",
        help="Local Transformers model directory or repository id; omitted disables local prose generation",
    )
    parser.add_argument("--llm-device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--watch", action="store_true", help="Repeat the cycle until interrupted")
    parser.add_argument("--interval-seconds", type=float, default=300.0)
    parser.add_argument("--auto-train", action="store_true", help="Automatically retrain the champion model on the latest data")
    parser.add_argument("--provider", choices=["csv", "yfinance"], default="yfinance")
    args = parser.parse_args()
    
    # Preserve order from the watchlist file by using a list and deduplicating manually
    seen = set()
    tickers = []
    for t in args.ticker:
        if t not in seen:
            seen.add(t)
            tickers.append(t)
    if args.watchlist:
        try:
            with open(args.watchlist, "r") as f:
                for line in f:
                    t = line.strip()
                    if t and not t.startswith("#") and t not in seen:
                        seen.add(t)
                        tickers.append(t)
        except Exception as e:
            parser.error(f"Failed to read watchlist {args.watchlist}: {e}")

    if not tickers:
        parser.error("At least one --ticker or a valid --watchlist must be provided.")

    rss_feeds = {}
    for item in args.rss_feed:
        ticker, separator, url = item.partition("=")
        if not separator or not ticker or not url:
            parser.error("--rss-feed must use TICKER=URL")
        rss_feeds[ticker] = url
    language_model = (
        TransformersLocalModel(args.llm_model, device=args.llm_device) if args.llm_model else None
    )

    def run_cycle() -> None:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"\n{'='*60}", flush=True)
        print(f"[{now}] Starting cycle for {len(tickers)} tickers...", flush=True)
        print(f"{'='*60}", flush=True)
        market_provider = YFinanceProvider() if args.provider == "yfinance" else LocalCsvProvider(Path(args.data_dir))
        results = run_market_intelligence_cycle(
            tickers,
            Path(args.data_dir),
            Path(args.model),
            AgentStore(Path(args.state)),
            market_provider=market_provider,
            rss_feeds=rss_feeds,
            language_model=language_model,
            auto_train=args.auto_train,
        )
        print(json.dumps(results, indent=2, default=str), flush=True)
        if args.watch:
            next_run = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            print(f"\n[{next_run}] Cycle done. Next run in {int(args.interval_seconds)}s (Ctrl+C to stop)", flush=True)

    if args.watch:
        scheduler = LocalScheduler()
        scheduler.add_job("market_intelligence_cycle", args.interval_seconds, run_cycle)
        try:
            scheduler.run_forever()
        except KeyboardInterrupt:
            scheduler.stop()
    else:
        run_cycle()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
