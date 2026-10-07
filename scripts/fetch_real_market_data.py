"""Fetch, validate, and persist real historical OHLCV data from Yahoo Finance."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.data.market import YFinanceProvider, normalize_ohlcv


def _clean_provider_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove provider rows that cannot represent a tradable OHLCV bar."""
    cleaned = frame.copy()
    numeric = ["open", "high", "low", "close", "volume"]
    for column in numeric:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    valid = cleaned["date"].notna() & (cleaned[numeric].notna().all(axis=1))
    valid &= (cleaned[["open", "high", "low", "close"]] > 0).all(axis=1)
    valid &= cleaned["volume"] > 0
    valid &= cleaned["high"] >= cleaned[["open", "low", "close"]].max(axis=1)
    valid &= cleaned["low"] <= cleaned[["open", "high", "close"]].min(axis=1)
    return cleaned.loc[valid].copy(), int((~valid).sum())


def fetch_real_data(
    tickers: list[str],
    output_dir: Path,
    period: str = "max",
    interval: str = "1d",
    backup_dir: Path | None = None,
) -> dict[str, int]:
    provider = YFinanceProvider()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "source": "yfinance",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "period": period,
        "interval": interval,
        "tickers": {},
    }
    counts: dict[str, int] = {}
    for ticker in tickers:
        raw = provider.fetch(ticker, period, interval)
        cleaned, dropped = _clean_provider_frame(raw)
        normalized = normalize_ohlcv(cleaned, ticker, provider.name)
        destination = output_dir / f"{ticker.replace('.', '_')}_ohlcv.csv"
        if destination.exists() and backup_dir is not None:
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, backup_dir / destination.name)
        normalized.loc[:, ["date", "open", "high", "low", "close", "volume"]].to_csv(
            destination, index=False, date_format="%Y-%m-%dT%H:%M:%SZ"
        )
        counts[ticker] = len(normalized)
        manifest["tickers"][ticker] = {
            "rows": len(normalized),
            "first_date": normalized["date"].iloc[0].isoformat(),
            "last_date": normalized["date"].iloc[-1].isoformat(),
            "file": destination.name,
            "dropped_invalid_rows": dropped,
        }
    (output_dir / "market_data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch real historical market data")
    parser.add_argument("--ticker", action="append", required=True)
    parser.add_argument("--period", default="max")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--backup-dir", type=Path, default=Path("data/backups/synthetic"))
    args = parser.parse_args()
    try:
        counts = fetch_real_data(args.ticker, args.output_dir, args.period, args.interval, args.backup_dir)
    except (RuntimeError, ValueError, OSError) as exc:
        parser.error(str(exc))
    for ticker, count in counts.items():
        print(f"{ticker}: wrote {count} real {args.interval} bars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
