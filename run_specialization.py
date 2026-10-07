"""Generate grounded analyst data and train the Arakandar 3B adapter."""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.fetch_real_market_data import fetch_real_data
from scripts.generate_analyst_training_data import generate
from scripts.train_arakandar_lora import train, load_examples


def main() -> int:
    parser = argparse.ArgumentParser(description="Automate Arakandar 3B analyst specialization")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--state", type=Path, default=Path("data/agent_state.sqlite3"))
    parser.add_argument("--signal-model", type=Path, default=Path("models/champion_lgbm.pkl"))
    parser.add_argument("--dataset", type=Path, default=Path("data/llm/train.jsonl"))
    parser.add_argument("--base-model", type=Path, default=Path("models/base/arakandar-3b"))
    parser.add_argument("--output", type=Path, default=Path("models/adapters/arakandar-3b-lora"))
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--ticker", action="append")
    parser.add_argument("--refresh-real-data", action="store_true", help="Fetch real Yahoo Finance history before training")
    args = parser.parse_args()
    tickers = args.ticker or ["BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK"]
    if args.refresh_real_data:
        counts = fetch_real_data(tickers, args.data_dir, backup_dir=Path("data/backups/synthetic"))
        print(f"Refreshed real market data: {counts}")
    count = generate(args.dataset, args.data_dir, args.state, args.signal_model, tickers)
    print(f"Generated {count} grounded examples")
    if len(load_examples(args.dataset)) < 8:
        parser.error("Generated dataset did not meet the eight-example minimum")
    train(args)
    print(f"Adapter ready at {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
