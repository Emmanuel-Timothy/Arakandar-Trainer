"""
run_training.py
CLI entry point to train the Arakandar decision engine from scratch,
continue training on new data, or fine-tune on a specific ticker.

USAGE
-----
Train from scratch on all CSVs in data/raw/:
    python run_training.py --mode scratch

Continue training the existing champion on new data (e.g. after appending
fresh daily bars to data/raw/*.csv):
    python run_training.py --mode continue --rounds 100

Fine-tune the champion onto a single ticker (e.g. a new IPO or a ticker
you want the model to specialize on):
    python run_training.py --mode finetune --ticker BBCA.JK --rounds 50

Add your own data:
    Drop any CSV with columns [date, open, high, low, close, volume] into
    data/raw/<TICKER>_ohlcv.csv, then re-run with --mode scratch or
    --mode continue. You can source real data from yfinance, IDX/KSEI
    exports, Perplexity Finance, or any broker export -- the pipeline only
    needs those 6 columns.
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import pandas as pd

from app.agent.ml.feature_pipeline import build_multi_ticker_dataset, build_feature_matrix
from app.agent.ml.trainer import (
    train_from_scratch, continue_training, fine_tune, CHAMPION_PATH, BACKEND
)
from app.agent.storage import AgentStore
from app.model_registry import ModelRegistry

DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"


def load_all_tickers() -> dict:
    ticker_to_df = {}
    for path in glob.glob(str(DATA_DIR / "*_ohlcv.csv")):
        ticker = os.path.basename(path).replace("_ohlcv.csv", "").replace("_", ".")
        df = pd.read_csv(path, parse_dates=["date"])
        ticker_to_df[ticker] = df
    if not ticker_to_df:
        raise FileNotFoundError(
            f"No OHLCV CSVs found in {DATA_DIR}. Add files named "
            f"<TICKER>_ohlcv.csv with columns [date,open,high,low,close,volume]."
        )
    return ticker_to_df


def main():
    parser = argparse.ArgumentParser(description="Arakandar model trainer")
    parser.add_argument("--mode", choices=["scratch", "continue", "finetune"], default="scratch")
    parser.add_argument("--ticker", default=None, help="Required for --mode finetune; optional filter for others")
    parser.add_argument("--rounds", type=int, default=100, help="Additional/fine-tune rounds")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--n-estimators", type=int, default=300, help="Rounds for from-scratch training")
    parser.add_argument("--max-depth", type=int, default=6, help="Maximum tree depth for from-scratch training")
    parser.add_argument("--horizon", type=int, default=5, help="Forward-looking bars for labeling")
    parser.add_argument("--out", default=str(CHAMPION_PATH), help="Output model path")
    parser.add_argument(
        "--register-candidate",
        action="store_true",
        help="Save to the requested path and register it as a candidate instead of approving it",
    )
    parser.add_argument("--state", default="data/agent_state.sqlite3", help="Agent state database")
    args = parser.parse_args()

    print(f"[Arakandar Trainer] Backend: {BACKEND}")
    ticker_to_df = load_all_tickers()
    print(f"[Arakandar Trainer] Loaded tickers: {list(ticker_to_df.keys())}")

    output_path = Path(args.out)
    if args.register_candidate and output_path.resolve() == Path(CHAMPION_PATH).resolve():
        output_path = output_path.with_name(f"{output_path.stem}.candidate{output_path.suffix}")

    if args.mode == "scratch":
        X, y, fwd, meta = build_multi_ticker_dataset(ticker_to_df, horizon=args.horizon)
        print(f"[Arakandar Trainer] Dataset shape: {X.shape}, label distribution:\n{y.value_counts()}")
        model, scaler, report = train_from_scratch(
            X, y, fwd, n_estimators=args.n_estimators,
            learning_rate=args.lr or 0.05, max_depth=args.max_depth, save_path=output_path,
        )
        print(report.to_json())

    elif args.mode == "continue":
        X, y, fwd, meta = build_multi_ticker_dataset(ticker_to_df, horizon=args.horizon)
        model, scaler, report = continue_training(
            Path(args.out), X, y, fwd,
            additional_rounds=args.rounds, learning_rate=args.lr, save_path=output_path,
        )
        print(report.to_json())

    elif args.mode == "finetune":
        if not args.ticker:
            raise ValueError("--ticker is required for --mode finetune")
        if args.ticker not in ticker_to_df:
            raise ValueError(f"Ticker {args.ticker} not found in data/raw/. Available: {list(ticker_to_df.keys())}")
        X, y, fwd, meta = build_feature_matrix(ticker_to_df[args.ticker], horizon=args.horizon)
        model, scaler, report = fine_tune(
            Path(args.out), X, y, fwd,
            fine_tune_rounds=args.rounds, fine_tune_lr=args.lr or 0.01, save_path=output_path,
        )
        print(report.to_json())

    if args.register_candidate:
        registry = ModelRegistry(AgentStore(Path(args.state)), CHAMPION_PATH)
        version_id = registry.register_candidate(output_path, {
            "val_accuracy": report.val_accuracy,
            "val_f1_macro": report.val_f1_macro,
            "val_log_loss": report.val_log_loss,
            "net_pnl_score": report.net_pnl_score,
        })
        print(f"[Arakandar Trainer] Candidate registered: {version_id}")
    else:
        print(f"[Arakandar Trainer] Model saved to: {output_path}")


if __name__ == "__main__":
    main()
