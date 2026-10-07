"""
feature_pipeline.py
Multi-Factor Feature Matrix builder (Arakandar ML Engine - Phase 3).

Combines technical numerical indicators, symbolic Market State Tokens, and
News Sentiment Token scores into a single feature matrix ready for the
Gradient Boosting classifier. Also derives the classification target label:
BUY / HOLD / SELL based on forward returns.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from app.agent.tokens.market_tokenizer import compute_technical_features, tokenize_dataframe, MARKET_TOKEN_VOCAB

NUMERIC_FEATURES: List[str] = [
    "rsi14", "macd", "macd_signal", "macd_diff", "ema_spread_20_50",
    "atr_pct", "obv_slope", "vol_ratio", "bb_width", "bb_width_pctile",
    "rolling_max_5", "rolling_min_5", "mom_5", "sma_10"
]
TOKEN_FEATURES: List[str] = [t for t in MARKET_TOKEN_VOCAB if t != "TK_NONE"]
NEWS_FEATURE: str = "news_sentiment_factor"

ALL_FEATURES: List[str] = NUMERIC_FEATURES + TOKEN_FEATURES + [NEWS_FEATURE]

LABEL_MAP = {"SELL": 0, "HOLD": 1, "BUY": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def make_labels(df: pd.DataFrame, horizon: int = 5, up_thresh: float = 1.5, down_thresh: float = -1.5):
    """Directional trend multi-class target using Triple Barrier Method:
    BUY if forward return over `horizon` bars exceeds ATR-scaled up_thresh,
    SELL if below down_thresh, else HOLD."""
    fwd_return = df["close"].shift(-horizon) / df["close"] - 1.0
    labels = pd.Series("HOLD", index=df.index)
    if "atr_pct" in df.columns:
        up_barrier = df["atr_pct"] * up_thresh
        down_barrier = df["atr_pct"] * down_thresh
    else:
        up_barrier = 0.02
        down_barrier = -0.02

    labels.loc[fwd_return > up_barrier] = "BUY"
    labels.loc[fwd_return < down_barrier] = "SELL"
    return labels, fwd_return


def build_feature_matrix(
    raw_ohlcv: pd.DataFrame,
    news_sentiment_series: Optional[pd.Series] = None,
    horizon: int = 5,
    up_thresh: float = 1.5,
    down_thresh: float = -1.5,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Builds the full multi-factor feature matrix + labels + forward returns
    for a single ticker's OHLCV history. Vectorized end-to-end."""
    df = raw_ohlcv.sort_values("date").reset_index(drop=True)
    feats = compute_technical_features(df)
    feats = tokenize_dataframe(feats)

    if news_sentiment_series is not None:
        feats[NEWS_FEATURE] = news_sentiment_series.reindex(feats.index).fillna(0.0)
    else:
        feats[NEWS_FEATURE] = 0.0

    labels, fwd_return = make_labels(feats, horizon, up_thresh, down_thresh)

    valid = feats[ALL_FEATURES].notna().all(axis=1) & fwd_return.notna()
    X = feats.loc[valid, ALL_FEATURES].reset_index(drop=True)
    y = labels.loc[valid].reset_index(drop=True)
    fwd = fwd_return.loc[valid].reset_index(drop=True)
    meta = feats.loc[valid, ["date", "close", "active_token_names"]].reset_index(drop=True)
    return X, y, fwd, meta


def build_multi_ticker_dataset(
    ticker_to_df: dict,
    ticker_to_news: Optional[dict] = None,
    horizon: int = 5,
    up_thresh: float = 1.5,
    down_thresh: float = -1.5,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Concatenates feature matrices across multiple tickers into one
    training-ready dataset. Adds a 'ticker' column to the metadata."""
    Xs, ys, fwds, metas = [], [], [], []
    ticker_to_news = ticker_to_news or {}
    for ticker, df in ticker_to_df.items():
        news_series = ticker_to_news.get(ticker)
        X, y, fwd, meta = build_feature_matrix(df, news_series, horizon, up_thresh, down_thresh)
        meta = meta.copy()
        meta["ticker"] = ticker
        Xs.append(X); ys.append(y); fwds.append(fwd); metas.append(meta)
    X_all = pd.concat(Xs, ignore_index=True)
    y_all = pd.concat(ys, ignore_index=True)
    fwd_all = pd.concat(fwds, ignore_index=True)
    meta_all = pd.concat(metas, ignore_index=True)
    # Keep multi-ticker samples in chronological order so temporal validation
    # never trains on a later date and validates on an earlier date.
    order = meta_all["date"].sort_values(kind="stable").index
    X_all = X_all.iloc[order].reset_index(drop=True)
    y_all = y_all.iloc[order].reset_index(drop=True)
    fwd_all = fwd_all.iloc[order].reset_index(drop=True)
    meta_all = meta_all.iloc[order].reset_index(drop=True)
    return X_all, y_all, fwd_all, meta_all
