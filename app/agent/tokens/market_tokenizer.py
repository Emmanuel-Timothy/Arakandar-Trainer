"""
market_tokenizer.py
Symbolic Market State Tokenizer (Arakandar Dual Token System - Phase 2).

Quantizes continuous technical indicators into discrete, human-readable
Market State Tokens. These tokens are fed as categorical features into the
Gradient Boosting engine and are also surfaced verbatim in the terminal UI
for explainability (e.g. "TK_RSI_OVERSOLD", "TK_GOLDEN_CROSS_20_50").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

MARKET_TOKEN_VOCAB: Dict[str, int] = {
    "TK_NONE": 0,
    "TK_RSI_OVERSOLD": 1,
    "TK_RSI_BULL_MOMENTUM": 2,
    "TK_RSI_NEUTRAL": 3,
    "TK_RSI_OVERBOUGHT": 4,
    "TK_MACD_BULL_CROSS": 5,
    "TK_MACD_BEAR_CROSS": 6,
    "TK_PRICE_ABOVE_EMA50": 7,
    "TK_PRICE_BELOW_EMA50": 8,
    "TK_GOLDEN_CROSS_20_50": 9,
    "TK_DEATH_CROSS_20_50": 10,
    "TK_ACCUMULATION_STRONG": 11,
    "TK_DISTRIBUTION_HIGH": 12,
    "TK_VOLUME_SPIKE_2X": 13,
    "TK_VOLUME_DRY_UP": 14,
    "TK_ATR_HIGH_VOLATILITY": 15,
    "TK_ATR_LOW_VOLATILITY": 16,
    "TK_BOLLINGER_SQUEEZE": 17,
    "TK_BOLLINGER_BREAKOUT_UP": 18,
    "TK_BOLLINGER_BREAKOUT_DOWN": 19,
}
TOKEN_ID_TO_NAME = {v: k for k, v in MARKET_TOKEN_VOCAB.items()}


@dataclass
class MarketTokenBundle:
    active_tokens: List[str] = field(default_factory=list)
    token_ids: List[int] = field(default_factory=list)

    def as_dict(self) -> Dict[str, int]:
        return {name: 1 for name in self.active_tokens}


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _macd(close: pd.Series):
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd_line = ema12 - ema26
    signal_line = _ema(macd_line, 9)
    return macd_line, signal_line


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff().fillna(0))
    return (direction * df["volume"]).cumsum()


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized computation of all raw numerical indicators used by both
    the tokenizer and the feature pipeline."""
    out = df.copy()
    out["rsi14"] = _rsi(out["close"], 14)
    macd_line, signal_line = _macd(out["close"])
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_diff"] = macd_line - signal_line
    out["ema20"] = _ema(out["close"], 20)
    out["ema50"] = _ema(out["close"], 50)
    out["ema_spread_20_50"] = (out["ema20"] - out["ema50"]) / out["ema50"]
    out["atr14"] = _atr(out, 14)
    out["atr_pct"] = out["atr14"] / out["close"]
    out["obv"] = _obv(out)
    out["obv_slope"] = out["obv"].diff(5)
    out["vol_sma20"] = out["volume"].rolling(20, min_periods=5).mean()
    out["vol_ratio"] = out["volume"] / out["vol_sma20"].replace(0, np.nan)
    bb_mid = out["close"].rolling(20, min_periods=5).mean()
    bb_std = out["close"].rolling(20, min_periods=5).std()
    out["bb_upper"] = bb_mid + 2 * bb_std
    out["bb_lower"] = bb_mid - 2 * bb_std
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / bb_mid.replace(0, np.nan)
    out["bb_width_pctile"] = out["bb_width"].rolling(100, min_periods=20).rank(pct=True)
    out["rolling_max_5"] = out["close"].rolling(5, min_periods=1).max()
    out["rolling_min_5"] = out["close"].rolling(5, min_periods=1).min()
    out["mom_5"] = out["close"] / out["close"].shift(5).bfill() - 1.0
    out["sma_10"] = out["close"].rolling(10, min_periods=1).mean()
    return out


def tokenize_row(row: pd.Series) -> MarketTokenBundle:
    """Quantize a single row of computed technical features into discrete
    symbolic Market State Tokens."""
    tokens: List[str] = []

    rsi = row.get("rsi14", 50)
    if rsi <= 30:
        tokens.append("TK_RSI_OVERSOLD")
    elif rsi >= 70:
        tokens.append("TK_RSI_OVERBOUGHT")
    elif rsi >= 55:
        tokens.append("TK_RSI_BULL_MOMENTUM")
    else:
        tokens.append("TK_RSI_NEUTRAL")

    if row.get("macd_diff", 0) > 0 and row.get("macd", 0) > row.get("macd_signal", 0):
        tokens.append("TK_MACD_BULL_CROSS")
    elif row.get("macd_diff", 0) < 0:
        tokens.append("TK_MACD_BEAR_CROSS")

    if row.get("close", 0) > row.get("ema50", 0):
        tokens.append("TK_PRICE_ABOVE_EMA50")
    else:
        tokens.append("TK_PRICE_BELOW_EMA50")

    spread = row.get("ema_spread_20_50", 0)
    if spread > 0.015:
        tokens.append("TK_GOLDEN_CROSS_20_50")
    elif spread < -0.015:
        tokens.append("TK_DEATH_CROSS_20_50")

    vol_ratio = row.get("vol_ratio", 1.0)
    obv_slope = row.get("obv_slope", 0)
    if vol_ratio >= 2.0:
        tokens.append("TK_VOLUME_SPIKE_2X")
    elif vol_ratio <= 0.5:
        tokens.append("TK_VOLUME_DRY_UP")

    if obv_slope > 0 and vol_ratio > 1.2:
        tokens.append("TK_ACCUMULATION_STRONG")
    elif obv_slope < 0 and vol_ratio > 1.2:
        tokens.append("TK_DISTRIBUTION_HIGH")

    atr_pct = row.get("atr_pct", 0.02)
    if atr_pct >= 0.035:
        tokens.append("TK_ATR_HIGH_VOLATILITY")
    elif atr_pct <= 0.012:
        tokens.append("TK_ATR_LOW_VOLATILITY")

    bb_pctile = row.get("bb_width_pctile", 0.5)
    close = row.get("close", 0)
    bb_upper = row.get("bb_upper", np.inf)
    bb_lower = row.get("bb_lower", -np.inf)
    if pd.notna(bb_pctile) and bb_pctile <= 0.15:
        tokens.append("TK_BOLLINGER_SQUEEZE")
    if close > bb_upper:
        tokens.append("TK_BOLLINGER_BREAKOUT_UP")
    elif close < bb_lower:
        tokens.append("TK_BOLLINGER_BREAKOUT_DOWN")

    ids = [MARKET_TOKEN_VOCAB[t] for t in tokens if t in MARKET_TOKEN_VOCAB]
    return MarketTokenBundle(active_tokens=tokens, token_ids=ids)


def tokenize_dataframe(df_with_features: pd.DataFrame) -> pd.DataFrame:
    """Applies tokenize_row across the whole dataframe and returns a
    dataframe with one-hot encoded token columns ready for the ML pipeline."""
    bundles = df_with_features.apply(tokenize_row, axis=1)
    token_cols = pd.DataFrame([b.as_dict() for b in bundles], index=df_with_features.index)
    for name in MARKET_TOKEN_VOCAB:
        if name not in token_cols.columns:
            token_cols[name] = 0
    token_cols = token_cols[list(MARKET_TOKEN_VOCAB.keys())].fillna(0).astype(int)
    result = pd.concat([df_with_features, token_cols], axis=1)
    result["active_token_names"] = bundles.apply(lambda b: ",".join(b.active_tokens))
    return result
