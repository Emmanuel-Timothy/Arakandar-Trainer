"""tests/test_tokens.py — Dual Token System unit tests."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from app.agent.tokens.nlp_tokenizer import tokenize_headline, batch_sentiment_factor
from app.agent.tokens.market_tokenizer import compute_technical_features, tokenize_dataframe, MARKET_TOKEN_VOCAB
from app.agent.tokens.session_tokens import SessionTokenLedger


def test_nlp_tokenizer_bounds():
    r = tokenize_headline("BBCA posts record profit and dividend hike, shares surge")
    assert -1.0 <= r.composite_sentiment <= 1.0
    assert r.composite_sentiment > 0

def test_nlp_tokenizer_negative():
    r = tokenize_headline("Company faces fraud scandal, shares plunge after default warning")
    assert r.composite_sentiment < 0

def test_batch_sentiment_factor_empty():
    assert batch_sentiment_factor([]) == 0.0

def test_market_tokenizer_categorical_ids():
    dates = pd.bdate_range("2023-01-01", periods=120)
    df = pd.DataFrame({
        "date": dates,
        "open": 100 + pd.Series(range(120)) * 0.1,
        "high": 101 + pd.Series(range(120)) * 0.1,
        "low": 99 + pd.Series(range(120)) * 0.1,
        "close": 100 + pd.Series(range(120)) * 0.1,
        "volume": 1_000_000,
    })
    feats = compute_technical_features(df)
    tokenized = tokenize_dataframe(feats)
    for name in MARKET_TOKEN_VOCAB:
        if name != "TK_NONE":
            assert name in tokenized.columns
    assert tokenized[list(MARKET_TOKEN_VOCAB.keys())[1:]].isin([0, 1]).all().all()

def test_session_token_ledger_deducts_and_tracks_burst():
    ledger = SessionTokenLedger()
    ledger.deduct("full_ml_decision")
    assert ledger.balance == 5000 - 150
    for _ in range(30):
        ledger.deduct("full_ml_decision")
    assert ledger.burst_mode is True
