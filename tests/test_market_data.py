"""Tests for normalized and idempotent market ingestion."""
from pathlib import Path

import pandas as pd
import pytest

from app.agent.storage import AgentStore
from app.data.market import MarketIngestor, normalize_ohlcv
from run_chat import _available_tickers, _comparison_reason, _evidence_reason, _extract_tickers, _global_watchlist, _indonesia_watchlist, _ticker_from_question


def _bars():
    return pd.DataFrame({
        "date": ["2026-01-02", "2026-01-03", "2026-01-03"],
        "open": [100, 101, 101],
        "high": [102, 103, 103],
        "low": [99, 100, 100],
        "close": [101, 102, 102],
        "volume": [1000, 1200, 1200],
    })


class FakeProvider:
    name = "fake"

    def fetch(self, ticker: str, period: str = "5d", interval: str = "1d") -> pd.DataFrame:
        return _bars()


def test_normalization_deduplicates_and_adds_metadata():
    result = normalize_ohlcv(_bars(), "BBCA.JK", "fake")

    assert len(result) == 2
    assert result["date"].dt.tz is not None
    assert result["ticker"].tolist() == ["BBCA.JK", "BBCA.JK"]


def test_ingestion_is_idempotent(tmp_path: Path):
    store = AgentStore(tmp_path / "state.sqlite3")
    ingestor = MarketIngestor(FakeProvider(), store)

    assert ingestor.sync("BBCA.JK") == 2
    assert ingestor.sync("BBCA.JK") == 2
    assert len(store.latest_market_bars("BBCA.JK")) == 2


def test_normalization_rejects_invalid_prices():
    invalid = _bars()
    invalid.loc[0, "close"] = -1

    with pytest.raises(ValueError, match="non-positive prices"):
        normalize_ohlcv(invalid, "BBCA.JK", "fake")


def test_normalization_drops_zero_volume_rows():
    invalid = _bars()
    invalid.loc[0, "volume"] = 0

    result = normalize_ohlcv(invalid, "BBCA.JK", "fake")

    assert len(result) == 1
    assert (result["volume"] > 0).all()


def test_ticker_detection_uses_store_path_and_recognizes_live_symbols(tmp_path: Path):
    store = AgentStore(tmp_path / "state.sqlite3")

    assert hasattr(store, "_db_path")
    assert _ticker_from_question("What is AAPL doing right now?", store, "BBCA.JK") == "AAPL"


def test_extract_tickers_handles_compare_queries_and_global_watchlists(tmp_path: Path):
    store = AgentStore(tmp_path / "state.sqlite3")

    tickers = _extract_tickers("Compare BBCA and BBRI with Nasdaq names like AAPL and MSFT", store, "BBCA.JK")
    assert set(["BBCA.JK", "BBRI.JK", "AAPL", "MSFT"]).issubset(set(tickers))
    assert _extract_tickers("List me a good buy stock in NASDAQ", store, "BBCA.JK")
    assert _extract_tickers("Any good buy in US stock?", store, "BBCA.JK")


def test_general_buy_question_scans_default_indonesian_candidates(tmp_path: Path):
    store = AgentStore(tmp_path / "state.sqlite3")

    tickers = _extract_tickers("What is the best stock to buy tomorrow?", store, "BBCA.JK")

    assert tickers == _indonesia_watchlist()
    assert len(tickers) >= 40


def test_generic_market_question_scans_without_active_ticker(tmp_path: Path):
    store = AgentStore(tmp_path / "state.sqlite3")

    tickers = _extract_tickers("Give me the market outlook", store, "BBCA.JK")

    assert tickers == _indonesia_watchlist()


def test_global_market_question_scans_all_configured_markets(tmp_path: Path):
    store = AgentStore(tmp_path / "state.sqlite3")

    tickers = _extract_tickers("What is the best stock across all markets?", store, "BBCA.JK")

    assert tickers == _global_watchlist()
    assert "BBCA.JK" in tickers
    assert "AAPL" in tickers


def test_country_and_index_questions_select_their_market_universe(tmp_path: Path):
    store = AgentStore(tmp_path / "state.sqlite3")

    nasdaq = _extract_tickers("What is best to buy in NASDAQ?", store, "BBCA.JK")
    nikkei = _extract_tickers("What is the Nikkei outlook?", store, "BBCA.JK")
    singapore = _extract_tickers("Best Singapore stock?", store, "BBCA.JK")

    assert nasdaq and all("." not in ticker for ticker in nasdaq)
    assert nikkei and all(ticker.endswith(".T") for ticker in nikkei)
    assert singapore and all(ticker.endswith(".SI") for ticker in singapore)


def test_available_tickers_includes_cached_symbols_and_defaults(tmp_path: Path):
    store = AgentStore(tmp_path / "state.sqlite3")
    store.upsert_market_bars([
        {
            "ticker": "TEST.JK",
            "timestamp": "2026-01-02T00:00:00+00:00",
            "open": 100,
            "high": 101,
            "low": 99,
            "close": 100,
            "volume": 1000,
            "source": "test",
        }
    ])

    tickers = _available_tickers(store)

    assert "TEST.JK" in tickers
    assert "BBCA.JK" in tickers


def test_pani_is_available_for_direct_ticker_questions(tmp_path: Path):
    store = AgentStore(tmp_path / "state.sqlite3")

    assert "PANI.JK" in _available_tickers(store)
    assert _ticker_from_question("What is happening with PANI?", store, "BBCA.JK") == "PANI.JK"


def test_comparison_reason_flattens_multiline_model_output():
    reason = _comparison_reason("### Market Context\nThe signal is HOLD.\nMore detail.", "HOLD")

    assert "\n" not in reason
    assert reason.startswith("### Market Context The signal is HOLD.")


def test_evidence_reason_is_complete_and_not_model_truncated():
    reason = _evidence_reason({
        "signal": "HOLD",
        "evidence": {"rsi14": 43.7, "macd_diff": -39.47, "confidence": 0.67},
    })

    assert reason == "HOLD signal; RSI 43.7 (neutral RSI); MACD -39.47 (negative momentum); confidence 67%."
    assert "..." not in reason
