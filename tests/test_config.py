from pathlib import Path

import pytest

from app.config import Settings


def test_production_requires_api_key(tmp_path: Path):
    settings = Settings(
        project_root=tmp_path,
        data_dir=tmp_path,
        state_path=tmp_path / "state.sqlite3",
        signal_model_path=tmp_path / "model.pkl",
        local_llm_path=None,
        api_key="",
        allowed_tickers=frozenset({"BBCA.JK"}),
        environment="production",
    )

    with pytest.raises(ValueError, match="API_KEY"):
        settings.validate()


def test_live_trading_is_rejected_until_implemented(tmp_path: Path):
    settings = Settings(
        project_root=tmp_path,
        data_dir=tmp_path,
        state_path=tmp_path / "state.sqlite3",
        signal_model_path=tmp_path / "model.pkl",
        local_llm_path=None,
        api_key="test-key",
        allowed_tickers=frozenset({"BBCA.JK"}),
        live_trading_enabled=True,
    )

    with pytest.raises(ValueError, match="Live trading is not implemented"):
        settings.validate()
