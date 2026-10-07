"""Environment-driven settings for the deployable service."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _boolean(value: str, default: bool = False) -> bool:
    if not value:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    state_path: Path
    signal_model_path: Path
    local_llm_path: Path | None
    api_key: str
    allowed_tickers: frozenset[str]
    live_trading_enabled: bool = False
    require_local_llm: bool = False
    environment: str = "development"

    @classmethod
    def from_environment(cls, project_root: Path | None = None) -> "Settings":
        root = Path(project_root or Path(__file__).resolve().parents[1])
        configured_tickers = os.getenv("ARAKANDAR_ALLOWED_TICKERS", "BBCA.JK")
        llm_value = os.getenv("ARAKANDAR_LLM_PATH", "").strip()
        return cls(
            project_root=root,
            data_dir=Path(os.getenv("ARAKANDAR_DATA_DIR", root / "data" / "raw")),
            state_path=Path(os.getenv("ARAKANDAR_STATE_PATH", root / "data" / "agent_state.sqlite3")),
            signal_model_path=Path(
                os.getenv("ARAKANDAR_SIGNAL_MODEL", root / "models" / "champion_lgbm.pkl")
            ),
            local_llm_path=Path(llm_value) if llm_value else None,
            api_key=os.getenv("ARAKANDAR_API_KEY", ""),
            allowed_tickers=frozenset(
                ticker.strip().upper()
                for ticker in configured_tickers.split(",")
                if ticker.strip()
            ),
            live_trading_enabled=_boolean(os.getenv("ARAKANDAR_LIVE_TRADING_ENABLED", "false")),
            require_local_llm=_boolean(os.getenv("ARAKANDAR_REQUIRE_LOCAL_LLM", "false")),
            environment=os.getenv("ARAKANDAR_ENVIRONMENT", "development"),
        )

    def validate(self) -> None:
        if self.environment.lower() in {"production", "prod"} and not self.api_key:
            raise ValueError("ARAKANDAR_API_KEY is required in production")
        if not self.allowed_tickers:
            raise ValueError("ARAKANDAR_ALLOWED_TICKERS must contain at least one ticker")
        if self.live_trading_enabled:
            raise ValueError(
                "Live trading is not implemented; keep ARAKANDAR_LIVE_TRADING_ENABLED=false"
            )
        if self.require_local_llm and (self.local_llm_path is None or not self.local_llm_path.is_dir()):
            raise ValueError("ARAKANDAR_LLM_PATH must point to a local model directory")
