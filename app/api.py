"""Authenticated HTTP boundary for the private analyst deployment."""
from __future__ import annotations

import secrets
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from app.agent.llm.local_model import TransformersLocalModel
from app.agent.orchestrator import ArakandarAgent
from app.agent.storage import AgentStore
from app.config import Settings
from app.paper_trading import PaperBroker


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    ticker: str = Field(min_length=1, max_length=20)
    headlines: list[str] = Field(default_factory=list, max_length=20)
    history: list[dict[str, str]] = Field(default_factory=list, max_length=20)


class PaperOrderRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=20)
    side: str = Field(pattern="^(BUY|SELL)$")
    quantity: float = Field(gt=0, le=1_000_000)
    price: float = Field(gt=0, le=1_000_000_000)


def create_app(settings: Settings | None = None, agent: ArakandarAgent | None = None) -> FastAPI:
    configured = settings or Settings.from_environment()
    configured.validate()

    if agent is None:
        language_model = None
        if configured.local_llm_path is not None:
            language_model = TransformersLocalModel(str(configured.local_llm_path), device="auto")
        agent = ArakandarAgent(
            configured.data_dir,
            configured.signal_model_path,
            AgentStore(configured.state_path),
            language_model=language_model,
        )
    broker = PaperBroker(agent.store)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield

    app = FastAPI(
        title="Arakandar Analyst API",
        version="1.0.0",
        docs_url=None if configured.environment.lower() in {"production", "prod"} else "/docs",
        lifespan=lifespan,
    )
    app.state.settings = configured
    app.state.agent = agent
    app.state.paper_broker = broker

    def authenticate(x_api_key: str | None = Header(default=None)) -> None:
        if not configured.api_key:
            if configured.environment.lower() in {"production", "prod"}:
                raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="API authentication is not configured")
            return
        if not x_api_key or not secrets.compare_digest(x_api_key, configured.api_key):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    def ticker_allowed(ticker: str) -> str:
        normalized = ticker.strip().upper()
        if normalized not in configured.allowed_tickers:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Ticker is not allowed")
        return normalized

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "environment": configured.environment}

    @app.get("/readyz")
    def readyz(_: None = Depends(authenticate)) -> dict[str, Any]:
        model_ready = configured.signal_model_path.exists()
        llm_ready = configured.local_llm_path is None or configured.local_llm_path.is_dir()
        ready = model_ready and llm_ready
        if not ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"status": "not_ready", "signal_model": model_ready, "local_llm": llm_ready},
            )
        return {"status": "ready", "live_trading_enabled": False}

    @app.post("/v1/chat")
    def chat(request: ChatRequest, _: None = Depends(authenticate)) -> dict[str, Any]:
        ticker = ticker_allowed(request.ticker)
        try:
            return agent.chat(
                request.question,
                ticker,
                headlines=request.headlines,
                history=request.history,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Analyst workflow unavailable") from exc

    @app.post("/v1/paper-orders", status_code=status.HTTP_201_CREATED)
    def paper_order(request: PaperOrderRequest, _: None = Depends(authenticate)) -> dict[str, Any]:
        ticker = ticker_allowed(request.ticker)
        try:
            return broker.submit_market_order(ticker, request.side, request.quantity, request.price)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    return app


app = create_app()
