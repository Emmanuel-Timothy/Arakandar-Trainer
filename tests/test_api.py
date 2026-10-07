from pathlib import Path

from fastapi.testclient import TestClient

from app.agent.storage import AgentStore
from app.api import create_app
from app.config import Settings


class FakeAgent:
    def __init__(self, store: AgentStore):
        self.store = store

    def chat(self, question: str, ticker: str, headlines=None, history=None):
        return {
            "task_id": "test-task",
            "ticker": ticker,
            "question": question,
            "answer": "Evidence-backed test response",
            "signal": "HOLD",
            "sources": [],
            "disclaimer": "Research signal only. No live order was placed.",
        }


def _client(tmp_path: Path, api_key: str = "test-key") -> TestClient:
    model_path = tmp_path / "champion_lgbm.pkl"
    model_path.touch()
    settings = Settings(
        project_root=tmp_path,
        data_dir=tmp_path,
        state_path=tmp_path / "state.sqlite3",
        signal_model_path=model_path,
        local_llm_path=None,
        api_key=api_key,
        allowed_tickers=frozenset({"BBCA.JK"}),
    )
    store = AgentStore(settings.state_path)
    return TestClient(create_app(settings, FakeAgent(store)))


def test_health_is_public_but_readiness_requires_authentication(tmp_path: Path):
    client = _client(tmp_path)

    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code == 401
    assert client.get("/readyz", headers={"X-API-Key": "test-key"}).status_code == 200


def test_chat_requires_authentication_and_ticker_allowlist(tmp_path: Path):
    client = _client(tmp_path)
    payload = {"question": "Explain the signal", "ticker": "BBCA.JK"}

    assert client.post("/v1/chat", json=payload).status_code == 401
    response = client.post("/v1/chat", json=payload, headers={"X-API-Key": "test-key"})
    assert response.status_code == 200
    assert response.json()["signal"] == "HOLD"

    rejected = client.post(
        "/v1/chat",
        json={**payload, "ticker": "UNALLOWED"},
        headers={"X-API-Key": "test-key"},
    )
    assert rejected.status_code == 400


def test_paper_order_is_authenticated_and_has_no_live_route(tmp_path: Path):
    client = _client(tmp_path)
    payload = {"ticker": "BBCA.JK", "side": "BUY", "quantity": 1, "price": 100}

    response = client.post("/v1/paper-orders", json=payload, headers={"X-API-Key": "test-key"})
    assert response.status_code == 201
    assert response.json()["status"] == "filled"
    assert client.post("/v1/live-orders", json=payload, headers={"X-API-Key": "test-key"}).status_code == 404
