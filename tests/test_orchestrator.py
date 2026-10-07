"""Tests for the custom multi-step analyst agent."""
from pathlib import Path

from app.agent.orchestrator import ArakandarAgent
from app.agent.storage import AgentStore


def test_agent_executes_tools_and_persists_audit(tmp_path: Path):
    store_path = tmp_path / "agent.sqlite3"
    store = AgentStore(store_path)
    agent = ArakandarAgent(
        data_dir=Path(__file__).resolve().parent.parent / "data" / "raw",
        model_path=tmp_path / "missing-model.pkl",
        store=store,
    )

    result = agent.run(
        "Explain the latest signal",
        "BBCA.JK",
        ["BBCA reports record profit and dividend hike"],
    )

    assert result["report"]["headline"].startswith("BBCA.JK signal:")
    assert result["report"]["disclaimer"]
    assert result["route"]["market_snapshot"] == "local_csv"
    assert len(store.task_events(result["task_id"])) == 4
    assert [event["status"] for event in store.task_events(result["task_id"])] == [
        "completed",
        "completed",
        "completed",
        "completed",
    ]


def test_agent_state_is_readable_after_store_reopen(tmp_path: Path):
    store_path = tmp_path / "agent.sqlite3"
    agent = ArakandarAgent(
        data_dir=Path(__file__).resolve().parent.parent / "data" / "raw",
        model_path=tmp_path / "missing-model.pkl",
        store=AgentStore(store_path),
    )
    result = agent.run("Explain the latest signal", "BBRI.JK")

    reopened = AgentStore(store_path)
    tasks = reopened.recent_tasks()

    assert tasks[0]["task_id"] == result["task_id"]
    assert tasks[0]["status"] == "completed"
