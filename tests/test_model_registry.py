"""Tests for manual model approval and candidate safety."""
from pathlib import Path

import pytest

from app.agent.storage import AgentStore
from app.model_registry import ModelRegistry


def test_candidate_is_not_champion_until_approved(tmp_path: Path):
    candidate = tmp_path / "candidate.pkl"
    candidate.write_bytes(b"candidate")
    champion = tmp_path / "champion.pkl"
    champion.write_bytes(b"old")
    registry = ModelRegistry(AgentStore(tmp_path / "state.sqlite3"), champion)

    version_id = registry.register_candidate(candidate, {"val_f1": 0.7})

    assert champion.read_bytes() == b"old"
    assert registry.store.model_versions()[0]["status"] == "candidate"
    registry.approve(version_id)
    assert champion.read_bytes() == b"candidate"
    assert registry.store.model_versions()[0]["status"] == "approved"


def test_rejected_candidate_cannot_be_approved(tmp_path: Path):
    candidate = tmp_path / "candidate.pkl"
    candidate.write_bytes(b"candidate")
    registry = ModelRegistry(AgentStore(tmp_path / "state.sqlite3"), tmp_path / "champion.pkl")
    version_id = registry.register_candidate(candidate, {})

    registry.reject(version_id)

    with pytest.raises(ValueError, match="Only candidates"):
        registry.approve(version_id)


def test_approved_model_can_be_restored(tmp_path: Path):
    candidate = tmp_path / "candidate.pkl"
    candidate.write_bytes(b"approved")
    champion = tmp_path / "champion.pkl"
    champion.write_bytes(b"old")
    registry = ModelRegistry(AgentStore(tmp_path / "state.sqlite3"), champion)
    version_id = registry.register_candidate(candidate, {})
    registry.approve(version_id)
    champion.write_bytes(b"changed")

    registry.rollback_to(version_id)

    assert champion.read_bytes() == b"approved"