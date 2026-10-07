"""Candidate/champion model lifecycle with explicit approval gates."""
from __future__ import annotations

import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from app.agent.storage import AgentStore


class ModelRegistry:
    def __init__(self, store: AgentStore, champion_path: Path):
        self.store = store
        self.champion_path = Path(champion_path)

    def register_candidate(self, artifact_path: Path, metrics: Dict[str, Any]) -> str:
        artifact_path = Path(artifact_path)
        if not artifact_path.exists():
            raise FileNotFoundError(artifact_path)
        version_id = f"model-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
        self.store.add_model_version({
            "version_id": version_id,
            "artifact_path": str(artifact_path.resolve()),
            "status": "candidate",
            "metrics": metrics,
        })
        return version_id

    def approve(self, version_id: str) -> None:
        versions = {version["version_id"]: version for version in self.store.model_versions()}
        version = versions.get(version_id)
        if version is None:
            raise KeyError(version_id)
        if version["status"] != "candidate":
            raise ValueError(f"Only candidates can be approved: {version['status']}")
        artifact = Path(version["artifact_path"])
        if not artifact.exists():
            raise FileNotFoundError(artifact)
        self.champion_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifact, self.champion_path)
        self.store.update_model_status(version_id, "approved")

    def reject(self, version_id: str) -> None:
        versions = {version["version_id"]: version for version in self.store.model_versions()}
        version = versions.get(version_id)
        if version is None:
            raise KeyError(version_id)
        if version["status"] != "candidate":
            raise ValueError(f"Only candidates can be rejected: {version['status']}")
        self.store.update_model_status(version_id, "rejected")

    def rollback_to(self, version_id: str) -> None:
        versions = {version["version_id"]: version for version in self.store.model_versions()}
        version = versions.get(version_id)
        if version is None:
            raise KeyError(version_id)
        if version["status"] != "approved":
            raise ValueError("Only approved models can be restored")
        artifact = Path(version["artifact_path"])
        if not artifact.exists():
            raise FileNotFoundError(artifact)
        shutil.copy2(artifact, self.champion_path)