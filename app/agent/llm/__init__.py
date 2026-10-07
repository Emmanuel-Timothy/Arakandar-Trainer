"""Optional local language-model integrations."""

from app.agent.llm.local_model import LocalLanguageModel, TransformersLocalModel

__all__ = ["LocalLanguageModel", "TransformersLocalModel"]