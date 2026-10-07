from pathlib import Path
from types import SimpleNamespace

import pytest

from app.agent.llm.context import build_market_messages
from app.agent.llm.local_model import (
    IncompleteGenerationError,
    PromptTooLongError,
    TransformersLocalModel,
)
from app.agent.orchestrator import ArakandarAgent, _answer_is_ticker_grounded
from app.agent.storage import AgentStore
from run_chat import _default_model_path, _ticker_from_question


class FakeLanguageModel:
    model_name = "fake-local-model"

    def __init__(self, response: str):
        self.response = response
        self.messages = []
        self.generation = {}

    def generate(self, messages, **generation):
        self.messages = messages
        self.generation = generation
        return self.response


def test_chat_ticker_can_be_inferred_from_direct_question(tmp_path: Path):
    (tmp_path / "BBRI_JK_ohlcv.csv").touch()

    assert _ticker_from_question("Analisa BBCA", tmp_path, "BBRI.JK") == "BBCA.JK"
    assert _ticker_from_question("What are the risks for BBRI?", tmp_path, "BBCA.JK") == "BBRI.JK"
    assert _ticker_from_question("What are the risks?", tmp_path, "BBCA.JK") == "BBCA.JK"


class FakeTokenizer:
    eos_token_id = 0

    def __init__(self, torch_module):
        self.torch = torch_module

    def apply_chat_template(self, messages, **kwargs):
        self.last_prompt = "|".join(message["content"] for message in messages)
        return self.last_prompt

    def __call__(self, prompt, **kwargs):
        return {
            "input_ids": self.torch.tensor([[1]]),
            "attention_mask": self.torch.tensor([[1]]),
        }

    def decode(self, tokens, **kwargs):
        return " ".join(str(token) for token in tokens.tolist() if token != self.eos_token_id)


class FakeGenerationModel:
    def __init__(self, torch_module, end_generation: bool, stop_early: bool = False):
        self.torch = torch_module
        self.end_generation = end_generation
        self.stop_early = stop_early
        self.calls = 0

    def parameters(self):
        yield SimpleNamespace(device=self.torch.device("cpu"))

    def generate(self, input_ids, attention_mask, max_new_tokens, **kwargs):
        self.calls += 1
        if self.end_generation and self.calls == 2:
            new_tokens = self.torch.tensor([[0]])
        elif self.stop_early:
            new_tokens = self.torch.tensor([[self.calls + 1]])
        else:
            new_tokens = self.torch.full((1, max_new_tokens), self.calls + 1)
        return self.torch.cat((input_ids, new_tokens), dim=1)


class LengthAwareFakeTokenizer(FakeTokenizer):
    def __call__(self, prompt, **kwargs):
        token_count = len(prompt)
        return {
            "input_ids": self.torch.ones((1, token_count), dtype=self.torch.long),
            "attention_mask": self.torch.ones((1, token_count), dtype=self.torch.long),
        }


def test_local_model_continues_until_eos_token():
    torch = pytest.importorskip("torch")
    model = TransformersLocalModel("unused")
    model._tokenizer = FakeTokenizer(torch)
    model._model = FakeGenerationModel(torch, end_generation=True)

    answer = model.generate([], max_new_tokens=2, max_total_new_tokens=4)

    assert answer == "2 2"
    assert model._model.calls == 2


def test_local_model_rejects_output_that_stays_truncated():
    torch = pytest.importorskip("torch")
    model = TransformersLocalModel("unused")
    model._tokenizer = FakeTokenizer(torch)
    model._model = FakeGenerationModel(torch, end_generation=False)

    with pytest.raises(IncompleteGenerationError):
        model.generate([], max_new_tokens=2, max_total_new_tokens=4)


def test_local_model_rejects_early_stop_without_eos():
    torch = pytest.importorskip("torch")
    model = TransformersLocalModel("unused")
    model._tokenizer = FakeTokenizer(torch)
    model._model = FakeGenerationModel(torch, end_generation=False, stop_early=True)

    with pytest.raises(IncompleteGenerationError):
        model.generate([], max_new_tokens=2, max_total_new_tokens=4)


def test_local_model_trims_history_before_current_prompt():
    torch = pytest.importorskip("torch")
    model = TransformersLocalModel("unused", max_input_tokens=10)
    model._tokenizer = LengthAwareFakeTokenizer(torch)
    model._model = FakeGenerationModel(torch, end_generation=True)

    model.generate(
        [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "old-context"},
            {"role": "user", "content": "q"},
        ],
        max_new_tokens=2,
        max_total_new_tokens=4,
    )

    assert model._tokenizer.last_prompt == "s|q"


def test_local_model_rejects_current_prompt_over_input_budget():
    torch = pytest.importorskip("torch")
    model = TransformersLocalModel("unused", max_input_tokens=5)
    model._tokenizer = LengthAwareFakeTokenizer(torch)
    model._model = FakeGenerationModel(torch, end_generation=True)

    with pytest.raises(PromptTooLongError):
        model.generate(
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "latest-question"},
            ],
            max_new_tokens=2,
        )


def test_default_model_prefers_3b_when_available():
    assert _default_model_path().name in {"arakandar-3b-lora", "arakandar-3b", "arakandar-1.5b"}


def test_context_contains_authoritative_signal_and_bounded_news():
    messages = build_market_messages(
        "Analisa berita terbaru",
        "BBCA.JK",
        {"as_of": "2026-01-01", "close": 9000, "rsi14": 42, "macd_diff": 1, "active_tokens": []},
        {
            "sentiment": 0.5,
            "headlines": ["one", "two", "three"],
            "articles": [
                {"title": "one", "source": "Wire", "published_at": "2026-01-01", "url": "https://one"},
                {"title": "two", "source": "Desk", "published_at": "2026-01-02", "url": "https://two"},
                {"title": "three", "source": "Desk", "published_at": "2026-01-03", "url": "https://three"},
            ],
        },
        {"signal": "BUY", "confidence": 0.8, "probabilities": {"BUY": 0.8}, "top_drivers": ["rsi14"]},
        max_news=2,
    )

    assert messages[0]["role"] == "system"
    assert "deterministic signal as authoritative" in messages[0]["content"]
    assert '"signal": "BUY"' in messages[1]["content"]
    assert '"url": "https://one"' in messages[1]["content"]
    assert '"url": "https://three"' not in messages[1]["content"]
    assert "Do not include URLs, citations, source lists" in messages[0]["content"]
    assert "Bottom line" in messages[0]["content"]


def test_agent_uses_injected_local_model_for_summary(tmp_path: Path):
    model = FakeLanguageModel("BBCA.JK: The local answer is grounded in the supplied evidence.")
    agent = ArakandarAgent(
        data_dir=Path(__file__).resolve().parent.parent / "data" / "raw",
        model_path=tmp_path / "missing-model.pkl",
        store=AgentStore(tmp_path / "agent.sqlite3"),
        language_model=model,
    )

    result = agent.run("Explain the latest signal", "BBCA.JK", ["Profit increased"])

    assert result["report"]["summary"] == model.response
    assert result["report"]["generation"] == {"model": "fake-local-model"}
    assert model.generation["max_new_tokens"] == 768
    assert model.messages[1]["role"] == "user"
    assert "strictly about that ticker" in model.messages[0]["content"]


def test_agent_refuses_out_of_scope_questions_without_calling_model(tmp_path: Path):
    model = FakeLanguageModel("Here is a Python script.")
    agent = ArakandarAgent(
        data_dir=Path(__file__).resolve().parent.parent / "data" / "raw",
        model_path=tmp_path / "missing-model.pkl",
        store=AgentStore(tmp_path / "agent.sqlite3"),
        language_model=model,
    )

    result = agent.run("Write a Python script to sort a list", "BBCA.JK")

    assert "only help with stock-market analysis" in result["report"]["summary"]
    assert result["report"]["generation"] == {"model": "scope_guard"}
    assert model.messages == []


def test_agent_accepts_indonesian_market_questions(tmp_path: Path):
    model = FakeLanguageModel("BBCA.JK: berita saham terbaru menunjukkan katalis positif.")
    agent = ArakandarAgent(
        data_dir=Path(__file__).resolve().parent.parent / "data" / "raw",
        model_path=tmp_path / "missing-model.pkl",
        store=AgentStore(tmp_path / "agent.sqlite3"),
        language_model=model,
    )

    result = agent.run("Analisa berita terbaru", "BBCA.JK")

    assert result["report"]["summary"] == model.response
    assert model.messages


def test_agent_passes_complete_news_records_to_local_model(tmp_path: Path):
    model = FakeLanguageModel("Evidence includes a cited article.")
    agent = ArakandarAgent(
        data_dir=Path(__file__).resolve().parent.parent / "data" / "raw",
        model_path=tmp_path / "missing-model.pkl",
        store=AgentStore(tmp_path / "agent.sqlite3"),
        language_model=model,
    )

    agent.run(
        "What is the latest catalyst?",
        "BBCA.JK",
        ["Profit increased"],
        news_articles=[
            {
                "title": "Profit increased",
                "source": "Market Wire",
                "published_at": "2026-01-01T00:00:00+00:00",
                "url": "https://example.test/article",
                "sentiment": 0.8,
            }
        ],
    )

    assert '"source": "Market Wire"' in model.messages[1]["content"]
    assert '"url": "https://example.test/article"' in model.messages[1]["content"]


def test_chat_returns_answer_sources_and_history(tmp_path: Path):
    model = FakeLanguageModel("BBCA remains a HOLD based on the supplied evidence.")
    agent = ArakandarAgent(
        data_dir=Path(__file__).resolve().parent.parent / "data" / "raw",
        model_path=tmp_path / "missing-model.pkl",
        store=AgentStore(tmp_path / "agent.sqlite3"),
        language_model=model,
    )

    response = agent.chat(
        "Should I wait for a pullback?",
        "BBCA.JK",
        headlines=["Profit increased"],
        news_articles=[
            {
                "title": "Profit increased",
                "source": "Market Wire",
                "published_at": "2026-01-01T00:00:00+00:00",
                "url": "https://example.test/article",
            }
        ],
        history=[{"role": "user", "content": "What changed?"}],
    )

    assert response["answer"] == model.response
    assert response["sources"][0]["url"] == "https://example.test/article"
    assert response["generation"] == {"model": "fake-local-model"}
    assert model.messages[-1]["role"] == "user"
    assert '"question": "Should I wait for a pullback?"' in model.messages[-1]["content"]


def test_chat_reads_cached_news_when_no_articles_are_supplied(tmp_path: Path):
    model = FakeLanguageModel("The answer uses cached local news.")
    store = AgentStore(tmp_path / "agent.sqlite3")
    store.upsert_news_articles(
        [
            {
                "article_id": "cached",
                "ticker": "BBCA.JK",
                "published_at": "2026-01-01T00:00:00+00:00",
                "title": "Cached profit update",
                "url": "https://example.test/cached",
                "source": "Cache Wire",
                "sentiment": 0.4,
            }
        ]
    )
    agent = ArakandarAgent(
        data_dir=Path(__file__).resolve().parent.parent / "data" / "raw",
        model_path=tmp_path / "missing-model.pkl",
        store=store,
        language_model=model,
        rss_feeds={},
    )

    response = agent.chat("What is the latest catalyst?", "BBCA.JK")

    assert response["sources"][0]["title"] == "Cached profit update"
    assert '"title": "Cached profit update"' in model.messages[1]["content"]


def test_agent_falls_back_when_local_model_returns_empty_text(tmp_path: Path):
    agent = ArakandarAgent(
        data_dir=Path(__file__).resolve().parent.parent / "data" / "raw",
        model_path=tmp_path / "missing-model.pkl",
        store=AgentStore(tmp_path / "agent.sqlite3"),
        language_model=FakeLanguageModel("  "),
    )

    result = agent.run("Explain the latest signal", "BBRI.JK")

    assert result["report"]["headline"].startswith("BBRI.JK signal:")
    assert result["report"]["language_model_fallback_reason"] == "empty local-model response"


def test_agent_returns_complete_fallback_for_incomplete_generation(tmp_path: Path):
    class IncompleteLanguageModel(FakeLanguageModel):
        def generate(self, messages, **generation):
            raise IncompleteGenerationError("generation stopped without EOS")

    agent = ArakandarAgent(
        data_dir=Path(__file__).resolve().parent.parent / "data" / "raw",
        model_path=tmp_path / "missing-model.pkl",
        store=AgentStore(tmp_path / "agent.sqlite3"),
        language_model=IncompleteLanguageModel("partial text"),
    )

    result = agent.run("Explain the latest signal", "BBRI.JK")

    assert result["report"]["summary"].startswith("BBRI.JK: latest close is ")
    assert result["report"]["language_model_fallback_reason"] == "generation stopped without EOS"


def test_agent_rejects_ticker_answer_that_drifts_to_ihsg(tmp_path: Path):
    model = FakeLanguageModel("BBRI has an outlook, while IHSG is the main focus.")
    agent = ArakandarAgent(
        data_dir=Path(__file__).resolve().parent.parent / "data" / "raw",
        model_path=tmp_path / "missing-model.pkl",
        store=AgentStore(tmp_path / "agent.sqlite3"),
        language_model=model,
    )

    result = agent.run("What are the risks for BBRI?", "BBRI.JK")

    assert result["report"]["summary"].startswith("BBRI.JK: latest close is ")
    assert result["report"]["language_model_fallback_reason"] == (
        "local-model response was not grounded in the requested ticker"
    )


def test_agent_rejects_answer_that_drifts_to_another_available_ticker(tmp_path: Path):
    model = FakeLanguageModel("BBCA.JK looks steady, while BBRI is the stronger opportunity.")
    agent = ArakandarAgent(
        data_dir=Path(__file__).resolve().parent.parent / "data" / "raw",
        model_path=tmp_path / "missing-model.pkl",
        store=AgentStore(tmp_path / "agent.sqlite3"),
        language_model=model,
    )

    result = agent.run("Analisa BBCA", "BBCA.JK")

    assert result["report"]["summary"].startswith("BBCA.JK: latest close is ")
    assert result["report"]["language_model_fallback_reason"] == (
        "local-model response was not grounded in the requested ticker"
    )


def test_ticker_grounding_allows_explicit_comparison():
    assert _answer_is_ticker_grounded(
        "BBCA is steadier than BBRI.",
        "Compare BBCA and BBRI",
        "BBCA.JK",
        {"BBCA.JK", "BBRI.JK"},
    )