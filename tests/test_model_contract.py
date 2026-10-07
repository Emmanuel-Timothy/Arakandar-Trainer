import pytest

from app.agent.llm.schema import ModelOutput, model_input_from_evidence


def test_model_input_contract_converts_existing_evidence():
    contract = model_input_from_evidence(
        "What is the catalyst?",
        "BBCA.JK",
        {"as_of": "2026-09-13", "close": 9875, "rsi14": 54, "active_tokens": ["TK_TEST"]},
        {
            "signal": "HOLD",
            "confidence": 0.62,
            "probabilities": {"BUY": 0.2, "HOLD": 0.62, "SELL": 0.18},
            "top_drivers": ["rsi14"],
            "source": "champion_model",
        },
        [{"title": "Profit update", "url": "https://example.test/news"}],
    )

    assert contract.to_dict()["market"]["ticker"] == "BBCA.JK"
    assert contract.to_dict()["signal"]["signal"] == "HOLD"
    assert contract.to_dict()["news"][0]["url"] == "https://example.test/news"


def test_model_input_rejects_invalid_signal():
    with pytest.raises(ValueError, match="signal must"):
        model_input_from_evidence(
            "Question",
            "BBCA.JK",
            {"close": 1},
            {"signal": "MAYBE", "confidence": 0.5},
        )


def test_model_output_rejects_empty_answer():
    with pytest.raises(ValueError, match="assistant response"):
        ModelOutput(assistant="").validate()