import json
from pathlib import Path

from scripts.generate_analyst_training_data import _answer, generate


def test_generated_answer_is_grounded_and_structured():
    answer = _answer(
        "Should I buy BBCA.JK?",
        "BBCA.JK",
        {"close": 6300, "rsi14": 43.7, "macd_diff": -39.47},
        {"headline_count": 1, "sentiment": 0.0},
        {"signal": "HOLD", "confidence": 0.67, "source": "champion_model"},
    )

    assert "View:" in answer
    assert "Evidence:" in answer
    assert "Risks:" in answer
    assert "Bottom line:" in answer
    assert "6300.00" in answer
    assert "BUY" not in answer.split("Bottom line:", 1)[0]


def test_generator_writes_jsonl_examples(tmp_path: Path):
    raw = Path(__file__).resolve().parent.parent / "data" / "raw"
    output = tmp_path / "train.jsonl"
    count = generate(
        output,
        raw,
        tmp_path / "state.sqlite3",
        tmp_path / "missing-model.pkl",
        ["BBCA.JK"],
    )

    assert count == 5
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["evidence"]["workflow"]
    assert rows[0]["evidence"]["memory"]["previous_turns"] == []