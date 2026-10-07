"""Tests for paper-only order and position accounting."""
from pathlib import Path

import pytest

from app.agent.storage import AgentStore
from app.paper_trading import PaperBroker


def test_paper_broker_tracks_position(tmp_path: Path):
    broker = PaperBroker(AgentStore(tmp_path / "state.sqlite3"))

    broker.submit_market_order("BBCA.JK", "BUY", 10, 100)
    broker.submit_market_order("BBCA.JK", "BUY", 10, 120)
    broker.submit_market_order("BBCA.JK", "SELL", 5, 110)

    position = broker.store.paper_positions()[0]
    assert position["quantity"] == 15
    assert position["average_price"] == 110
    assert len(broker.store.paper_orders()) == 3


def test_paper_broker_rejects_oversell(tmp_path: Path):
    broker = PaperBroker(AgentStore(tmp_path / "state.sqlite3"))

    with pytest.raises(ValueError, match="short position"):
        broker.submit_market_order("BBCA.JK", "SELL", 1, 100)