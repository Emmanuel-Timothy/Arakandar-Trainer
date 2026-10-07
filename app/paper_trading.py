"""Paper-only execution service with no broker side effects."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Dict

from app.agent.storage import AgentStore


class PaperBroker:
    def __init__(self, store: AgentStore, fee_rate: float = 0.0015):
        self.store = store
        self.fee_rate = fee_rate

    def submit_market_order(self, ticker: str, side: str, quantity: float, price: float) -> Dict:
        side = side.upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")
        if quantity <= 0 or price <= 0:
            raise ValueError("quantity and price must be positive")
        positions = {position["ticker"]: position for position in self.store.paper_positions()}
        current = positions.get(ticker, {"quantity": 0.0, "average_price": 0.0})
        signed_quantity = quantity if side == "BUY" else -quantity
        new_quantity = float(current["quantity"]) + signed_quantity
        if new_quantity < 0:
            raise ValueError("paper order would create a short position")
        current_quantity = float(current["quantity"])
        average_price = (
            ((current_quantity * float(current["average_price"])) + (quantity * price)) / new_quantity
            if side == "BUY" and new_quantity > 0 else float(current["average_price"])
        )
        if new_quantity == 0:
            average_price = 0.0
        order = {
            "order_id": f"paper-{uuid.uuid4().hex}", "ticker": ticker, "side": side,
            "quantity": quantity, "price": price, "status": "filled",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.store.add_paper_order(order)
        self.store.upsert_paper_position({"ticker": ticker, "quantity": new_quantity, "average_price": average_price})
        return order