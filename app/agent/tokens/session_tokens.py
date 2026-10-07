"""
session_tokens.py
Terminal Usage Token System (Arakandar Dual Token System - Phase 2).

Bloomberg-style session token ledger. Not tied to any external LLM API -
purely a UX/gamification layer that tracks "compute budget" per terminal
session and is returned alongside every API response.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

TASK_COSTS: Dict[str, int] = {
    "quote": 10,
    "news_fetch": 20,
    "chat_reply": 40,
    "full_ml_decision": 150,
    "model_retrain": 500,
}

DEFAULT_SESSION_BUDGET = 5000


@dataclass
class SessionTokenLedger:
    balance: int = DEFAULT_SESSION_BUDGET
    total_budget: int = DEFAULT_SESSION_BUDGET
    burst_mode: bool = False
    history: list = field(default_factory=list)

    def deduct(self, task: str) -> int:
        cost = TASK_COSTS.get(task, 10)
        self.balance = max(0, self.balance - cost)
        self.history.append({"task": task, "cost": cost, "balance_after": self.balance})
        self.burst_mode = self.balance < (0.2 * self.total_budget)
        return cost

    def status(self) -> Dict:
        return {
            "balance": self.balance,
            "total_budget": self.total_budget,
            "burst_mode": self.burst_mode,
            "display": f"TOKENS: {self.balance:,} / {self.total_budget:,}"
                       + (" [BURST MODE: ACTIVE]" if self.burst_mode else ""),
        }

    def reset(self):
        self.balance = self.total_budget
        self.burst_mode = False
        self.history.clear()
