"""Persistent state for agent task runs and tool audit events."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class AgentStore:
    """Small SQLite repository used by the local agent and UI."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._db_path = str(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS task_runs (
                    task_id TEXT PRIMARY KEY,
                    task TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                );
                CREATE TABLE IF NOT EXISTS tool_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    output_json TEXT,
                    error TEXT,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    FOREIGN KEY(task_id) REFERENCES task_runs(task_id)
                );
                CREATE TABLE IF NOT EXISTS market_bars (
                    ticker TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    source TEXT NOT NULL,
                    inserted_at TEXT NOT NULL,
                    PRIMARY KEY(ticker, timestamp)
                );
                CREATE TABLE IF NOT EXISTS news_articles (
                    article_id TEXT PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    published_at TEXT NOT NULL,
                    title TEXT NOT NULL,
                    url TEXT NOT NULL,
                    source TEXT NOT NULL,
                    sentiment REAL NOT NULL,
                    inserted_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS model_versions (
                    version_id TEXT PRIMARY KEY,
                    artifact_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    approved_at TEXT,
                    rejected_at TEXT
                );
                CREATE TABLE IF NOT EXISTS paper_orders (
                    order_id TEXT PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS paper_positions (
                    ticker TEXT PRIMARY KEY,
                    quantity REAL NOT NULL,
                    average_price REAL NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def start_task(self, task_id: str, task: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO task_runs(task_id, task, status, created_at) VALUES (?, ?, ?, ?)",
                (task_id, task, "running", now),
            )

    def finish_task(self, task_id: str, result: Dict[str, Any]) -> None:
        with self._connection() as connection:
            connection.execute(
                "UPDATE task_runs SET status = ?, result_json = ?, completed_at = ? WHERE task_id = ?",
                ("completed", json.dumps(result, default=str), datetime.now(timezone.utc).isoformat(), task_id),
            )

    def fail_task(self, task_id: str, error: str) -> None:
        with self._connection() as connection:
            connection.execute(
                "UPDATE task_runs SET status = ?, error = ?, completed_at = ? WHERE task_id = ?",
                ("failed", error, datetime.now(timezone.utc).isoformat(), task_id),
            )

    def record_tool_event(
        self,
        task_id: str,
        tool_name: str,
        status: str,
        inputs: Dict[str, Any],
        output: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        started_at: Optional[str] = None,
    ) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO tool_events(
                    task_id, tool_name, status, input_json, output_json, error,
                    started_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    tool_name,
                    status,
                    json.dumps(inputs, default=str),
                    json.dumps(output, default=str) if output is not None else None,
                    error,
                    started_at or datetime.now(timezone.utc).isoformat(),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def recent_tasks(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM task_runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def task_events(self, task_id: str) -> List[Dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM tool_events WHERE task_id = ? ORDER BY id", (task_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def upsert_market_bars(self, bars: List[Dict[str, Any]]) -> int:
        if not bars:
            return 0
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            before = connection.total_changes
            connection.executemany(
                """
                INSERT INTO market_bars(
                    ticker, timestamp, open, high, low, close, volume, source, inserted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, timestamp) DO UPDATE SET
                    open=excluded.open, high=excluded.high, low=excluded.low,
                    close=excluded.close, volume=excluded.volume,
                    source=excluded.source, inserted_at=excluded.inserted_at
                """,
                [
                    (
                        bar["ticker"], bar["timestamp"], bar["open"], bar["high"],
                        bar["low"], bar["close"], bar["volume"], bar["source"], now,
                    )
                    for bar in bars
                ],
            )
            return connection.total_changes - before

    def latest_market_bars(self, ticker: str, limit: int = 200) -> List[Dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM market_bars WHERE ticker = ? ORDER BY timestamp DESC LIMIT ?",
                (ticker, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def market_tickers(self) -> List[str]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT DISTINCT ticker FROM market_bars ORDER BY ticker"
            ).fetchall()
        return [str(row["ticker"]) for row in rows]

    def upsert_news_articles(self, articles: List[Dict[str, Any]]) -> int:
        if not articles:
            return 0
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            before = connection.total_changes
            connection.executemany(
                """
                INSERT INTO news_articles(
                    article_id, ticker, published_at, title, url, source, sentiment, inserted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(article_id) DO UPDATE SET
                    ticker=excluded.ticker, published_at=excluded.published_at,
                    title=excluded.title, url=excluded.url, source=excluded.source,
                    sentiment=excluded.sentiment, inserted_at=excluded.inserted_at
                """,
                [
                    (
                        article["article_id"], article["ticker"], article["published_at"],
                        article["title"], article["url"], article["source"], article["sentiment"], now,
                    )
                    for article in articles
                ],
            )
            return connection.total_changes - before

    def latest_news(self, ticker: str, limit: int = 20) -> List[Dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM news_articles WHERE ticker = ? ORDER BY published_at DESC LIMIT ?",
                (ticker, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def add_model_version(self, version: Dict[str, Any]) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO model_versions(
                    version_id, artifact_path, status, metrics_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    version["version_id"], version["artifact_path"], version.get("status", "candidate"),
                    json.dumps(version.get("metrics", {}), default=str),
                    version.get("created_at", datetime.now(timezone.utc).isoformat()),
                ),
            )

    def update_model_status(self, version_id: str, status: str) -> None:
        timestamp_column = {"approved": "approved_at", "rejected": "rejected_at"}.get(status)
        if timestamp_column:
            with self._connection() as connection:
                connection.execute(
                    f"UPDATE model_versions SET status = ?, {timestamp_column} = ? WHERE version_id = ?",
                    (status, datetime.now(timezone.utc).isoformat(), version_id),
                )
        else:
            with self._connection() as connection:
                connection.execute("UPDATE model_versions SET status = ? WHERE version_id = ?", (status, version_id))

    def model_versions(self) -> List[Dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM model_versions ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]

    def add_paper_order(self, order: Dict[str, Any]) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO paper_orders(order_id, ticker, side, quantity, price, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    order["order_id"], order["ticker"], order["side"], order["quantity"],
                    order["price"], order.get("status", "filled"),
                    order.get("created_at", datetime.now(timezone.utc).isoformat()),
                ),
            )

    def upsert_paper_position(self, position: Dict[str, Any]) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO paper_positions(ticker, quantity, average_price, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    quantity=excluded.quantity, average_price=excluded.average_price,
                    updated_at=excluded.updated_at
                """,
                (
                    position["ticker"], position["quantity"], position["average_price"],
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def paper_orders(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM paper_orders ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]

    def paper_positions(self) -> List[Dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM paper_positions ORDER BY ticker").fetchall()
        return [dict(row) for row in rows]
