"""Small restartable scheduler for local polling workflows."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, List

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScheduledJob:
    name: str
    interval_seconds: float
    action: Callable[[], object]
    next_run: float = 0.0


class LocalScheduler:
    """Run registered jobs in one process with per-job failure isolation."""

    def __init__(self):
        self._jobs: List[ScheduledJob] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def add_job(self, name: str, interval_seconds: float, action: Callable[[], object]) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be greater than zero")
        with self._lock:
            self._jobs.append(ScheduledJob(name, interval_seconds, action))

    def run_once(self) -> List[str]:
        completed = []
        with self._lock:
            jobs = list(self._jobs)
        for job in jobs:
            try:
                job.action()
                completed.append(job.name)
            except Exception:
                logger.exception("Scheduled job failed: %s", job.name)
        return completed

    def run_forever(self, poll_seconds: float = 1.0) -> None:
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be greater than zero")
        next_runs = {job.name: 0.0 for job in self._jobs}
        self._stop.clear()
        while not self._stop.is_set():
            now = time.monotonic()
            for job in list(self._jobs):
                if now < next_runs[job.name]:
                    continue
                try:
                    job.action()
                except Exception:
                    logger.exception("Scheduled job failed: %s", job.name)
                next_runs[job.name] = now + job.interval_seconds
            self._stop.wait(poll_seconds)

    def stop(self) -> None:
        self._stop.set()