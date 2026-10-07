"""Tests for local scheduled workflow execution."""
from app.scheduler import LocalScheduler


def test_run_once_isolates_failed_jobs():
    calls = []

    def successful_job():
        calls.append("success")

    def failing_job():
        calls.append("failed")
        raise RuntimeError("expected")

    scheduler = LocalScheduler()
    scheduler.add_job("good", 60, successful_job)
    scheduler.add_job("bad", 60, failing_job)

    assert scheduler.run_once() == ["good"]
    assert calls == ["success", "failed"]
