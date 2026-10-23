from __future__ import annotations

import threading
import time

import pytest

from voxcpm.inference_queue import ByteLRUCache, InferenceQueueFullError, InferenceScheduler


def test_scheduler_runs_jobs_in_fifo_order_with_one_worker() -> None:
    scheduler = InferenceScheduler(workers=1, queue_size=3)
    release = threading.Event()
    order: list[int] = []

    def run(value: int):
        def generate():
            if value == 1:
                release.wait(timeout=1)
            order.append(value)
            yield value

        return generate

    first = scheduler.stream(run(1))
    second = scheduler.stream(run(2))
    third = scheduler.stream(run(3))
    release.set()
    try:
        assert list(first) == [1]
        assert list(second) == [2]
        assert list(third) == [3]
        assert order == [1, 2, 3]
    finally:
        scheduler.close()


def test_scheduler_rejects_requests_beyond_bounded_waiting_capacity() -> None:
    scheduler = InferenceScheduler(workers=1, queue_size=1)
    started = threading.Event()
    release = threading.Event()

    def blocked():
        started.set()
        release.wait(timeout=1)
        yield 1

    active = scheduler.stream(blocked)
    consumer = threading.Thread(target=lambda: list(active), daemon=True)
    consumer.start()
    assert started.wait(timeout=1)
    waiting = scheduler.stream(lambda: iter([2]))
    with pytest.raises(InferenceQueueFullError):
        scheduler.stream(lambda: iter([3]))
    release.set()
    consumer.join(timeout=1)
    try:
        assert list(waiting) == [2]
    finally:
        scheduler.close()


def test_scheduler_reports_queue_position_and_running_state() -> None:
    scheduler = InferenceScheduler(workers=1, queue_size=2)
    release = threading.Event()
    events: list[tuple[str, dict]] = []

    def blocked():
        release.wait(timeout=1)
        yield 1

    stream = scheduler.stream(blocked, callback=lambda event, details: events.append((event, details)))
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline and not any(event == "running" for event, _ in events):
        time.sleep(0.01)
    release.set()
    try:
        assert list(stream) == [1]
        assert any(event == "running" for event, _ in events)
    finally:
        scheduler.close()


def test_scheduler_executes_inference_on_the_consuming_request_thread() -> None:
    scheduler = InferenceScheduler(workers=1, queue_size=1)
    caller_thread = threading.get_ident()
    execution_threads: list[int] = []

    def generate():
        execution_threads.append(threading.get_ident())
        yield "audio"

    try:
        assert list(scheduler.stream(generate)) == ["audio"]
        assert execution_threads == [caller_thread]
    finally:
        scheduler.close()


def test_byte_cache_is_bounded_and_reuses_exact_bytes() -> None:
    cache = ByteLRUCache(max_entries=1, max_bytes=8)
    calls = 0

    def create() -> bytes:
        nonlocal calls
        calls += 1
        return b"voice"

    assert cache.get_or_create("a", create) == b"voice"
    assert cache.get_or_create("a", create) == b"voice"
    assert calls == 1
    cache.get_or_create("b", lambda: b"other")
    assert cache.snapshot()["entries"] == 1
