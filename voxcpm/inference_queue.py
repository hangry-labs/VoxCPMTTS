from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass, field
from typing import Any


class InferenceQueueFullError(RuntimeError):
    """Raised when bounded inference admission has no remaining capacity."""


QueueStatusCallback = Callable[[str, dict[str, Any]], None]


@dataclass
class _InferenceJob:
    run: Callable[[], Iterator[Any]]
    callback: QueueStatusCallback | None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    submitted_at: float = field(default_factory=time.time)
    cancelled: threading.Event = field(default_factory=threading.Event)


class InferenceScheduler:
    """Bounded FIFO admission with a fixed number of model workers."""

    def __init__(self, *, workers: int, queue_size: int) -> None:
        if workers < 1:
            raise ValueError("workers must be greater than zero")
        if queue_size < 1:
            raise ValueError("queue_size must be greater than zero")
        self.workers = workers
        self.queue_size = queue_size
        self._condition = threading.Condition()
        self._pending: list[_InferenceJob] = []
        self._active: dict[str, _InferenceJob] = {}
        self._closed = False

    @staticmethod
    def _notify(job: _InferenceJob, event: str, **details: Any) -> None:
        if job.callback is not None:
            try:
                job.callback(event, {"job_id": job.id, **details})
            except Exception:
                # Progress reporting must never change inference behavior.
                pass

    def _pending_updates_locked(self) -> list[tuple[_InferenceJob, int]]:
        return [(job, position) for position, job in enumerate(self._pending, start=1)]

    def _notify_pending(self, updates: list[tuple[_InferenceJob, int]]) -> None:
        for job, position in updates:
            self._notify(
                job,
                "queued",
                queue_position=position,
                queued=len(updates),
                capacity=self.queue_size,
            )

    def stream(
        self,
        run: Callable[[], Iterator[Any]],
        *,
        callback: QueueStatusCallback | None = None,
    ) -> Generator[Any, None, None]:
        job = _InferenceJob(run=run, callback=callback)
        started = False
        with self._condition:
            if self._closed:
                raise RuntimeError("Inference scheduler is closed")
            if len(self._active) < self.workers:
                self._active[job.id] = job
                started = True
                active = len(self._active)
            elif len(self._pending) >= self.queue_size:
                raise InferenceQueueFullError(
                    f"Inference queue is full ({self.queue_size} waiting requests). Please retry shortly."
                )
            else:
                self._pending.append(job)
                position = len(self._pending)
                queued = len(self._pending)
        if started:
            self._notify(job, "running", queue_position=0, active_workers=active)
        else:
            self._notify(
                job,
                "queued",
                queue_position=position,
                queued=queued,
                capacity=self.queue_size,
            )
        return self._consume(job)

    def _release(self, job: _InferenceJob) -> None:
        promoted: list[_InferenceJob] = []
        with self._condition:
            self._active.pop(job.id, None)
            while self._pending and len(self._active) < self.workers:
                candidate = self._pending.pop(0)
                if candidate.cancelled.is_set():
                    continue
                self._active[candidate.id] = candidate
                promoted.append(candidate)
            updates = self._pending_updates_locked()
            active = len(self._active)
            self._condition.notify_all()
        self._notify_pending(updates)
        for candidate in promoted:
            self._notify(candidate, "running", queue_position=0, active_workers=active)

    def _consume(self, job: _InferenceJob) -> Generator[Any, None, None]:
        iterator: Iterator[Any] | None = None
        try:
            with self._condition:
                while job.id not in self._active and not job.cancelled.is_set():
                    self._condition.wait()
            if job.cancelled.is_set():
                raise RuntimeError("Inference request was cancelled")
            iterator = job.run()
            yield from iterator
        finally:
            if iterator is not None:
                close = getattr(iterator, "close", None)
                if callable(close):
                    close()
            self._release(job)

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "workers": self.workers,
                "capacity": self.queue_size,
                "queued": len(self._pending),
                "active": len(self._active),
                "pending_job_ids": [job.id for job in self._pending],
                "active_job_ids": list(self._active),
            }

    def close(self) -> None:
        deadline = time.monotonic() + 30
        with self._condition:
            if self._closed:
                return
            self._closed = True
            for job in (*self._pending, *self._active.values()):
                job.cancelled.set()
            self._pending.clear()
            self._condition.notify_all()
            while self._active:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)


class ByteLRUCache:
    """A bounded, thread-safe LRU for immutable encoded conditioning bytes."""

    def __init__(self, *, max_entries: int, max_bytes: int) -> None:
        if max_entries < 0 or max_bytes < 0:
            raise ValueError("cache limits cannot be negative")
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self._values: OrderedDict[str, bytes] = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0

    def get_or_create(self, key: str, factory: Callable[[], bytes]) -> bytes:
        with self._lock:
            cached = self._values.get(key)
            if cached is not None:
                self._values.move_to_end(key)
                self._hits += 1
                return cached
            self._misses += 1
            value = factory()
            if not isinstance(value, bytes):
                raise TypeError("latent cache values must be bytes")
            if not self.max_entries or not self.max_bytes or len(value) > self.max_bytes:
                return value
            previous = self._values.pop(key, None)
            if previous is not None:
                self._bytes -= len(previous)
            self._values[key] = value
            self._bytes += len(value)
            while len(self._values) > self.max_entries or self._bytes > self.max_bytes:
                _, removed = self._values.popitem(last=False)
                self._bytes -= len(removed)
            return value

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._values),
                "bytes": self._bytes,
                "max_entries": self.max_entries,
                "max_bytes": self.max_bytes,
                "hits": self._hits,
                "misses": self._misses,
            }

    def clear(self) -> None:
        with self._lock:
            self._values.clear()
            self._bytes = 0
