"""
Progress reporting and cooperative cancellation shared by the pipeline and UI.

Long-running steps call `progress(fraction, message)` and `cancel.check()`
between units of work. The UI thread can set the cancel flag at any time; the
worker raises `AnalysisCancelled` at its next check-point.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

ProgressCallback = Callable[[float, str], None]


class AnalysisCancelled(RuntimeError):
    """Raised inside the worker when the user cancelled the analysis."""


class CancelToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        if self._event.is_set():
            raise AnalysisCancelled("Analysis cancelled by user")


@dataclass
class ProgressState:
    """Thread-safe progress holder the UI can poll."""

    fraction: float = 0.0
    message: str = ""
    stage: str = ""
    started_at: float = field(default_factory=time.time)
    history: list[tuple[float, float, str]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, fraction: float, message: str) -> None:
        with self._lock:
            self.fraction = float(min(max(fraction, 0.0), 1.0))
            self.message = message
            self.history.append((time.time() - self.started_at, self.fraction, message))

    def snapshot(self) -> tuple[float, str]:
        with self._lock:
            return self.fraction, self.message

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.started_at


def no_progress(_fraction: float, _message: str) -> None:
    """Default no-op progress callback."""


def scaled(progress: ProgressCallback, start: float, end: float) -> ProgressCallback:
    """Map a sub-task's 0..1 progress into the [start, end] slice of the parent."""

    def _inner(fraction: float, message: str) -> None:
        progress(start + (end - start) * min(max(fraction, 0.0), 1.0), message)

    return _inner
