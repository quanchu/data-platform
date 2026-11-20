"""Optional per-run observations; ordinary library calls have no observer."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_observer: ContextVar[Callable[[dict], None] | None] = ContextVar(
    "digest_observer", default=None
)


@contextmanager
def observe(callback: Callable[[dict], None]) -> Iterator[None]:
    token = _observer.set(callback)
    try:
        yield
    finally:
        _observer.reset(token)


def emit(stage: str, *, status: str = "completed", spans=(), **data: Any) -> None:
    callback = _observer.get()
    if callback is not None:
        callback(
            {
                "stage": stage,
                "status": status,
                "spans": [{"start": start, "end": end} for start, end in spans],
                "data": data,
            }
        )
