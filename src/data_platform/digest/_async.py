"""Bounded, ordered async work with complete failure and cancellation cleanup."""

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from typing import cast


def positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


async def ordered_map[T, R](
    function: Callable[[T], Awaitable[R]],
    items: Iterable[T],
    *,
    max_concurrency: int,
) -> list[R]:
    """Return input-ordered results; never leave workers running after exit."""
    positive_int(max_concurrency, "max_concurrency")
    values = list(items)
    results: list[R | None] = [None] * len(values)
    pending = iter(enumerate(values))

    async def worker():
        for index, value in pending:
            results[index] = await function(value)

    tasks = [
        asyncio.create_task(worker()) for _ in range(min(max_concurrency, len(values)))
    ]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return cast(list[R], results)
