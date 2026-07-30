"""Retry helper for CockroachDB serialization failures.

CockroachDB runs SERIALIZABLE by default, so a transaction that loses a race is
refused with SQLSTATE 40001 and is expected to be retried by the client. This is
normal operation, not an error condition: any write path that does not retry
will fail under concurrency.

Day 19 hardens this with structured logging and metrics. This is the core of it.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from sqlalchemy.exc import DBAPIError

log = logging.getLogger(__name__)

T = TypeVar("T")

SERIALIZATION_FAILURE = "40001"


def _sqlstate(exc: BaseException) -> str | None:
    """Dig the SQLSTATE out, whichever layer wrapped the driver error."""
    seen = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        for attr in ("sqlstate", "pgcode"):
            code = getattr(current, attr, None)
            if code:
                return str(code)
        current = getattr(current, "orig", None) or current.__cause__
    return None


def is_retryable(exc: BaseException) -> bool:
    return _sqlstate(exc) == SERIALIZATION_FAILURE


async def with_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 5,
    base_delay: float = 0.05,
    description: str = "transaction",
) -> T:
    """Run an async operation, retrying it on 40001.

    `operation` must start its own transaction, because a transaction that hit
    a serialization failure is dead and cannot be reused.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return await operation()
        except DBAPIError as exc:
            if not is_retryable(exc) or attempt == max_attempts:
                raise
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, base_delay)
            log.warning(
                "%s hit a serialization failure (attempt %d/%d); retrying in %.3fs",
                description,
                attempt,
                max_attempts,
                delay,
            )
            await asyncio.sleep(delay)

    raise AssertionError("unreachable")
