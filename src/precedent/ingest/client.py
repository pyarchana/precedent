"""A GitHub GraphQL client that survives an all-night run.

Three failure modes matter here, and they need different responses:

  * primary rate limit:   the budget in `rateLimit.remaining` is spent.
    Sleep until `resetAt`. Predictable, so we pre-empt it rather than
    waiting to be refused.
  * secondary rate limit: GitHub decided we were being abusive. Comes back
    as 403/429, sometimes with `Retry-After`. Back off exponentially.
  * query timeout:        the query itself was too expensive to serve in
    ~10s. Retrying identically will fail identically, so the caller shrinks
    the page and tries again.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Self

import httpx

log = logging.getLogger(__name__)

GRAPHQL_URL = "https://api.github.com/graphql"

# Sleep rather than spend the last of the budget; leaves room for a retry.
RATE_LIMIT_FLOOR = 100


class QueryTooExpensive(RuntimeError):
    """GitHub could not serve the query in time. Ask for less."""


class GitHubGraphQLError(RuntimeError):
    """A GraphQL error that is not a timeout and not a rate limit."""


@dataclass
class RateLimit:
    limit: int
    cost: int
    remaining: int
    reset_at: datetime

    @classmethod
    def from_payload(cls, data: dict[str, Any]) -> RateLimit | None:
        rl = data.get("rateLimit")
        if not rl:
            return None
        return cls(
            limit=rl["limit"],
            cost=rl["cost"],
            remaining=rl["remaining"],
            reset_at=datetime.fromisoformat(rl["resetAt"]),
        )

    def seconds_until_reset(self) -> float:
        return max(0.0, (self.reset_at - datetime.now(UTC)).total_seconds())


def _is_timeout_error(errors: list[dict[str, Any]]) -> bool:
    for err in errors:
        msg = (err.get("message") or "").lower()
        if "timeout" in msg or "timed out" in msg:
            return True
    return False


class GitHubGraphQL:
    def __init__(self, token: str, *, max_attempts: int = 8) -> None:
        self._client = httpx.Client(
            headers={
                "Authorization": f"bearer {token}",
                "Accept": "application/vnd.github+json",
                "User-Agent": "precedent-ingest",
            },
            timeout=httpx.Timeout(60.0, connect=15.0),
        )
        self.max_attempts = max_attempts
        self.last_rate_limit: RateLimit | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def execute(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        """Run a query, retrying transient failures. Returns the `data` block.

        Raises QueryTooExpensive when the caller should retry with smaller
        page sizes.
        """
        self._respect_primary_limit()

        for attempt in range(1, self.max_attempts + 1):
            try:
                resp = self._client.post(GRAPHQL_URL, json={"query": query, "variables": variables})
            except httpx.TransportError as exc:
                self._sleep_backoff(attempt, f"transport error: {exc}")
                continue

            if resp.status_code in (403, 429):
                self._handle_secondary_limit(resp, attempt)
                continue

            if resp.status_code >= 500:
                self._sleep_backoff(attempt, f"server error {resp.status_code}")
                continue

            if resp.status_code != 200:
                raise GitHubGraphQLError(f"HTTP {resp.status_code} from GitHub: {resp.text[:500]}")

            payload = resp.json()
            errors = payload.get("errors") or []

            if errors and _is_timeout_error(errors):
                raise QueryTooExpensive(errors[0].get("message", "query timed out"))

            data = payload.get("data")

            # Partial data with errors is common on big repos (a single
            # unreachable node). Keep the data, record the damage.
            if errors and data:
                log.warning(
                    "partial response, %d error(s); first: %s",
                    len(errors),
                    errors[0].get("message"),
                )
            elif errors:
                raise GitHubGraphQLError(str(errors[:3]))

            if data is None:
                raise GitHubGraphQLError(f"no data in response: {str(payload)[:500]}")

            self.last_rate_limit = RateLimit.from_payload(data)
            if self.last_rate_limit:
                log.debug(
                    "cost=%d remaining=%d",
                    self.last_rate_limit.cost,
                    self.last_rate_limit.remaining,
                )
            return data

        raise GitHubGraphQLError(f"gave up after {self.max_attempts} attempts")

    def _respect_primary_limit(self) -> None:
        rl = self.last_rate_limit
        if rl is None or rl.remaining > RATE_LIMIT_FLOOR:
            return
        wait = rl.seconds_until_reset() + 5
        log.warning(
            "primary rate limit near exhaustion (%d left); sleeping %.0fs until %s",
            rl.remaining,
            wait,
            rl.reset_at.isoformat(),
        )
        time.sleep(wait)
        # Budget is refilled; stop gating on the stale reading.
        self.last_rate_limit = None

    def _handle_secondary_limit(self, resp: httpx.Response, attempt: int) -> None:
        retry_after = resp.headers.get("retry-after")
        if retry_after:
            try:
                wait = float(retry_after)
            except ValueError:
                wait = 60.0
            log.warning("secondary rate limit; Retry-After=%.0fs", wait)
            time.sleep(wait + 1)
            return

        # No Retry-After. If the primary budget is what ran out, wait for reset.
        reset = resp.headers.get("x-ratelimit-reset")
        remaining = resp.headers.get("x-ratelimit-remaining")
        if reset and remaining == "0":
            wait = max(0.0, float(reset) - time.time()) + 5
            log.warning("primary limit hit via header; sleeping %.0fs", wait)
            time.sleep(wait)
            return

        self._sleep_backoff(attempt, f"HTTP {resp.status_code} (secondary limit)")

    @staticmethod
    def _sleep_backoff(attempt: int, reason: str) -> None:
        wait = min(300.0, 2.0**attempt) + random.uniform(0, 5)
        log.warning("%s; retry %d in %.1fs", reason, attempt, wait)
        time.sleep(wait)
