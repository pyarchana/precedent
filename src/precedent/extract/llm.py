"""Chat completions with a hard spend ceiling.

The project runs on a fixed, unreplenishable budget, so the interesting part
of this module is not the request, it is the refusal to make one. Spend is
tracked from the usage the API reports, and the ceiling is checked before each
call rather than after, so the limit cannot be overshot by a request already
in flight.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)

CHAT_URL = "https://api.openai.com/v1/chat/completions"

# USD per million tokens.
PRICING = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4o": (2.50, 10.00),
}


class BudgetExhausted(RuntimeError):
    """The configured spend ceiling has been reached. Not an error condition."""


class QuotaExhausted(RuntimeError):
    """The account has no credit. Retrying will not help."""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    cached_calls: int = 0

    def cost(self, model: str) -> float:
        rate_in, rate_out = PRICING.get(model, PRICING["gpt-4o-mini"])
        return self.input_tokens / 1e6 * rate_in + self.output_tokens / 1e6 * rate_out


class ChatClient:
    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gpt-4o-mini",
        max_spend: float = 2.00,
        max_attempts: int = 6,
        timeout: float = 120.0,
    ) -> None:
        if not api_key:
            raise ValueError("no OpenAI API key. Set OPENAI_API_KEY in .env.")
        self.model = model
        self.max_spend = max_spend
        self.max_attempts = max_attempts
        self.usage = Usage()
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout, connect=15.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    @property
    def spent(self) -> float:
        return self.usage.cost(self.model)

    @property
    def remaining(self) -> float:
        return max(0.0, self.max_spend - self.spent)

    def check_budget(self) -> None:
        """Raise before spending, never after."""
        if self.spent >= self.max_spend:
            raise BudgetExhausted(
                f"spend ceiling reached: ${self.spent:.4f} of ${self.max_spend:.2f} "
                f"across {self.usage.calls} calls"
            )

    async def complete_json(self, messages: list[dict[str, str]]) -> dict:
        """One JSON-mode completion. Returns the parsed object."""
        self.check_budget()

        payload = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }

        for attempt in range(1, self.max_attempts + 1):
            try:
                response = await self._client.post(CHAT_URL, json=payload)
            except httpx.TransportError as exc:
                await self._backoff(attempt, f"transport error: {exc!r}")
                continue

            if response.status_code == 429:
                body = response.text
                if "insufficient_quota" in body:
                    raise QuotaExhausted(
                        "The OpenAI account has no remaining credit. A valid key is "
                        "not enough; the balance is empty."
                    )
                retry_after = response.headers.get("retry-after")
                delay = float(retry_after) + 1 if retry_after else None
                if delay:
                    log.warning("rate limited, waiting %.1fs (attempt %d)", delay, attempt)
                    await asyncio.sleep(delay)
                else:
                    await self._backoff(attempt, "rate limited")
                continue

            if response.status_code >= 500:
                await self._backoff(attempt, f"HTTP {response.status_code}")
                continue

            if response.status_code != 200:
                raise RuntimeError(
                    f"chat request failed with HTTP {response.status_code}: {response.text[:400]}"
                )

            data = response.json()
            usage = data.get("usage") or {}
            self.usage.input_tokens += usage.get("prompt_tokens", 0)
            self.usage.output_tokens += usage.get("completion_tokens", 0)
            self.usage.calls += 1

            content = data["choices"][0]["message"]["content"]
            try:
                return json.loads(content)
            except json.JSONDecodeError as exc:
                # JSON mode makes this rare, and a single malformed reply should
                # not abort a long run.
                log.warning("model returned unparseable JSON: %s", exc)
                return {"is_convention": False, "reason": "unparseable model output"}

        raise RuntimeError(f"chat request gave up after {self.max_attempts} attempts")

    @staticmethod
    async def _backoff(attempt: int, reason: str) -> None:
        delay = min(60.0, 2.0**attempt) + random.uniform(0, 2)
        log.warning("%s; retry %d in %.1fs", reason, attempt, delay)
        await asyncio.sleep(delay)
