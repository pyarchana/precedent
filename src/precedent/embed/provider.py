"""Embedding providers.

Kept behind a small protocol so the corpus can be re-embedded with a different
model later without touching the backfill driver. Note that swapping to a model
of a different width also requires altering the VECTOR columns, which is why
`dimensions` is part of the interface and checked against the schema.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Protocol, runtime_checkable

import httpx

log = logging.getLogger(__name__)

OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"

# text-embedding-3-small rejects inputs over 8192 tokens outright, with a 400
# that fails the whole request and therefore the whole batch.
#
# Counting characters instead of tokens does not work here. The usual "four
# characters per token" rule holds for prose, but this corpus is full of diff
# hunks, tracebacks and code, which tokenize closer to two characters per
# token. A 24,000 character cap looked like a comfortable 6,000 tokens and was
# in fact over the limit for exactly the comments most worth embedding.
MAX_TOKENS = 8_000

# Fallback only, for when tiktoken is unavailable. Deliberately pessimistic.
MAX_CHARS = 12_000

try:
    import tiktoken

    _ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:  # noqa: BLE001 - any failure here just means the fallback
    _ENCODING = None


def truncate_for_embedding(value: str) -> str:
    """Cut text down to something the embeddings endpoint will accept.

    Short inputs are returned unchanged, which keeps their content hash stable
    across changes to this function, so the embedding cache survives a fix here
    and only long comments have to be paid for again.
    """
    if _ENCODING is None:
        return value[:MAX_CHARS]
    tokens = _ENCODING.encode(value, disallowed_special=())
    if len(tokens) <= MAX_TOKENS:
        return value
    return _ENCODING.decode(tokens[:MAX_TOKENS])


def _is_quota_error(response: httpx.Response) -> bool:
    try:
        error = response.json().get("error") or {}
    except ValueError:
        return False
    return "insufficient_quota" in (error.get("code") or "", error.get("type") or "")


class QuotaExhausted(RuntimeError):
    """The account has no credit left.

    Arrives as HTTP 429, the same status as a rate limit, but retrying it is
    pointless: it will still be true in an hour. Worth its own type so the
    backfill stops immediately with an actionable message instead of backing
    off six times against a wall.
    """


@runtime_checkable
class EmbeddingProvider(Protocol):
    model: str
    dimensions: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbeddings:
    """Talks to the embeddings endpoint directly rather than via the SDK.

    One less dependency, and the retry behaviour we need here is the same
    backoff already used against GitHub.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        max_attempts: int = 10,
        timeout: float = 120.0,
    ) -> None:
        if not api_key:
            raise ValueError(
                "no OpenAI API key. Set OPENAI_API_KEY in .env before running the backfill."
            )
        self.model = model
        self.dimensions = dimensions
        self.max_attempts = max_attempts
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout, connect=15.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch, preserving input order."""
        if not texts:
            return []

        payload = {
            "model": self.model,
            "input": [truncate_for_embedding(t) for t in texts],
            "dimensions": self.dimensions,
        }

        for attempt in range(1, self.max_attempts + 1):
            try:
                response = await self._client.post(OPENAI_EMBEDDINGS_URL, json=payload)
            except httpx.TransportError as exc:
                # repr, because these often stringify to nothing at all.
                await self._backoff(attempt, f"transport error: {exc!r}")
                continue

            if response.status_code == 429:
                if _is_quota_error(response):
                    raise QuotaExhausted(
                        "The OpenAI account has no remaining quota. A valid key is not "
                        "enough: the account needs credit. Add a payment method and buy "
                        "credits at https://platform.openai.com/settings/organization/billing"
                    )
                retry_after = response.headers.get("retry-after")
                if retry_after:
                    delay = float(retry_after) + 1
                    # Logged rather than slept through quietly: six silent
                    # retries followed by a bare "gave up" tells you nothing
                    # about why, which is exactly what happened the first time.
                    log.warning(
                        "rate limited, honouring Retry-After=%.1fs (attempt %d/%d)",
                        delay,
                        attempt,
                        self.max_attempts,
                    )
                    await asyncio.sleep(delay)
                else:
                    await self._backoff(attempt, "rate limited without Retry-After")
                continue

            if response.status_code >= 500:
                await self._backoff(attempt, f"HTTP {response.status_code}")
                continue

            if response.status_code != 200:
                raise RuntimeError(
                    f"embeddings request failed with HTTP {response.status_code}: "
                    f"{response.text[:400]}"
                )

            data = response.json()["data"]
            # The API documents index ordering, but the whole pipeline depends
            # on lining vectors up with row ids, so sort rather than trust it.
            data.sort(key=lambda item: item["index"])
            vectors = [item["embedding"] for item in data]

            if len(vectors) != len(texts):
                raise RuntimeError(f"asked for {len(texts)} embeddings and got {len(vectors)}")
            return vectors

        raise RuntimeError(
            f"embeddings request gave up after {self.max_attempts} attempts. "
            "If the log shows repeated rate limiting, lower --concurrency or "
            "--chunk-size: a newly funded account starts on a low usage tier."
        )

    @staticmethod
    async def _backoff(attempt: int, reason: str) -> None:
        delay = min(60.0, 2.0**attempt) + random.uniform(0, 2)
        log.warning("%s; retry %d in %.1fs", reason, attempt, delay)
        await asyncio.sleep(delay)
