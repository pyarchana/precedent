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

# text-embedding-3-small accepts 8191 tokens per input. Counting tokens exactly
# would mean another dependency, so this caps on characters well below the
# limit. Review comments are far shorter than this; PR descriptions are what
# occasionally run long.
MAX_CHARS = 24_000


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
        max_attempts: int = 6,
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
            "input": [t[:MAX_CHARS] for t in texts],
            "dimensions": self.dimensions,
        }

        for attempt in range(1, self.max_attempts + 1):
            try:
                response = await self._client.post(OPENAI_EMBEDDINGS_URL, json=payload)
            except httpx.TransportError as exc:
                await self._backoff(attempt, f"transport error: {exc}")
                continue

            if response.status_code == 429 or response.status_code >= 500:
                retry_after = response.headers.get("retry-after")
                if retry_after:
                    await asyncio.sleep(float(retry_after) + 1)
                else:
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

        raise RuntimeError(f"embeddings request gave up after {self.max_attempts} attempts")

    @staticmethod
    async def _backoff(attempt: int, reason: str) -> None:
        delay = min(60.0, 2.0**attempt) + random.uniform(0, 2)
        log.warning("%s; retry %d in %.1fs", reason, attempt, delay)
        await asyncio.sleep(delay)
