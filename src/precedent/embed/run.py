"""Backfill embeddings for review comments.

Restartable and cheap to restart. Work is tracked by the rows themselves: a
comment is outstanding exactly while `embedding IS NULL`, which the partial
index `idx_rc_unembedded` makes cheap to ask. There is no separate cursor to
get out of step with the data.

Money is protected by `embedding_cache`, keyed on a hash of the text. Identical
bodies are paid for once no matter how often they recur or how many times the
job is restarted, and a crash between the API responding and the rows being
updated costs nothing on the retry.

    python -m precedent.embed.run --estimate
    python -m precedent.embed.run
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import sys
import time

from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncEngine

from precedent.config import get_settings
from precedent.db.engine import create_engine
from precedent.db.retry import with_retry
from precedent.embed.provider import (
    EmbeddingProvider,
    OpenAIEmbeddings,
    truncate_for_embedding,
)
from precedent.embed.vector import encode

log = logging.getLogger("precedent.embed")

SELECT_REPO = text("SELECT id FROM repos WHERE owner = :owner AND name = :name")

SELECT_PENDING = text("""
    SELECT id, body
    FROM review_comments
    WHERE repo_id = :repo_id AND embedding IS NULL
    LIMIT :limit
""")

COUNT_PENDING = text("""
    SELECT count(*), coalesce(sum(length(body)), 0)
    FROM review_comments
    WHERE repo_id = :repo_id AND embedding IS NULL
""")

SELECT_CACHED = text("""
        SELECT content_hash, embedding
        FROM embedding_cache
        WHERE model = :model AND content_hash IN :hashes
    """).bindparams(bindparam("hashes", expanding=True))

INSERT_CACHE = text("""
    INSERT INTO embedding_cache (model, content_hash, embedding)
    VALUES (:model, :content_hash, CAST(:embedding AS VECTOR(1536)))
    ON CONFLICT (model, content_hash) DO NOTHING
""")

UPDATE_EMBEDDING = text("""
    UPDATE review_comments
    SET embedding = CAST(:embedding AS VECTOR(1536)),
        embedding_model = :model,
        embedded_at = now()
    WHERE repo_id = :repo_id AND id = :id
""")


def content_hash(body: str) -> str:
    """Hash what is actually sent to the API, truncation included.

    Hashing the untruncated body would let two comments that differ only past
    the cut point claim different cache entries for an identical request.
    """
    return hashlib.sha256(truncate_for_embedding(body).encode("utf-8")).hexdigest()


async def _fetch_repo_id(engine: AsyncEngine, owner: str, name: str) -> str:
    async with engine.connect() as conn:
        row = (await conn.execute(SELECT_REPO, {"owner": owner, "name": name})).first()
    if row is None:
        raise RuntimeError(f"{owner}/{name} is not in the repos table. Run the transform first.")
    return str(row[0])


async def _embed_uncached(
    provider: EmbeddingProvider,
    texts: list[str],
    *,
    chunk_size: int,
    concurrency: int,
) -> list[list[float]]:
    """Embed distinct texts, several requests in flight at once."""
    chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]
    semaphore = asyncio.Semaphore(concurrency)

    async def one(chunk: list[str]) -> list[list[float]]:
        async with semaphore:
            return await provider.embed(chunk)

    results = await asyncio.gather(*(one(chunk) for chunk in chunks))
    return [vector for chunk_result in results for vector in chunk_result]


async def _process_batch(
    engine: AsyncEngine,
    provider: EmbeddingProvider,
    repo_id: str,
    rows: list[tuple[str, str]],
    *,
    chunk_size: int,
    concurrency: int,
) -> tuple[int, int]:
    """Embed one batch of rows. Returns (rows written, texts actually charged for)."""
    hashes = {row_id: content_hash(body) for row_id, body in rows}
    distinct = {h: body for (row_id, body), h in zip(rows, hashes.values(), strict=True)}

    async with engine.connect() as conn:
        cached_rows = (
            await conn.execute(
                SELECT_CACHED,
                {"model": provider.model, "hashes": list(distinct)},
            )
        ).all()
    vectors: dict[str, str] = {h: str(v) for h, v in cached_rows}

    missing = [h for h in distinct if h not in vectors]
    if missing:
        fresh = await _embed_uncached(
            provider,
            [distinct[h] for h in missing],
            chunk_size=chunk_size,
            concurrency=concurrency,
        )
        cache_rows = []
        for h, vector in zip(missing, fresh, strict=True):
            encoded = encode(vector)
            vectors[h] = encoded
            cache_rows.append({"model": provider.model, "content_hash": h, "embedding": encoded})

        async def write_cache() -> None:
            async with engine.begin() as conn:
                await conn.execute(INSERT_CACHE, cache_rows)

        # Cached before the rows are updated, so a crash in between still
        # leaves the paid-for vectors recoverable.
        await with_retry(write_cache, description="cache embeddings")

    updates = [
        {
            "repo_id": repo_id,
            "id": row_id,
            "embedding": vectors[hashes[row_id]],
            "model": provider.model,
        }
        for row_id, _ in rows
    ]

    async def write_rows() -> None:
        async with engine.begin() as conn:
            await conn.execute(UPDATE_EMBEDDING, updates)

    await with_retry(write_rows, description="write embeddings")
    return len(updates), len(missing)


async def backfill(
    *,
    dsn: str,
    owner: str,
    name: str,
    batch_size: int,
    chunk_size: int,
    concurrency: int,
    limit: int | None,
    estimate_only: bool,
) -> int:
    settings = get_settings()
    engine = create_engine(dsn)
    provider: EmbeddingProvider | None = None

    try:
        repo_id = await _fetch_repo_id(engine, owner, name)

        async with engine.connect() as conn:
            pending, total_chars = (await conn.execute(COUNT_PENDING, {"repo_id": repo_id})).one()

        # Roughly four characters per token for English prose. CockroachDB
        # returns sum() over an integer as DECIMAL, which will not divide by a
        # float, so convert before doing arithmetic.
        tokens = float(total_chars) / 4
        log.info(
            "%d comments need embedding, about %.1fM tokens, roughly $%.2f at "
            "text-embedding-3-small pricing before cache hits",
            pending,
            tokens / 1e6,
            tokens / 1e6 * 0.02,
        )
        if estimate_only or pending == 0:
            return 0

        provider = OpenAIEmbeddings(
            settings.openai_api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dim,
        )

        written = charged = 0
        started = time.monotonic()

        while True:
            if limit is not None and written >= limit:
                log.info("reached --limit=%d", limit)
                break

            take = batch_size if limit is None else min(batch_size, limit - written)
            async with engine.connect() as conn:
                rows = [
                    (str(r[0]), r[1])
                    for r in (
                        await conn.execute(SELECT_PENDING, {"repo_id": repo_id, "limit": take})
                    ).all()
                ]

            if not rows:
                log.info("nothing left to embed")
                break

            batch_written, batch_charged = await _process_batch(
                engine,
                provider,
                repo_id,
                rows,
                chunk_size=chunk_size,
                concurrency=concurrency,
            )
            written += batch_written
            charged += batch_charged

            elapsed = time.monotonic() - started
            rate = written / elapsed if elapsed else 0
            remaining = max(0, pending - written)
            log.info(
                "%d/%d embedded | %d sent to the API (%.0f%% cache hits) | "
                "%.0f rows/s | ~%.0f min left",
                written,
                pending,
                charged,
                100 * (1 - charged / written) if written else 0,
                rate,
                remaining / rate / 60 if rate else 0,
            )

        log.info("done: %d rows embedded, %d texts charged for", written, charged)
    finally:
        if provider is not None:
            await provider.aclose()
        await engine.dispose()
    return 0


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Backfill review comment embeddings.")
    parser.add_argument("--dsn", default=settings.cockroach_dsn or None)
    parser.add_argument("--owner", default=settings.target_repo_owner)
    parser.add_argument("--name", default=settings.target_repo_name)
    parser.add_argument("--batch-size", type=int, default=512, help="Rows claimed per round.")
    parser.add_argument("--chunk-size", type=int, default=128, help="Texts per API request.")
    parser.add_argument("--concurrency", type=int, default=4, help="Requests in flight.")
    parser.add_argument("--limit", type=int, default=None, help="Stop after N rows.")
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Report how much is outstanding and what it would cost, then stop.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    if not args.dsn:
        parser.error("no DSN: pass --dsn or set COCKROACH_DSN in .env")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s | %(message)s",
        stream=sys.stdout,
    )
    # One line per HTTP request buries the progress output.
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return asyncio.run(
        backfill(
            dsn=args.dsn,
            owner=args.owner,
            name=args.name,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            concurrency=args.concurrency,
            limit=args.limit,
            estimate_only=args.estimate,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
