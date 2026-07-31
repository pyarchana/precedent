"""Copy a curated slice of episodic memory from one cluster to another.

Used to move the demo subset from the local development cluster into
CockroachDB Cloud. Embeddings travel with the rows, which is the whole point:
re-embedding in the target would cost real money for vectors that already
exist.

The selection is deliberate rather than a size cap. Maintainer comments below
about 120 characters are overwhelmingly "LGTM", "thanks", "please rebase":
retrieval noise that carries no convention. What is worth deploying is what
the people who own the codebase said at length.

    python scripts/replicate_subset.py --target "$COCKROACH_DSN" --estimate
    python scripts/replicate_subset.py --target "$COCKROACH_DSN"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from precedent.config import get_settings
from precedent.db.engine import create_engine
from precedent.db.retry import with_retry
from precedent.transform.load import ensure_repo

log = logging.getLogger("precedent.replicate")

LOCAL_DSN = "postgresql://root@localhost:26257/precedent?sslmode=disable"

SELECT_BATCH = text("""
    SELECT id, github_node_id, kind::STRING AS kind, pr_number, thread_id,
           in_reply_to, author, author_association, is_maintainer, file_path,
           "line", diff_hunk, body, url, created_at,
           embedding::STRING AS embedding, embedding_model
    FROM review_comments
    WHERE repo_id = :repo_id
      AND is_maintainer
      AND length(body) >= :min_length
      AND id > :after
    ORDER BY id
    LIMIT :limit
""")

COUNT_SUBSET = text("""
    SELECT count(*), coalesce(sum(length(body)), 0)
    FROM review_comments
    WHERE repo_id = :repo_id AND is_maintainer AND length(body) >= :min_length
""")

# Like the transform's upsert, but carrying the embedding across rather than
# leaving it null for a backfill to pay for again. Written out in full rather
# than derived from the transform's statement: generating one query by string
# surgery on another is unreadable and breaks silently when the original moves.
UPSERT_WITH_EMBEDDING = text("""
    INSERT INTO review_comments (
        repo_id, github_node_id, kind, pr_number, thread_id, in_reply_to,
        author, author_association, is_maintainer, file_path, "line",
        diff_hunk, body, url, created_at,
        embedding, embedding_model, embedded_at
    ) VALUES (
        :repo_id, :github_node_id, CAST(:kind AS comment_kind), :pr_number,
        :thread_id, :in_reply_to, :author, :author_association, :is_maintainer,
        :file_path, :line, :diff_hunk, :body, :url, :created_at,
        CAST(:embedding AS VECTOR(1536)), :embedding_model, now()
    )
    ON CONFLICT (repo_id, github_node_id) DO UPDATE SET
        kind               = excluded.kind,
        pr_number          = excluded.pr_number,
        thread_id          = excluded.thread_id,
        in_reply_to        = excluded.in_reply_to,
        author             = excluded.author,
        author_association = excluded.author_association,
        is_maintainer      = excluded.is_maintainer,
        file_path          = excluded.file_path,
        "line"             = excluded."line",
        diff_hunk          = excluded.diff_hunk,
        body               = excluded.body,
        url                = excluded.url,
        created_at         = excluded.created_at,
        embedding          = excluded.embedding,
        embedding_model    = excluded.embedding_model,
        embedded_at        = now()
""")

COPY_CONTRIBUTORS = text("""
    INSERT INTO contributors (
        repo_id, login, is_maintainer, pr_count, comment_count,
        areas_touched, first_seen_at, last_seen_at, updated_at
    ) VALUES (
        :repo_id, :login, :is_maintainer, :pr_count, :comment_count,
        :areas_touched, :first_seen_at, :last_seen_at, now()
    )
    ON CONFLICT (repo_id, login) DO UPDATE SET
        is_maintainer = excluded.is_maintainer,
        pr_count      = excluded.pr_count,
        comment_count = excluded.comment_count,
        areas_touched = excluded.areas_touched,
        first_seen_at = excluded.first_seen_at,
        last_seen_at  = excluded.last_seen_at,
        updated_at    = now()
""")


async def replicate(
    *,
    source_dsn: str,
    target_dsn: str,
    owner: str,
    name: str,
    min_length: int,
    batch_size: int,
    estimate_only: bool,
) -> int:
    source = create_engine(source_dsn)
    target = create_engine(target_dsn)

    try:
        async with source.connect() as conn:
            source_repo_id = str(
                (
                    await conn.execute(
                        text("SELECT id FROM repos WHERE owner = :o AND name = :n"),
                        {"o": owner, "n": name},
                    )
                ).scalar_one()
            )
            # A full scan summing body lengths is the most expensive statement
            # here, and on a memory-constrained node it is the one that fails.
            # It only drives the progress display, so losing it should not stop
            # a replication that would otherwise work.
            try:
                rows, body_bytes = (
                    await conn.execute(
                        COUNT_SUBSET, {"repo_id": source_repo_id, "min_length": min_length}
                    )
                ).one()
            except DBAPIError as exc:
                if estimate_only:
                    raise
                log.warning("could not count the subset (%s); copying without a total", exc)
                rows, body_bytes = 0, 0

        if rows:
            # 1536 float32 values, plus the text, plus row overhead.
            estimate_mb = (rows * 1536 * 4 + float(body_bytes) * 1.2) / 1024 / 1024
            log.info(
                "%d comments at %d+ characters, roughly %.0f MB before the vector index",
                rows,
                min_length,
                estimate_mb,
            )
        if estimate_only:
            return 0

        target_repo_id = await ensure_repo(target, owner, name)
        log.info("target repo id %s", target_repo_id)

        copied = 0
        after = "00000000-0000-0000-0000-000000000000"
        started = time.monotonic()

        while True:
            async with source.connect() as conn:
                batch = (
                    (
                        await conn.execute(
                            SELECT_BATCH,
                            {
                                "repo_id": source_repo_id,
                                "min_length": min_length,
                                "after": after,
                                "limit": batch_size,
                            },
                        )
                    )
                    .mappings()
                    .all()
                )

            if not batch:
                break

            payload = []
            for row in batch:
                record = dict(row)
                # Ids are regenerated by the target's default, so the source id
                # is only used to page through the source in a stable order.
                after = str(record.pop("id"))
                record["repo_id"] = target_repo_id
                payload.append(record)

            async def write(rows_to_write=payload) -> None:
                async with target.begin() as conn:
                    await conn.execute(UPSERT_WITH_EMBEDDING, rows_to_write)

            await with_retry(write, description="replicate batch")
            copied += len(payload)

            elapsed = time.monotonic() - started
            rate = copied / elapsed if elapsed else 0
            if rows:
                log.info(
                    "%d/%d copied | %.0f rows/s | ~%.0f min left",
                    copied,
                    rows,
                    rate,
                    max(0.0, (rows - copied) / rate / 60) if rate else 0,
                )
            else:
                log.info("%d copied | %.0f rows/s", copied, rate)

        log.info("copying contributors")
        async with source.connect() as conn:
            contributors = (
                (
                    await conn.execute(
                        text(
                            "SELECT login, is_maintainer, pr_count, comment_count, "
                            "areas_touched, first_seen_at, last_seen_at "
                            "FROM contributors WHERE repo_id = :r"
                        ),
                        {"r": source_repo_id},
                    )
                )
                .mappings()
                .all()
            )

        contributor_rows = [{"repo_id": target_repo_id, **dict(c)} for c in contributors]
        for i in range(0, len(contributor_rows), 500):
            chunk = contributor_rows[i : i + 500]

            async def write_contributors(rows_to_write=chunk) -> None:
                async with target.begin() as conn:
                    await conn.execute(COPY_CONTRIBUTORS, rows_to_write)

            await with_retry(write_contributors, description="replicate contributors")

        log.info("done: %d comments, %d contributors", copied, len(contributor_rows))
    finally:
        await source.dispose()
        await target.dispose()
    return 0


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Replicate a demo subset to another cluster.")
    parser.add_argument("--source", default=LOCAL_DSN)
    parser.add_argument("--target", default=settings.cockroach_dsn or None)
    parser.add_argument("--owner", default=settings.target_repo_owner)
    parser.add_argument("--name", default=settings.target_repo_name)
    parser.add_argument("--min-length", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--estimate", action="store_true")
    args = parser.parse_args()

    if not args.target:
        parser.error("no target: pass --target or set COCKROACH_DSN in .env")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s | %(message)s",
        stream=sys.stdout,
    )

    return asyncio.run(
        replicate(
            source_dsn=args.source,
            target_dsn=args.target,
            owner=args.owner,
            name=args.name,
            min_length=args.min_length,
            batch_size=args.batch_size,
            estimate_only=args.estimate,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
