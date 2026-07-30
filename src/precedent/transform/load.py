"""Write normalized records into CockroachDB.

Idempotent by construction: every comment upserts on (repo_id, github_node_id),
which is GitHub's own identifier. Reloading the whole corpus after a schema
change is therefore safe and needs no calls to the GitHub API.
"""

from __future__ import annotations

import logging
from dataclasses import asdict

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from precedent.db.retry import with_retry
from precedent.transform.normalize import CommentRecord, RejectRecord

log = logging.getLogger(__name__)

UPSERT_REPO = text("""
    INSERT INTO repos (owner, name)
    VALUES (:owner, :name)
    ON CONFLICT (owner, name) DO UPDATE SET name = excluded.name
    RETURNING id
""")

# The embedding is reset only when the body actually changed, so re-running the
# transform does not invalidate an expensive backfill. Touching the embedding
# column at all costs the column-family benefit on this path, which is the
# right trade: a stale embedding attached to edited text is silently wrong.
UPSERT_COMMENT = text("""
    INSERT INTO review_comments (
        repo_id, github_node_id, kind, pr_number, thread_id, in_reply_to,
        author, author_association, is_maintainer, file_path, "line",
        diff_hunk, body, url, created_at
    ) VALUES (
        :repo_id, :github_node_id, CAST(:kind AS comment_kind), :pr_number,
        :thread_id, :in_reply_to, :author, :author_association, :is_maintainer,
        :file_path, :line, :diff_hunk, :body, :url, :created_at
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
        embedding          = CASE
                                WHEN review_comments.body IS DISTINCT FROM excluded.body
                                THEN NULL
                                ELSE review_comments.embedding
                             END
""")

UPSERT_REJECT = text("""
    INSERT INTO ingest_rejects (repo_id, github_node_id, pr_number, author, reason)
    VALUES (:repo_id, :github_node_id, :pr_number, :author, :reason)
    ON CONFLICT (repo_id, github_node_id) DO UPDATE SET
        reason = excluded.reason,
        author = excluded.author
""")

# Derived wholesale from review_comments rather than accumulated during the
# load, so it is correct no matter how many times the transform is re-run or
# how the corpus is sliced.
REFRESH_CONTRIBUTORS = text("""
    INSERT INTO contributors (
        repo_id, login, is_maintainer, pr_count, comment_count,
        areas_touched, first_seen_at, last_seen_at, updated_at
    )
    SELECT
        repo_id,
        author,
        bool_or(is_maintainer),
        count(DISTINCT pr_number),
        count(*),
        COALESCE(
            array_agg(DISTINCT split_part(file_path, '/', 1))
                FILTER (WHERE file_path IS NOT NULL),
            ARRAY[]::STRING[]
        ),
        min(created_at),
        max(created_at),
        now()
    FROM review_comments
    WHERE repo_id = :repo_id AND author IS NOT NULL
    GROUP BY repo_id, author
    ON CONFLICT (repo_id, login) DO UPDATE SET
        is_maintainer = excluded.is_maintainer,
        pr_count      = excluded.pr_count,
        comment_count = excluded.comment_count,
        areas_touched = excluded.areas_touched,
        first_seen_at = excluded.first_seen_at,
        last_seen_at  = excluded.last_seen_at,
        updated_at    = now()
""")


async def ensure_repo(engine: AsyncEngine, owner: str, name: str) -> str:
    async def op() -> str:
        async with engine.begin() as conn:
            result = await conn.execute(UPSERT_REPO, {"owner": owner, "name": name})
            return str(result.scalar_one())

    return await with_retry(op, description="ensure_repo")


async def write_batch(
    engine: AsyncEngine,
    repo_id: str,
    comments: list[CommentRecord],
    rejects: list[RejectRecord],
) -> None:
    """One transaction per batch, retried as a unit on serialization failure."""
    # A single statement may not touch the same row twice: CockroachDB rejects
    # ON CONFLICT DO UPDATE that would affect one row more than once. Keeping
    # the last occurrence matches what a second pass would have written anyway.
    comment_rows = list(
        {c.github_node_id: {"repo_id": repo_id, **asdict(c)} for c in comments}.values()
    )
    reject_rows = list(
        {r.github_node_id: {"repo_id": repo_id, **asdict(r)} for r in rejects}.values()
    )

    async def op() -> None:
        async with engine.begin() as conn:
            if comment_rows:
                await conn.execute(UPSERT_COMMENT, comment_rows)
            if reject_rows:
                await conn.execute(UPSERT_REJECT, reject_rows)

    await with_retry(op, description=f"write_batch({len(comment_rows)} comments)")


async def refresh_contributors(engine: AsyncEngine, repo_id: str) -> int:
    async def op() -> int:
        async with engine.begin() as conn:
            result = await conn.execute(REFRESH_CONTRIBUTORS, {"repo_id": repo_id})
            return result.rowcount

    return await with_retry(op, description="refresh_contributors")
