"""Remove filled-in pull request templates from episodic memory.

`normalize.py` now rejects these at load time, but the corpus was loaded
before that filter existed. This clears what is already stored.

Deleted in small batches rather than one statement. The single-statement
version ran for fifteen minutes without finishing: each row cascades into
`rule_evidence` and has to be removed from the vector index, which is
expensive in request units, and the cluster is on a free tier with a zero
spend limit. Small transactions finish; one large one gets throttled.

Rules that lose evidence are recounted afterwards, so a rule built entirely on
template text ends up with zero evidence and zero confidence rather than
keeping a score it can no longer justify.

    python scripts/purge_templates.py --dry-run
    python scripts/purge_templates.py
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time

from sqlalchemy import text

from precedent.config import get_settings
from precedent.db.engine import create_engine
from precedent.db.retry import with_retry
from precedent.extract.confidence import score

log = logging.getLogger("precedent.purge")

MATCH = "%Replace xxxx with the%issue number%"

COUNT = text("SELECT count(*) FROM review_comments WHERE repo_id = :r AND body ILIKE :m")

# Selecting ids first keeps each delete a small, keyed operation rather than a
# scan inside a transaction.
PICK = text("""
    SELECT id FROM review_comments
    WHERE repo_id = :r AND body ILIKE :m
    LIMIT :n
""")

DELETE_BATCH = text("DELETE FROM review_comments WHERE repo_id = :r AND id = :i")

RECOUNT = text("""
    SELECT count(DISTINCT rc.pr_number) AS prs, count(DISTINCT rc.author) AS authors,
           count(*) AS evidence, count(*) FILTER (WHERE rc.is_maintainer) AS maintainer_evidence,
           min(rc.created_at) AS first_at, max(rc.created_at) AS last_at
    FROM rule_evidence re
    JOIN review_comments rc ON rc.repo_id = re.repo_id AND rc.id = re.comment_id
    WHERE re.repo_id = :r AND re.rule_id = :rule
""")

UPDATE_COUNTS = text("""
    UPDATE rules
    SET confidence = :confidence, evidence_count = :evidence_count,
        maintainer_evidence_count = :maintainer_evidence_count,
        first_evidence_at = :first_at, last_evidence_at = :last_at, updated_at = now()
    WHERE repo_id = :r AND id = :rule
""")

# A rule with nothing left to stand on is retired, not deleted.
RETIRE_UNSUPPORTED = text("""
    UPDATE rules SET status = 'retired',
        supersession_reason = 'all supporting evidence was pull request template text',
        updated_at = now()
    WHERE repo_id = :r AND id = :rule AND status = 'active'
""")


async def main_async(args: argparse.Namespace) -> int:
    engine = create_engine(args.dsn)
    owner, _, name = args.repo.partition("/")
    started = time.monotonic()

    try:
        async with engine.connect() as conn:
            repo_id = str(
                (
                    await conn.execute(
                        text("SELECT id FROM repos WHERE owner = :o AND name = :n"),
                        {"o": owner, "n": name},
                    )
                ).scalar_one()
            )
            total = (await conn.execute(COUNT, {"r": repo_id, "m": MATCH})).scalar_one()

        log.info("%d template comments to remove", total)
        if args.dry_run or total == 0:
            return 0

        removed = 0
        while True:
            async with engine.connect() as conn:
                ids = [
                    str(r[0])
                    for r in (
                        await conn.execute(PICK, {"r": repo_id, "m": MATCH, "n": args.batch_size})
                    ).all()
                ]
            if not ids:
                break

            async def drop(batch=ids) -> None:
                async with engine.begin() as conn:
                    await conn.execute(DELETE_BATCH, [{"r": repo_id, "i": i} for i in batch])

            await with_retry(drop, description="delete template batch")
            removed += len(ids)
            elapsed = time.monotonic() - started
            log.info(
                "%d/%d removed | %.0f rows/s | ~%.0f min left",
                removed,
                total,
                removed / elapsed if elapsed else 0,
                (total - removed) / (removed / elapsed) / 60 if removed and elapsed else 0,
            )

        log.info("recounting rules whose evidence may have changed")
        async with engine.connect() as conn:
            rule_ids = [
                str(r[0])
                for r in (
                    await conn.execute(
                        text("SELECT id FROM rules WHERE repo_id = :r"), {"r": repo_id}
                    )
                ).all()
            ]

        emptied = 0
        for rule_id in rule_ids:

            async def recount(rule=rule_id) -> bool:
                async with engine.begin() as conn:
                    row = (
                        (await conn.execute(RECOUNT, {"r": repo_id, "rule": rule})).mappings().one()
                    )
                    breakdown = score(
                        distinct_authors=row["authors"] or 0,
                        distinct_prs=row["prs"] or 0,
                        first_evidence_at=row["first_at"],
                        last_evidence_at=row["last_at"],
                    )
                    await conn.execute(
                        UPDATE_COUNTS,
                        {
                            "r": repo_id,
                            "rule": rule,
                            "confidence": breakdown.confidence,
                            "evidence_count": row["evidence"] or 0,
                            "maintainer_evidence_count": row["maintainer_evidence"] or 0,
                            "first_at": row["first_at"],
                            "last_at": row["last_at"],
                        },
                    )
                    if not row["evidence"]:
                        await conn.execute(RETIRE_UNSUPPORTED, {"r": repo_id, "rule": rule})
                        return True
                    return False

            if await with_retry(recount, description="recount rule"):
                emptied += 1

        log.info(
            "removed %d comments, retired %d rules left with no evidence, %.0fs",
            removed,
            emptied,
            time.monotonic() - started,
        )
    finally:
        await engine.dispose()
    return 0


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Purge pull request template comments.")
    parser.add_argument("--dsn", default=settings.cockroach_dsn or None)
    parser.add_argument("--repo", default=settings.repo_slug)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dsn:
        parser.error("no DSN: pass --dsn or set COCKROACH_DSN in .env")

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s", stream=sys.stdout
    )
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
