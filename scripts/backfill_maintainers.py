"""Reclassify stored comments using the derived maintainer list.

`normalize.py` now consults `config/maintainers.yaml`, but the corpus was loaded
before that list existed, so everything already stored still carries the
classification GitHub's `authorAssociation` gave it. This corrects it in place
rather than reloading 316,685 rows.

Three tables carry the flag, and all three have to move together:

  * `review_comments.is_maintainer` is what the subset replication filters on,
    so it decides whose guidance the deployed agent can see at all.
  * `contributors.is_maintainer` is entity memory's view of the same fact.
  * `rule_evidence.is_maintainer` is denormalised from the comment at the time
    the evidence was recorded, and `rules.maintainer_evidence_count` is
    derived from it. Leaving those stale would make a rule's evidence
    breakdown disagree with the comments it points at.

Updated in batches keyed by author. A single statement covering every affected
row previously ran for fifteen minutes without finishing on a free-tier
cluster; small transactions finish.

    python scripts/backfill_maintainers.py --dry-run
    python scripts/backfill_maintainers.py
    python scripts/backfill_maintainers.py --local
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
from precedent.transform.maintainers import load_maintainers

log = logging.getLogger("precedent.backfill")

LOCAL_DSN = "postgresql://root@localhost:26257/precedent?sslmode=disable"

# Case-insensitive, because the list is stored lowercased and GitHub logins are
# not. An equality on lower(author) cannot use the author index, but this runs
# once per author rather than once per row.
COUNT_MISFLAGGED = text("""
    SELECT count(*) FROM review_comments
    WHERE repo_id = :repo_id AND lower(author) = :login AND NOT is_maintainer
""")

FLAG_COMMENTS = text("""
    UPDATE review_comments SET is_maintainer = true
    WHERE repo_id = :repo_id AND lower(author) = :login AND NOT is_maintainer
    LIMIT :batch
""")

FLAG_CONTRIBUTOR = text("""
    UPDATE contributors SET is_maintainer = true, updated_at = now()
    WHERE repo_id = :repo_id AND lower(login) = :login AND NOT is_maintainer
""")

# Evidence rows follow the comment they point at.
RESYNC_EVIDENCE = text("""
    UPDATE rule_evidence AS re SET is_maintainer = rc.is_maintainer
    FROM review_comments AS rc
    WHERE rc.repo_id = re.repo_id AND rc.id = re.comment_id
      AND re.repo_id = :repo_id AND re.is_maintainer != rc.is_maintainer
""")

RECOUNT_MAINTAINER_EVIDENCE = text("""
    UPDATE rules SET maintainer_evidence_count = (
        SELECT count(*) FROM rule_evidence re
        WHERE re.repo_id = rules.repo_id AND re.rule_id = rules.id AND re.is_maintainer
    ), updated_at = now()
    WHERE repo_id = :repo_id
""")


async def main_async(args: argparse.Namespace) -> int:
    known = load_maintainers(args.repo)
    if not known:
        log.error("no maintainer list; run scripts/derive_maintainers.py --write first")
        return 1

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

            pending = {}
            for login in sorted(known):
                n = (
                    await conn.execute(COUNT_MISFLAGGED, {"repo_id": repo_id, "login": login})
                ).scalar_one()
                if n:
                    pending[login] = n

        total = sum(pending.values())
        log.info(
            "%d comments to reclassify across %d people%s",
            total,
            len(pending),
            ""
            if not pending
            else ": "
            + ", ".join(f"{k} {v:,}" for k, v in sorted(pending.items(), key=lambda x: -x[1])[:8]),
        )
        if args.dry_run:
            return 0

        done = 0
        for login in pending:
            while True:

                async def flag(who=login) -> int:
                    async with engine.begin() as conn:
                        result = await conn.execute(
                            FLAG_COMMENTS,
                            {"repo_id": repo_id, "login": who, "batch": args.batch_size},
                        )
                        return result.rowcount

                moved = await with_retry(flag, description="flag comments")
                if not moved:
                    break
                done += moved
                elapsed = time.monotonic() - started
                log.info(
                    "%d/%d reclassified | %.0f rows/s | %s",
                    done,
                    total,
                    done / elapsed if elapsed else 0,
                    login,
                )

            async def flag_person(who=login) -> None:
                async with engine.begin() as conn:
                    await conn.execute(FLAG_CONTRIBUTOR, {"repo_id": repo_id, "login": who})

            await with_retry(flag_person, description="flag contributor")

        log.info("resyncing rule evidence")

        async def resync() -> int:
            async with engine.begin() as conn:
                moved = (await conn.execute(RESYNC_EVIDENCE, {"repo_id": repo_id})).rowcount
                await conn.execute(RECOUNT_MAINTAINER_EVIDENCE, {"repo_id": repo_id})
                return moved

        evidence_moved = await with_retry(resync, description="resync evidence")

        log.info(
            "reclassified %d comments and %d evidence rows in %.0fs",
            done,
            evidence_moved,
            time.monotonic() - started,
        )
    finally:
        await engine.dispose()
    return 0


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Reclassify stored comments by maintainer list.")
    parser.add_argument("--dsn", default=None)
    parser.add_argument("--local", action="store_true", help="Target the local Docker node.")
    parser.add_argument("--repo", default=settings.repo_slug)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.dsn = args.dsn or (LOCAL_DSN if args.local else settings.cockroach_dsn)
    if not args.dsn:
        parser.error("no DSN: use --local, pass --dsn, or set COCKROACH_DSN")

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s", stream=sys.stdout
    )
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
