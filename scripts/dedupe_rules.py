"""Merge rules already in memory that say the same thing.

Deduplication runs when a rule is written, comparing it against what is
already there. That is not enough on its own: the comparison is only as good
as the judge prompt at the time, and tightening the prompt afterwards leaves
the earlier duplicates in place.

The audit found eleven active rules all telling contributors to put the GitHub
issue number somewhere. Several were the same convention stated with different
precision, which the judge had been calling "compatible" rather than "same".

This walks existing rules newest-confidence-first and merges any pair the
judge now considers equivalent. Evidence moves to the surviving rule and
confidence is recomputed from it, so nothing is lost and nothing is inflated.

    python scripts/dedupe_rules.py --dry-run
    python scripts/dedupe_rules.py
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from sqlalchemy import text

from precedent.config import get_settings
from precedent.db.engine import create_engine
from precedent.db.retry import with_retry
from precedent.extract import contradiction
from precedent.extract.llm import BudgetExhausted, ChatClient
from precedent.extract.persist import CONSIDER_DISTANCE

log = logging.getLogger("precedent.dedupe")

ACTIVE_RULES = text("""
    SELECT id, statement, confidence, last_evidence_at, embedding::STRING AS embedding
    FROM rules
    WHERE repo_id = :repo_id AND status = 'active'
    ORDER BY confidence DESC, evidence_count DESC
""")

NEIGHBOURS = text("""
    SELECT id, statement, last_evidence_at, distance
    FROM (
        SELECT id, statement, last_evidence_at, status,
               embedding <-> CAST(:query AS VECTOR(1536)) AS distance
        FROM rules
        WHERE repo_id = :repo_id
        ORDER BY embedding <-> CAST(:query AS VECTOR(1536))
        LIMIT 12
    )
    WHERE status = 'active' AND id != :self_id AND distance <= :max_distance
    ORDER BY distance
""")

# Evidence moves rather than being copied, so the losing rule ends up with
# none and the surviving one carries the full history.
MOVE_EVIDENCE = text("""
    UPDATE rule_evidence SET rule_id = :winner
    WHERE repo_id = :repo_id AND rule_id = :loser
      AND comment_id NOT IN (
          SELECT comment_id FROM rule_evidence
          WHERE repo_id = :repo_id AND rule_id = :winner
      )
""")

DROP_LEFTOVER_EVIDENCE = text("""
    DELETE FROM rule_evidence WHERE repo_id = :repo_id AND rule_id = :loser
""")

# The loser is retired rather than deleted, and points at what absorbed it,
# so "why did this rule disappear" has an answer.
RETIRE_MERGED = text("""
    UPDATE rules
    SET status = 'retired',
        supersession_reason = :reason,
        updated_at = now()
    WHERE repo_id = :repo_id AND id = :loser
""")

RECOUNT = text("""
    SELECT count(DISTINCT rc.pr_number) AS prs, count(DISTINCT rc.author) AS authors,
           count(*) AS evidence, count(*) FILTER (WHERE rc.is_maintainer) AS maintainer_evidence,
           min(rc.created_at) AS first_at, max(rc.created_at) AS last_at
    FROM rule_evidence re
    JOIN review_comments rc ON rc.repo_id = re.repo_id AND rc.id = re.comment_id
    WHERE re.repo_id = :repo_id AND re.rule_id = :rule_id
""")

UPDATE_COUNTS = text("""
    UPDATE rules
    SET confidence = :confidence, evidence_count = :evidence_count,
        maintainer_evidence_count = :maintainer_evidence_count,
        first_evidence_at = :first_evidence_at, last_evidence_at = :last_evidence_at,
        updated_at = now()
    WHERE repo_id = :repo_id AND id = :rule_id
""")


async def main_async(args: argparse.Namespace) -> int:
    from precedent.extract.confidence import score

    settings = get_settings()
    engine = create_engine(args.dsn)
    client = ChatClient(settings.openai_api_key, model=args.model, max_spend=args.max_spend)

    owner, _, name = args.repo.partition("/")
    merged = compared = 0

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
            rules = (await conn.execute(ACTIVE_RULES, {"repo_id": repo_id})).mappings().all()

        log.info("%d active rules", len(rules))
        absorbed: set[str] = set()

        for rule in rules:
            winner = str(rule["id"])
            if winner in absorbed:
                continue

            async with engine.connect() as conn:
                neighbours = (
                    (
                        await conn.execute(
                            NEIGHBOURS,
                            {
                                "repo_id": repo_id,
                                "query": rule["embedding"],
                                "self_id": winner,
                                "max_distance": args.max_distance,
                            },
                        )
                    )
                    .mappings()
                    .all()
                )

            for neighbour in neighbours:
                loser = str(neighbour["id"])
                if loser in absorbed:
                    continue

                compared += 1
                try:
                    verdict = await client.complete_json(
                        contradiction.build_messages(
                            old_statement=rule["statement"],
                            new_statement=neighbour["statement"],
                            old_date="unknown",
                            new_date="unknown",
                        )
                    )
                except BudgetExhausted as exc:
                    log.warning("stopping: %s", exc)
                    return 0

                if verdict.get("relation") != "same":
                    continue

                merged += 1
                absorbed.add(loser)
                log.info(
                    "merge at %.3f\n  keep: %s\n  drop: %s",
                    neighbour["distance"],
                    rule["statement"][:88],
                    neighbour["statement"][:88],
                )
                if args.dry_run:
                    continue

                async def do_merge(winner=winner, loser=loser) -> None:
                    async with engine.begin() as conn:
                        await conn.execute(
                            MOVE_EVIDENCE,
                            {"repo_id": repo_id, "winner": winner, "loser": loser},
                        )
                        await conn.execute(
                            DROP_LEFTOVER_EVIDENCE, {"repo_id": repo_id, "loser": loser}
                        )
                        await conn.execute(
                            RETIRE_MERGED,
                            {
                                "repo_id": repo_id,
                                "loser": loser,
                                "reason": f"merged into {winner}",
                            },
                        )
                        row = (
                            (await conn.execute(RECOUNT, {"repo_id": repo_id, "rule_id": winner}))
                            .mappings()
                            .one()
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
                                "repo_id": repo_id,
                                "rule_id": winner,
                                "confidence": breakdown.confidence,
                                "evidence_count": row["evidence"] or 0,
                                "maintainer_evidence_count": row["maintainer_evidence"] or 0,
                                "first_evidence_at": row["first_at"],
                                "last_evidence_at": row["last_at"],
                            },
                        )

                await with_retry(do_merge, description="merge rules")
    finally:
        await client.aclose()
        await engine.dispose()

    verb = "would merge" if args.dry_run else "merged"
    log.info("%s %d rules after %d comparisons, $%.4f", verb, merged, compared, client.spent)
    return 0


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Merge equivalent rules already in memory.")
    parser.add_argument("--dsn", default=settings.cockroach_dsn or None)
    parser.add_argument("--repo", default=settings.repo_slug)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max-spend", type=float, default=0.10)
    parser.add_argument("--max-distance", type=float, default=CONSIDER_DISTANCE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dsn:
        parser.error("no DSN: pass --dsn or set COCKROACH_DSN in .env")

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s", stream=sys.stdout
    )
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
