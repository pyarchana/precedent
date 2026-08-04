"""Write extracted rules into semantic memory, merging rather than duplicating.

Three behaviours the plan asks for, in order of how often they fire:

  * **Merge.** A newly extracted rule that says what an existing rule already
    says adds its evidence to that rule instead of becoming a second copy.
    Evidence counts and confidence are then recomputed from the evidence
    actually recorded, never incremented blindly, so re-running the pipeline
    cannot inflate a rule's standing.

  * **Supersede.** A new rule that contradicts an existing one, and whose
    evidence is more recent, marks the old one superseded and points at the
    replacement. Copy-on-Write becoming the default is the obvious example:
    advice written before it is not wrong so much as out of date.

  * **Insert.** Everything else.

Nothing is ever hard-deleted. A superseded rule keeps its evidence and its
supersession reason, because "we used to say X, then this happened" is the
part that makes the memory explicable rather than merely correct.

## A note on the distance metric

CockroachDB's `<->` is **L2 distance**, not cosine. Thresholds here must be in
those units. Measuring similarity with cosine in Python and then comparing the
result against `<->` in SQL produces a threshold that silently never fires,
which is exactly what happened on the first attempt: eighteen rules were
inserted and none merged, including a pair that plainly said the same thing.

For unit-normalised embeddings, which is what the OpenAI models return, the
two are related by `L2 = sqrt(2 * cosine_distance)`. That was confirmed
against the database rather than assumed:

    duplicate pair       cosine 0.1815  ->  L2 0.6025
    nearest distinct     cosine 0.4258  ->  L2 0.9231

So the thresholds below are in L2, and the gap between 0.60 and 0.92 is what
makes 0.75 a safe place to draw the line.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from precedent.db.retry import with_retry
from precedent.embed.provider import EmbeddingProvider
from precedent.embed.vector import encode
from precedent.extract.confidence import score

log = logging.getLogger(__name__)

# Below this L2 distance, two statements say the same thing. Sits in the gap
# between the observed duplicate pair at 0.60 and the nearest genuinely
# distinct pair at 0.92.
MERGE_DISTANCE = 0.75

# Between the merge threshold and this, rules are related enough that one may
# contradict the other. Beyond it they are simply about different things and
# asking an LLM to compare them is wasted money.
CONTRADICTION_WINDOW = 1.10

Outcome = Literal["inserted", "merged", "superseded"]


@dataclass(slots=True)
class Candidate:
    statement: str
    rationale: str | None
    scope: str
    scope_pattern: str | None
    comment_ids: list[str]
    distinct_prs: int
    distinct_authors: int
    first_evidence_at: datetime | None
    last_evidence_at: datetime | None


@dataclass(slots=True)
class PersistResult:
    outcome: Outcome
    rule_id: str
    distance: float | None = None
    superseded_rule_id: str | None = None


# Two-stage, for the same reason as comment search: any predicate in the same
# SELECT as the vector ordering stops CockroachDB using the vector index. The
# status filter therefore lives in the outer query, which means over-fetching
# five candidates so a superseded rule at the front does not hide a live one.
NEAREST_RULE = text("""
    SELECT id, statement, confidence, last_evidence_at, distance
    FROM (
        SELECT id, statement, confidence, last_evidence_at, status,
               embedding <-> CAST(:query AS VECTOR(1536)) AS distance
        FROM rules
        WHERE repo_id = :repo_id
        ORDER BY embedding <-> CAST(:query AS VECTOR(1536))
        LIMIT 5
    )
    WHERE status = 'active'
    ORDER BY distance
    LIMIT 1
""")

INSERT_RULE = text("""
    INSERT INTO rules (
        repo_id, statement, rationale, scope, scope_pattern,
        confidence, evidence_count, maintainer_evidence_count,
        embedding, embedding_model, first_evidence_at, last_evidence_at
    ) VALUES (
        :repo_id, :statement, :rationale, CAST(:scope AS rule_scope), :scope_pattern,
        :confidence, :evidence_count, :maintainer_evidence_count,
        CAST(:embedding AS VECTOR(1536)), :embedding_model,
        :first_evidence_at, :last_evidence_at
    )
    RETURNING id
""")

INSERT_EVIDENCE = text("""
    INSERT INTO rule_evidence (repo_id, rule_id, comment_id, is_maintainer, weight)
    SELECT :repo_id, :rule_id, id, is_maintainer, 1.0
    FROM review_comments
    WHERE repo_id = :repo_id AND id = :comment_id
    ON CONFLICT (repo_id, rule_id, comment_id) DO NOTHING
""")

# Counts are derived from the evidence table rather than incremented, so that
# re-running extraction over the same comments cannot inflate a rule.
RECOUNT = text("""
    SELECT count(DISTINCT rc.pr_number)  AS prs,
           count(DISTINCT rc.author)     AS authors,
           count(*)                      AS evidence,
           count(*) FILTER (WHERE rc.is_maintainer) AS maintainer_evidence,
           min(rc.created_at)            AS first_at,
           max(rc.created_at)            AS last_at
    FROM rule_evidence re
    JOIN review_comments rc
      ON rc.repo_id = re.repo_id AND rc.id = re.comment_id
    WHERE re.repo_id = :repo_id AND re.rule_id = :rule_id
""")

UPDATE_COUNTS = text("""
    UPDATE rules
    SET confidence = :confidence,
        evidence_count = :evidence_count,
        maintainer_evidence_count = :maintainer_evidence_count,
        first_evidence_at = :first_evidence_at,
        last_evidence_at = :last_evidence_at,
        updated_at = now()
    WHERE repo_id = :repo_id AND id = :rule_id
""")

MARK_SUPERSEDED = text("""
    UPDATE rules
    SET status = 'superseded',
        superseded_by = :replacement_id,
        superseded_at = now(),
        supersession_reason = :reason,
        updated_at = now()
    WHERE repo_id = :repo_id AND id = :rule_id
""")


async def _record_evidence(engine: AsyncEngine, repo_id: str, rule_id: str, comment_ids) -> None:
    rows = [{"repo_id": repo_id, "rule_id": rule_id, "comment_id": cid} for cid in comment_ids]
    if not rows:
        return

    async def op() -> None:
        async with engine.begin() as conn:
            await conn.execute(INSERT_EVIDENCE, rows)

    await with_retry(op, description="record evidence")


async def _recompute(engine: AsyncEngine, repo_id: str, rule_id: str) -> float:
    """Recalculate counts and confidence from the evidence on record."""

    async def op() -> float:
        async with engine.begin() as conn:
            row = (
                (await conn.execute(RECOUNT, {"repo_id": repo_id, "rule_id": rule_id}))
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
                    "rule_id": rule_id,
                    "confidence": breakdown.confidence,
                    "evidence_count": row["evidence"] or 0,
                    "maintainer_evidence_count": row["maintainer_evidence"] or 0,
                    "first_evidence_at": row["first_at"],
                    "last_evidence_at": row["last_at"],
                },
            )
            return breakdown.confidence

    return await with_retry(op, description="recompute confidence")


async def persist_rule(
    engine: AsyncEngine,
    provider: EmbeddingProvider,
    *,
    repo_id: str,
    candidate: Candidate,
    merge_distance: float = MERGE_DISTANCE,
) -> PersistResult:
    """Insert, or merge into the nearest rule that already says this."""
    vector = encode((await provider.embed([candidate.statement]))[0])

    async with engine.connect() as conn:
        nearest = (
            (await conn.execute(NEAREST_RULE, {"repo_id": repo_id, "query": vector}))
            .mappings()
            .first()
        )

    if nearest is not None and float(nearest["distance"]) <= merge_distance:
        rule_id = str(nearest["id"])
        await _record_evidence(engine, repo_id, rule_id, candidate.comment_ids)
        await _recompute(engine, repo_id, rule_id)
        log.info(
            "merged into %s at distance %.3f: %s",
            rule_id[:8],
            nearest["distance"],
            candidate.statement[:70],
        )
        return PersistResult(outcome="merged", rule_id=rule_id, distance=float(nearest["distance"]))

    breakdown = score(
        distinct_authors=candidate.distinct_authors,
        distinct_prs=candidate.distinct_prs,
        first_evidence_at=candidate.first_evidence_at,
        last_evidence_at=candidate.last_evidence_at,
    )

    async def insert() -> str:
        async with engine.begin() as conn:
            return str(
                (
                    await conn.execute(
                        INSERT_RULE,
                        {
                            "repo_id": repo_id,
                            "statement": candidate.statement,
                            "rationale": candidate.rationale,
                            "scope": candidate.scope,
                            "scope_pattern": candidate.scope_pattern,
                            "confidence": breakdown.confidence,
                            "evidence_count": len(candidate.comment_ids),
                            "maintainer_evidence_count": len(candidate.comment_ids),
                            "embedding": vector,
                            "embedding_model": getattr(provider, "model", None),
                            "first_evidence_at": candidate.first_evidence_at,
                            "last_evidence_at": candidate.last_evidence_at,
                        },
                    )
                ).scalar_one()
            )

    rule_id = await with_retry(insert, description="insert rule")
    await _record_evidence(engine, repo_id, rule_id, candidate.comment_ids)
    await _recompute(engine, repo_id, rule_id)

    return PersistResult(
        outcome="inserted",
        rule_id=rule_id,
        distance=float(nearest["distance"]) if nearest is not None else None,
    )


async def supersede(
    engine: AsyncEngine,
    *,
    repo_id: str,
    old_rule_id: str,
    replacement_rule_id: str,
    reason: str,
) -> None:
    """Mark a rule superseded. It keeps its evidence and its history."""

    async def op() -> None:
        async with engine.begin() as conn:
            await conn.execute(
                MARK_SUPERSEDED,
                {
                    "repo_id": repo_id,
                    "rule_id": old_rule_id,
                    "replacement_id": replacement_rule_id,
                    "reason": reason,
                },
            )

    await with_retry(op, description="supersede rule")
    log.info("superseded %s with %s: %s", old_rule_id[:8], replacement_rule_id[:8], reason)
