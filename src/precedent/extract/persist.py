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

## Why distance alone cannot decide a merge

Embeddings place opposites close together. "Use single quotes for strings in
Cython files" and "Use double quotes for strings in Cython files" sit 0.32
apart, far nearer than the two genuine duplicates at 0.60, because they share
a subject, a structure and almost every word. They are also exact opposites.

An earlier version merged on distance alone. Feeding it a rule that reversed
an existing convention did not supersede that convention, it was absorbed as
*further evidence for* it, raising the confidence of the rule it contradicted.

That is precisely the failure the correction loop must not have: a
maintainer's correction would strengthen the thing being corrected. So
distance now only selects candidates to compare, and a model decides whether
they agree, conflict, or merely resemble each other. Merges are rare enough
that the cost of asking is negligible.
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

# Distance decides which rules are worth comparing. It does NOT decide the
# outcome. See the note on antonyms below.
#
# Anything nearer than this is a candidate for merging or superseding; beyond
# it, rules are about different things and comparing them is wasted money.
CONSIDER_DISTANCE = 1.10

# Nearer than this, a rule is close enough that merging it without asking
# would be actively dangerous rather than merely imprecise.
MERGE_DISTANCE = 0.75

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
    # Returned so a caller can look for contradictions without paying to
    # embed the same statement a second time.
    statement_vector: str | None = None


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

# Rules near enough to be about the same subject, but not near enough to be
# the same rule. Excludes the rule just inserted, which is at distance zero
# from itself.
NEIGHBOURS_IN_BAND = text("""
    SELECT id, statement, last_evidence_at, distance
    FROM (
        SELECT id, statement, last_evidence_at, status,
               embedding <-> CAST(:query AS VECTOR(1536)) AS distance
        FROM rules
        WHERE repo_id = :repo_id
        ORDER BY embedding <-> CAST(:query AS VECTOR(1536))
        LIMIT 10
    )
    WHERE status = 'active'
      AND distance > :lo
      AND distance <= :hi
      AND id != :exclude_id
    ORDER BY distance
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
    judge=None,
    consider_distance: float = CONSIDER_DISTANCE,
    merge_distance: float = MERGE_DISTANCE,
) -> PersistResult:
    """Insert, merge into an equivalent rule, or supersede one it contradicts.

    `judge` compares two statements and returns "same", "contradicts" or
    "compatible". Without one, nothing is merged: distance alone cannot tell
    agreement from contradiction, and guessing wrong turns a correction into
    reinforcement of the error.
    """
    vector = encode((await provider.embed([candidate.statement]))[0])

    async with engine.connect() as conn:
        nearest = (
            (await conn.execute(NEAREST_RULE, {"repo_id": repo_id, "query": vector}))
            .mappings()
            .first()
        )

    distance = float(nearest["distance"]) if nearest is not None else None

    if nearest is not None and distance <= consider_distance:
        if judge is None:
            log.warning(
                "no judge available; inserting rather than merging at distance %.3f: %s",
                distance,
                candidate.statement[:60],
            )
        else:
            old_date = nearest["last_evidence_at"]
            verdict = await judge(
                old_statement=nearest["statement"],
                new_statement=candidate.statement,
                old_date=old_date.date().isoformat() if old_date else "unknown",
                new_date=candidate.last_evidence_at.date().isoformat()
                if candidate.last_evidence_at
                else "unknown",
            )
            relation = verdict.get("relation")

            if relation == "same":
                rule_id = str(nearest["id"])
                await _record_evidence(engine, repo_id, rule_id, candidate.comment_ids)
                await _recompute(engine, repo_id, rule_id)
                log.info(
                    "merged into %s at distance %.3f: %s",
                    rule_id[:8],
                    distance,
                    candidate.statement[:70],
                )
                return PersistResult(
                    outcome="merged",
                    rule_id=rule_id,
                    distance=distance,
                    statement_vector=vector,
                )

            if relation == "contradicts":
                # Recency decides direction. If the existing rule is at least
                # as current, the new one is the outdated view and goes in as
                # an ordinary rule rather than displacing anything.
                newer = (
                    candidate.last_evidence_at
                    and old_date
                    and candidate.last_evidence_at > old_date
                )
                if newer:
                    log.info(
                        "contradiction at distance %.3f, superseding %s: %s",
                        distance,
                        str(nearest["id"])[:8],
                        verdict.get("reason", "")[:80],
                    )
                    result = await _insert_new(engine, repo_id, candidate, vector, provider)
                    await supersede(
                        engine,
                        repo_id=repo_id,
                        old_rule_id=str(nearest["id"]),
                        replacement_rule_id=result.rule_id,
                        reason=verdict.get("reason", "contradicted by more recent evidence"),
                    )
                    return PersistResult(
                        outcome="superseded",
                        rule_id=result.rule_id,
                        distance=distance,
                        superseded_rule_id=str(nearest["id"]),
                        statement_vector=vector,
                    )

    result = await _insert_new(engine, repo_id, candidate, vector, provider)
    return PersistResult(
        outcome="inserted",
        rule_id=result.rule_id,
        distance=distance,
        statement_vector=vector,
    )


async def _insert_new(
    engine: AsyncEngine,
    repo_id: str,
    candidate: Candidate,
    vector: str,
    provider: EmbeddingProvider,
) -> PersistResult:
    """Write a genuinely new rule, with its evidence and derived confidence."""
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
    return PersistResult(outcome="inserted", rule_id=rule_id, statement_vector=vector)


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
