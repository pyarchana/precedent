"""Gather what memory holds about a question, before any answer is written.

Retrieval is separated from answering deliberately. If the agent turns out to
answer badly, the first question is always whether it was given the right
material, and that is only checkable when the material is a value you can
inspect rather than something assembled inside a prompt.

Two stores are consulted. Rules are the distilled conventions and are what the
answer should be built from. Comments are the episodic evidence and are what
makes the answer citable, because "PR #12345, jbrockmendel" is checkable in a
way that "pandas convention" is not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from precedent.embed.provider import EmbeddingProvider
from precedent.embed.vector import encode
from precedent.memory.search import SearchHit, search_comments

log = logging.getLogger(__name__)

# Beyond this L2 distance a rule is not about the question. Set from the same
# measurements as the clustering thresholds: genuine matches sit well under
# 1.0, and everything past about 1.2 shares only vocabulary.
MAX_RULE_DISTANCE = 1.15

# Retrieved rules below this confidence are carried but flagged. They are not
# hidden: a weakly evidenced convention is still what the project said, and
# concealing it would make the memory look more certain than it is.
LOW_CONFIDENCE = 0.4


@dataclass(slots=True)
class RetrievedRule:
    id: str
    statement: str
    rationale: str | None
    scope: str
    scope_pattern: str | None
    confidence: float
    evidence_count: int
    distance: float
    citations: list[SearchHit] = field(default_factory=list)

    @property
    def is_weak(self) -> bool:
        return self.confidence < LOW_CONFIDENCE


@dataclass(slots=True)
class Recall:
    """Everything memory has to say about one question."""

    question: str
    rules: list[RetrievedRule] = field(default_factory=list)
    comments: list[SearchHit] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """True when memory holds nothing relevant.

        The agent must be able to say so. An agent that falls back on the base
        model's general pandas knowledge proves nothing about memory, which is
        why five of the evaluation questions are deliberately unanswerable.
        """
        return not self.rules and not self.comments


SEARCH_RULES = text("""
    SELECT id, statement, rationale, scope::STRING AS scope, scope_pattern,
           confidence, evidence_count, status, distance
    FROM (
        SELECT id, statement, rationale, scope, scope_pattern, confidence,
               evidence_count, status,
               embedding <-> CAST(:query AS VECTOR(1536)) AS distance
        FROM rules
        WHERE repo_id = :repo_id
        ORDER BY embedding <-> CAST(:query AS VECTOR(1536))
        LIMIT :candidates
    )
    WHERE status = 'active' AND distance <= :max_distance
    ORDER BY distance
    LIMIT :k
""")

# The comments a rule was actually learned from. These are what the answer
# cites, so they are fetched per rule rather than searched for separately:
# a citation must be evidence *for that rule*, not merely a comment that
# resembles the question.
EVIDENCE_FOR_RULE = text("""
    SELECT rc.id, rc.pr_number, rc.kind::STRING AS kind, rc.body, rc.author,
           rc.is_maintainer, rc.file_path, rc.url, rc.created_at
    FROM rule_evidence re
    JOIN review_comments rc
      ON rc.repo_id = re.repo_id AND rc.id = re.comment_id
    WHERE re.repo_id = :repo_id AND re.rule_id = :rule_id
    ORDER BY rc.created_at DESC
    LIMIT :per_rule
""")


async def recall(
    engine: AsyncEngine,
    provider: EmbeddingProvider,
    *,
    repo_id: str,
    question: str,
    rule_k: int = 5,
    comment_k: int = 6,
    per_rule_citations: int = 3,
    max_rule_distance: float = MAX_RULE_DISTANCE,
) -> Recall:
    """Retrieve rules and supporting evidence for a question."""
    vector = encode((await provider.embed([question]))[0])

    async with engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    SEARCH_RULES,
                    {
                        "repo_id": repo_id,
                        "query": vector,
                        # Over-fetch, because the status filter runs outside
                        # the vector-ordered subquery.
                        "candidates": max(rule_k * 4, 20),
                        "max_distance": max_rule_distance,
                        "k": rule_k,
                    },
                )
            )
            .mappings()
            .all()
        )

        rules: list[RetrievedRule] = []
        for row in rows:
            evidence = (
                (
                    await conn.execute(
                        EVIDENCE_FOR_RULE,
                        {
                            "repo_id": repo_id,
                            "rule_id": str(row["id"]),
                            "per_rule": per_rule_citations,
                        },
                    )
                )
                .mappings()
                .all()
            )
            rules.append(
                RetrievedRule(
                    id=str(row["id"]),
                    statement=row["statement"],
                    rationale=row["rationale"],
                    scope=str(row["scope"]),
                    scope_pattern=row["scope_pattern"],
                    confidence=float(row["confidence"]),
                    evidence_count=row["evidence_count"],
                    distance=float(row["distance"]),
                    citations=[
                        SearchHit(
                            id=str(e["id"]),
                            pr_number=e["pr_number"],
                            kind=str(e["kind"]),
                            body=e["body"],
                            author=e["author"],
                            is_maintainer=e["is_maintainer"],
                            file_path=e["file_path"],
                            url=e["url"],
                            created_at=e["created_at"],
                            distance=0.0,
                        )
                        for e in evidence
                    ],
                )
            )

    # Episodic memory as well, for questions the rules do not cover. A
    # contributor may ask something no convention addresses but that a
    # maintainer once answered directly.
    hits = await search_comments(
        engine, provider, repo_id=repo_id, question_vector=vector, k=comment_k
    )

    log.info(
        "recall for %r: %d rules, %d comments",
        question[:60],
        len(rules),
        len(hits),
    )
    return Recall(question=question, rules=rules, comments=list(hits))
