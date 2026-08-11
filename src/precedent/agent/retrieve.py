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
from collections import defaultdict
from dataclasses import dataclass, field

from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncEngine

from precedent.embed.provider import EmbeddingProvider
from precedent.embed.vector import encode
from precedent.extract.confidence import STATED_ORIGINS
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
    origin: str = "extracted"
    citations: list[SearchHit] = field(default_factory=list)

    @property
    def is_correction(self) -> bool:
        """Retired a wrong answer's rule."""
        return self.origin == "correction"

    @property
    def is_taught(self) -> bool:
        """Stated directly by a maintainer on a pull request."""
        return self.origin == "taught"

    @property
    def is_stated(self) -> bool:
        """A maintainer said this, rather than it being inferred from a pattern."""
        return self.origin in STATED_ORIGINS

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
           confidence, evidence_count, status, origin, distance
    FROM (
        SELECT id, statement, rationale, scope, scope_pattern, confidence,
               evidence_count, status, origin,
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
# cites, so they are fetched by rule rather than searched for separately: a
# citation must be evidence *for that rule*, not merely a comment that
# resembles the question.
#
# Fetched for every rule in one statement. One query per rule meant five
# sequential round trips to the cluster, which on a cluster in another city is
# most of the time the agent spends answering. The window function keeps the
# per-rule limit that the per-rule query used to give.
EVIDENCE_FOR_RULES = text("""
        SELECT rule_id, id, pr_number, kind, body, author, is_maintainer,
               file_path, url, created_at
        FROM (
            SELECT re.rule_id, rc.id, rc.pr_number, rc.kind::STRING AS kind,
                   rc.body, rc.author, rc.is_maintainer, rc.file_path, rc.url,
                   rc.created_at,
                   row_number() OVER (
                       PARTITION BY re.rule_id ORDER BY rc.created_at DESC
                   ) AS rn
            FROM rule_evidence re
            JOIN review_comments rc
              ON rc.repo_id = re.repo_id AND rc.id = re.comment_id
            WHERE re.repo_id = :repo_id AND re.rule_id IN :rule_ids
        )
        WHERE rn <= :per_rule
        ORDER BY rule_id, created_at DESC
    """).bindparams(bindparam("rule_ids", expanding=True))


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
    confident_enough: float = 0.75,
    confident_rules_needed: int = 2,
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

        by_rule: dict[str, list[SearchHit]] = defaultdict(list)
        if rows:
            evidence = (
                (
                    await conn.execute(
                        EVIDENCE_FOR_RULES,
                        {
                            "repo_id": repo_id,
                            "rule_ids": [str(r["id"]) for r in rows],
                            "per_rule": per_rule_citations,
                        },
                    )
                )
                .mappings()
                .all()
            )
            for e in evidence:
                by_rule[str(e["rule_id"])].append(
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
                )

        rules = [
            RetrievedRule(
                id=str(row["id"]),
                statement=row["statement"],
                rationale=row["rationale"],
                scope=str(row["scope"]),
                scope_pattern=row["scope_pattern"],
                confidence=float(row["confidence"]),
                evidence_count=row["evidence_count"],
                distance=float(row["distance"]),
                origin=str(row["origin"]),
                citations=by_rule.get(str(row["id"]), []),
            )
            for row in rows
        ]

    # Episodic memory as well, for questions the rules do not cover: a
    # contributor may ask something no convention addresses but that a
    # maintainer once answered directly.
    #
    # Skipped when the rules already answer confidently. Citations come from
    # `rule_evidence` keyed by rule id, not from this search, so its only job
    # is covering gaps. Searching 60,000 comments to add colour to an answer
    # that is already well evidenced costs seconds and buys nothing, and it
    # costs a great deal more whenever the vector index is unavailable.
    confident = [r for r in rules if r.confidence >= confident_enough]
    if len(confident) >= confident_rules_needed:
        log.info(
            "recall for %r: %d rules (%d confident), skipping comment search",
            question[:60],
            len(rules),
            len(confident),
        )
        return Recall(question=question, rules=rules, comments=[])

    hits = await search_comments(
        engine, provider, repo_id=repo_id, query_vector=vector, k=comment_k
    )

    log.info(
        "recall for %r: %d rules, %d comments",
        question[:60],
        len(rules),
        len(hits),
    )
    return Recall(question=question, rules=rules, comments=list(hits))
