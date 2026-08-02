"""Group related review comments into candidate clusters for rule extraction.

A rule is a convention the project holds, not a remark someone made once. So a
cluster is only worth sending to an LLM when it shows the same guidance given
on *separate occasions*. Reading real clusters made three things obvious, and
each becomes a filter that runs before any paid call:

  * **Distance matters.** Genuine conventions clustered at 0.83 to 0.95.
    Beyond about 0.95 the comments merely shared vocabulary: same words about
    attributes or offsets, no shared rule.

  * **Distinct pull requests, not distinct comments.** One cluster looked like
    four pieces of evidence until you noticed three came from a single PR.
    That is one conversation restated, not a convention observed repeatedly.

  * **Distinct authors strengthen it.** One maintainer with an opinion is
    weaker evidence than three independently saying the same thing, and the
    confidence score should reflect that.

Filtering here is free. Every cluster rejected at this stage is an LLM call
not made, which matters when the whole project runs on a fixed budget.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

log = logging.getLogger(__name__)

# Beyond this, neighbours share vocabulary rather than a convention.
MAX_DISTANCE = 0.95

# Below this, the comments are the *same text* repeated: a canned maintainer
# reply, or the pull request template checklist pasted into eight PRs. Those
# clusters look like overwhelming evidence and are nothing of the sort. In a
# 300 cluster run they produced seven "rules", five of which were the same
# template restated.
MIN_DISTANCE = 0.15

# Below this many separate pull requests, it is one conversation.
MIN_DISTINCT_PRS = 3

# One person repeating themselves is an opinion. A convention needs more than
# one voice, and this also catches boilerplate posted by a single maintainer.
MIN_DISTINCT_AUTHORS = 2

SELECT_SEEDS = text("""
    SELECT id
    FROM review_comments
    WHERE repo_id = :repo_id
      AND is_maintainer
      AND length(body) BETWEEN :min_length AND :max_length
    ORDER BY id
    LIMIT :limit
    OFFSET :offset
""")

# The seed's own stored vector is the query, so building a cluster needs no
# embedding call. It must be read first and passed back as a bound parameter:
# writing the lookup inline as a correlated subquery reads naturally and stops
# the optimizer using the vector index, turning each cluster into a scan over
# every vector in the repository.
SELECT_SEED_VECTOR = text("""
    SELECT embedding::STRING
    FROM review_comments
    WHERE repo_id = :repo_id AND id = :seed_id
""")

SELECT_NEIGHBOURS = text("""
    SELECT id, pr_number, author, file_path, created_at, body, distance
    FROM (
        SELECT id, pr_number, author, file_path, created_at, body,
               embedding <-> CAST(:query AS VECTOR(1536)) AS distance
        FROM review_comments
        WHERE repo_id = :repo_id
        ORDER BY embedding <-> CAST(:query AS VECTOR(1536))
        LIMIT :k
    )
    WHERE distance <= :max_distance
    ORDER BY distance
""")


@dataclass(slots=True)
class ClusterComment:
    id: str
    pr_number: int
    author: str | None
    file_path: str | None
    created_at: object
    body: str
    distance: float


@dataclass(slots=True)
class Cluster:
    seed_id: str
    comments: list[ClusterComment] = field(default_factory=list)

    @property
    def distinct_prs(self) -> int:
        return len({c.pr_number for c in self.comments})

    @property
    def distinct_authors(self) -> int:
        return len({c.author for c in self.comments if c.author})

    @property
    def comment_ids(self) -> set[str]:
        return {c.id for c in self.comments}

    @property
    def mean_distance(self) -> float:
        """Mean distance of the neighbours, excluding the seed.

        The seed is returned at distance zero, so including it biased the mean
        down by roughly 1/k and made every cluster look tighter than it was.
        """
        others = [c.distance for c in self.comments if c.id != self.seed_id]
        if not others:
            return 0.0
        return sum(others) / len(others)

    def rejection_reason(
        self,
        *,
        min_distinct_prs: int = MIN_DISTINCT_PRS,
        min_distinct_authors: int = MIN_DISTINCT_AUTHORS,
        min_distance: float = MIN_DISTANCE,
    ) -> str | None:
        """Why this cluster is not worth paying to extract, or None if it is."""
        if self.distinct_prs < min_distinct_prs:
            return "too few distinct pull requests"
        if self.distinct_authors < min_distinct_authors:
            return "a single author"
        if self.mean_distance < min_distance:
            return "duplicated text, not a convention"
        return None

    def is_worth_extracting(self, **kwargs) -> bool:
        return self.rejection_reason(**kwargs) is None

    def render(self, *, max_chars: int = 900) -> str:
        """The cluster as the model will see it."""
        lines = []
        for c in self.comments:
            where = f" on {c.file_path}" if c.file_path else ""
            body = " ".join(c.body.split())[:max_chars]
            lines.append(f"[PR #{c.pr_number}, {c.author or 'unknown'}{where}]\n{body}")
        return "\n\n".join(lines)


async def fetch_seeds(
    engine: AsyncEngine,
    repo_id: str,
    *,
    limit: int,
    offset: int = 0,
    min_length: int = 150,
    max_length: int = 1500,
) -> list[str]:
    """Candidate cluster centres: maintainer comments long enough to hold a rule."""
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                SELECT_SEEDS,
                {
                    "repo_id": repo_id,
                    "limit": limit,
                    "offset": offset,
                    "min_length": min_length,
                    "max_length": max_length,
                },
            )
        ).all()
    return [str(r[0]) for r in rows]


async def build_cluster(
    engine: AsyncEngine,
    repo_id: str,
    seed_id: str,
    *,
    k: int = 8,
    max_distance: float = MAX_DISTANCE,
) -> Cluster:
    async with engine.connect() as conn:
        vector = (
            await conn.execute(SELECT_SEED_VECTOR, {"repo_id": repo_id, "seed_id": seed_id})
        ).scalar_one_or_none()
        if vector is None:
            return Cluster(seed_id=seed_id)

        rows = (
            (
                await conn.execute(
                    SELECT_NEIGHBOURS,
                    {
                        "repo_id": repo_id,
                        "query": str(vector),
                        "k": k,
                        "max_distance": max_distance,
                    },
                )
            )
            .mappings()
            .all()
        )

    return Cluster(
        seed_id=seed_id,
        comments=[
            ClusterComment(
                id=str(r["id"]),
                pr_number=r["pr_number"],
                author=r["author"],
                file_path=r["file_path"],
                created_at=r["created_at"],
                body=r["body"],
                distance=float(r["distance"]),
            )
            for r in rows
        ],
    )
