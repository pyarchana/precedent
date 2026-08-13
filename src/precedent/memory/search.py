"""Semantic search over episodic memory.

Returns comments with enough metadata to cite them, because an answer the agent
cannot attribute to a real pull request is indistinguishable from the base model
recalling pandas trivia.

## The index this is shaped for is not currently built

`idx_rc_embedding` is defined in migration 0002 and is **not on the cluster**.
It was dropped for the embedding backfill, which is a real and documented
speedup (9 rows/sec with the index against 57.8 without, so nine hours against
eighty minutes), and the rebuild never completed on a free-tier serverless
cluster. `EXPLAIN` on this query today reports a full scan of 86,000 rows.

That is survivable rather than fine, and it is stated here rather than in a
comment nobody reads. This path is secondary: `agent/retrieve.py` searches
`rules`, whose vector index **is** built and used, and only falls through to
comments when the rules are not confident enough to answer alone. Check the
state before trusting any timing measured against this module:

    SELECT index_name FROM [SHOW INDEXES FROM review_comments]

## Why the query is shaped the way it is anyway

Every claim below was checked with EXPLAIN against a real cluster while the
index existed, and the shapes remain correct for when it is rebuilt. Getting
them wrong is the difference between an approximate lookup and a distance
computation over every vector in the repository.

  * `ORDER BY <-> ... LIMIT` is the only shape the vector index recognises.
    Ordering by a computed distance alias does not qualify.
  * `repo_id = :repo_id` is required, not optional. The index is prefixed on
    repo_id, and without an equality on it the optimizer falls back to a scan.
  * There must be no `embedding IS NOT NULL` predicate. It looks harmless and
    forces a full scan, since the optimizer cannot push it into the vector
    index. It is also pointless: rows with no vector are not in the index.
  * **Any** metadata predicate in the same SELECT defeats the index, including
    `is_maintainer`, a `file_path` prefix, or a `kind` filter. So filtering
    happens in an outer query over an over-fetched candidate set, which keeps
    the inner lookup on the index.

The last point costs recall rather than correctness: if the filter is very
selective, the top `candidates` neighbours may not contain `k` matching rows.
`exhausted_candidates` on the result says when that happened.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from precedent.embed.provider import EmbeddingProvider
from precedent.embed.vector import encode

log = logging.getLogger(__name__)

# Neighbours pulled from the index per requested result when filtering. High
# enough that ordinary filters still fill k, low enough to stay cheap.
DEFAULT_OVERFETCH = 20
MIN_CANDIDATES = 100


@dataclass(slots=True)
class SearchHit:
    id: str
    pr_number: int
    kind: str
    body: str
    author: str | None
    is_maintainer: bool
    file_path: str | None
    url: str | None
    created_at: datetime
    distance: float

    @property
    def citation(self) -> str:
        where = f" ({self.file_path})" if self.file_path else ""
        return f"PR #{self.pr_number}{where} by {self.author or 'unknown'}"


@dataclass(slots=True)
class SearchResult:
    hits: list[SearchHit] = field(default_factory=list)
    candidates_considered: int = 0
    exhausted_candidates: bool = False
    """True when filtering consumed the whole candidate pool, so there may be
    further matches beyond the neighbours that were examined."""

    def __iter__(self):
        return iter(self.hits)

    def __len__(self) -> int:
        return len(self.hits)

    def __getitem__(self, index: int) -> SearchHit:
        return self.hits[index]


_SEARCH = """
    SELECT id, pr_number, kind, body, author, is_maintainer, file_path, url,
           created_at, distance
    FROM (
        SELECT id, pr_number, kind, body, author, is_maintainer, file_path,
               url, created_at,
               embedding <-> CAST(:query AS VECTOR(1536)) AS distance
        FROM review_comments
        WHERE repo_id = :repo_id
        ORDER BY embedding <-> CAST(:query AS VECTOR(1536))
        LIMIT :candidates
    )
    {filters}
    ORDER BY distance
    LIMIT :k
"""


async def search_comments(
    engine: AsyncEngine,
    provider: EmbeddingProvider,
    *,
    repo_id: str,
    query: str | None = None,
    query_vector: str | None = None,
    k: int = 10,
    maintainers_only: bool = False,
    path_prefix: str | None = None,
    kinds: tuple[str, ...] | None = None,
    overfetch: int = DEFAULT_OVERFETCH,
) -> SearchResult:
    """Find the comments most similar to a natural-language question.

    Accepts an already-encoded `query_vector` so a caller that has embedded
    the question for another purpose does not pay to embed it twice.
    """
    if query_vector is None:
        if query is None:
            raise ValueError("pass either query or query_vector")
        query_vector = encode((await provider.embed([query]))[0])

    conditions: list[str] = []
    params: dict[str, object] = {
        "repo_id": repo_id,
        "query": query_vector,
        "k": k,
    }

    if maintainers_only:
        conditions.append("is_maintainer")
    if path_prefix:
        conditions.append("file_path LIKE :path_prefix")
        params["path_prefix"] = f"{path_prefix}%"
    if kinds:
        # kind is an enum column, so string parameters need an explicit cast.
        placeholders = ", ".join(f"CAST(:kind_{i} AS comment_kind)" for i in range(len(kinds)))
        conditions.append(f"kind IN ({placeholders})")
        for i, kind in enumerate(kinds):
            params[f"kind_{i}"] = kind

    # With no filters the outer query is a passthrough, so fetch exactly k.
    candidates = k if not conditions else max(k * overfetch, MIN_CANDIDATES)
    params["candidates"] = candidates

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    statement = text(_SEARCH.format(filters=where))

    async with engine.connect() as conn:
        rows = (await conn.execute(statement, params)).mappings().all()

    exhausted = bool(conditions) and len(rows) < k
    if exhausted:
        log.warning(
            "filters matched only %d of %d requested results within %d neighbours; "
            "raise overfetch for better recall",
            len(rows),
            k,
            candidates,
        )

    return SearchResult(
        hits=[
            SearchHit(
                id=str(row["id"]),
                pr_number=row["pr_number"],
                kind=str(row["kind"]),
                body=row["body"],
                author=row["author"],
                is_maintainer=row["is_maintainer"],
                file_path=row["file_path"],
                url=row["url"],
                created_at=row["created_at"],
                distance=float(row["distance"]),
            )
            for row in rows
        ],
        candidates_considered=candidates,
        exhausted_candidates=exhausted,
    )
