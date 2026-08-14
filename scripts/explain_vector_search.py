"""Show that the agent's retrieval actually uses the vector index.

The claim "distributed vector indexing" is easy to make and easy to get wrong,
because the index is defeated by things that look harmless. Passing the query
vector as a subquery rather than a literal defeats it. So does putting any
metadata predicate in the same SELECT as the vector ordering. Both produce a
plan with a `scan` node instead of a `vector search` node, and both still return
correct results, so nothing fails and the index quietly does nothing.

This runs the exact query shape `agent/retrieve.py` uses, with a real embedded
question, and prints the plan. If the output does not contain `vector search`,
the index is not being used and any timing measured against it is meaningless.

    python scripts/explain_vector_search.py
    python scripts/explain_vector_search.py --question "how do I name test variables?"
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from sqlalchemy import text

from precedent.config import get_settings
from precedent.db.engine import create_engine
from precedent.embed.provider import OpenAIEmbeddings
from precedent.embed.vector import encode

# The two-stage shape from agent/retrieve.py. The status filter sits in an outer
# query on purpose: moving it inside, next to the ORDER BY, is what silently
# turns this back into a full scan.
QUERY = text("""
    EXPLAIN
    SELECT id, statement, confidence
    FROM (
        SELECT id, statement, confidence, status,
               embedding <-> CAST(:query AS VECTOR(1536)) AS distance
        FROM rules
        WHERE repo_id = :repo_id
        ORDER BY embedding <-> CAST(:query AS VECTOR(1536))
        LIMIT 20
    )
    WHERE status = 'active'
    ORDER BY distance
    LIMIT 5
""")


async def run(question: str, repo: str) -> int:
    settings = get_settings()
    engine = create_engine(settings.cockroach_dsn)
    provider = OpenAIEmbeddings(
        settings.openai_api_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dim,
    )
    owner, _, name = repo.partition("/")

    try:
        vector = encode((await provider.embed([question]))[0])
        async with engine.connect() as conn:
            repo_id = str(
                (
                    await conn.execute(
                        text("SELECT id FROM repos WHERE owner = :o AND name = :n"),
                        {"o": owner, "n": name},
                    )
                ).scalar_one()
            )
            rows = await conn.execute(QUERY, {"repo_id": repo_id, "query": vector})
            plan = "\n".join(str(row[0]) for row in rows)
    finally:
        await provider.aclose()
        await engine.dispose()

    print(f"question: {question}\n")
    print(plan)

    used = "vector search" in plan
    print()
    if used:
        print("USING THE VECTOR INDEX. rules@idx_rules_embedding served this query.")
        return 0
    print("NOT using the vector index. This planned as a scan; any timing here is misleading.")
    return 1


def main() -> int:
    # EXPLAIN draws its tree with box characters, which the Windows console
    # encodes as cp1252 by default and then dies on.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default="Do I need a whatsnew note for my bug fix?")
    parser.add_argument("--repo", default=settings.repo_slug)
    args = parser.parse_args()

    if not settings.cockroach_dsn:
        parser.error("no DSN: set COCKROACH_DSN in .env")

    return asyncio.run(run(args.question, args.repo))


if __name__ == "__main__":
    sys.exit(main())
