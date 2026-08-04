"""Load extracted rules from a run's JSONL into semantic memory.

Kept separate from extraction on purpose. Extraction is expensive and its
output is worth reading before it becomes rows; loading is cheap and can be
re-run against the same file as often as the merge logic needs debugging.

Re-running is safe. Evidence is keyed on (rule, comment), and counts are
recomputed from the evidence table rather than incremented, so loading the
same file twice leaves the same rules with the same confidence.

    python scripts/load_rules.py data/extracted/rules-20260803-014616.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from sqlalchemy import text

from precedent.config import get_settings
from precedent.db.engine import create_engine
from precedent.embed.provider import OpenAIEmbeddings
from precedent.extract import contradiction
from precedent.extract.llm import BudgetExhausted, ChatClient
from precedent.extract.persist import Candidate, persist_rule

log = logging.getLogger("precedent.load_rules")


def parse_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


async def main_async(args: argparse.Namespace) -> int:
    settings = get_settings()
    records = [
        json.loads(line)
        for line in Path(args.path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rules = [r for r in records if r["result"].get("is_convention")]
    log.info("%d rules in %s", len(rules), args.path)

    # Strongest first, so that when two near-identical rules arrive together
    # the better-evidenced one becomes the canonical statement and the weaker
    # merges into it, rather than the other way round.
    rules.sort(key=lambda r: r.get("confidence", 0), reverse=True)

    engine = create_engine(args.dsn)
    provider = OpenAIEmbeddings(
        settings.openai_api_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dim,
    )

    # Contradiction checking is the only paid part of loading, and it fires
    # rarely, so it gets its own small ceiling and can be turned off entirely.
    client = None
    judge = None
    if not args.no_supersede:
        client = ChatClient(settings.openai_api_key, model=args.model, max_spend=args.max_spend)

        async def judge(**kwargs):
            try:
                return await client.complete_json(contradiction.build_messages(**kwargs))
            except BudgetExhausted as exc:
                log.warning("contradiction checks stopped: %s", exc)
                return {"relation": "compatible", "reason": "budget exhausted"}

    owner, _, name = args.repo.partition("/")
    outcomes: Counter[str] = Counter()

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

        for record in rules:
            result = record["result"]
            scope = result.get("scope") or "repo"
            pattern = result.get("scope_pattern")
            # The schema requires a pattern for these scopes, and the model
            # sometimes omits it. Demote rather than reject: the rule is still
            # true, we just do not know how narrowly it applies.
            if scope in ("directory", "file") and not pattern:
                scope = "repo"

            candidate = Candidate(
                statement=result["statement"],
                rationale=result.get("rationale"),
                scope=scope,
                scope_pattern=pattern if scope in ("directory", "file") else None,
                comment_ids=record.get("comment_ids", []),
                distinct_prs=record.get("distinct_prs", 0),
                distinct_authors=record.get("distinct_authors", 0),
                first_evidence_at=parse_dt(record.get("first_evidence_at")),
                last_evidence_at=parse_dt(record.get("last_evidence_at")),
            )

            result = await persist_rule(
                engine, provider, repo_id=repo_id, candidate=candidate, judge=judge
            )
            outcomes[result.outcome] += 1

    finally:
        await provider.aclose()
        if client is not None:
            await client.aclose()
        await engine.dispose()

    log.info("outcomes: %s", dict(outcomes))
    if client is not None:
        log.info("contradiction checks: %d calls, $%.4f", client.usage.calls, client.spent)
    return 0


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Load extracted rules into memory.")
    parser.add_argument("path", help="A rules-*.jsonl file from an extraction run.")
    parser.add_argument("--dsn", default=settings.cockroach_dsn or None)
    parser.add_argument("--repo", default=settings.repo_slug)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max-spend", type=float, default=0.05)
    parser.add_argument(
        "--no-supersede",
        action="store_true",
        help="Skip contradiction checking, the only step here that costs money.",
    )
    args = parser.parse_args()

    if not args.dsn:
        parser.error("no DSN: pass --dsn or set COCKROACH_DSN in .env")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s | %(message)s",
        stream=sys.stdout,
    )
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
