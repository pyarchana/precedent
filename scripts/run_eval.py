"""Run the evaluation questions through semantic search and report what came back.

This scores retrieval, not answers. The question it settles is whether the
right raw material reaches the agent at all, because no amount of prompting
fixes an answer built from the wrong comments.

Two numbers, because the obvious one is misleading on its own.

**recall@k** asks whether a pull request recorded in questions.yaml appears in
the top k. It is cheap and objective, and it badly understates performance. In
a corpus where dozens of threads express the same convention, it measures
whether retrieval found *my* citation rather than whether it found an answer.
Asked where the GitHub issue number goes in a test, retrieval returned "Can
you add the GitHub issue number as a comment? See two tests below for an
example" and scored zero, because that comment is on a pull request I had not
thought to list.

**answer rate** asks a model whether the retrieved comments actually support
the expected answer. It costs about a tenth of a cent for the whole set and is
the number worth reading. recall@k is kept as a floor and a sanity check: if
it collapses to zero, something is broken regardless of what a judge says.

    python scripts/run_eval.py --dsn "..." --k 10
    python scripts/run_eval.py --dsn "..." --tag testing --show 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import yaml
from sqlalchemy import text

from precedent.config import REPO_ROOT, get_settings
from precedent.db.engine import create_engine
from precedent.embed.provider import OpenAIEmbeddings
from precedent.extract.llm import ChatClient
from precedent.memory.search import search_comments

DEFAULT_QUESTIONS = REPO_ROOT / "eval" / "questions.yaml"

JUDGE_SYSTEM = """\
You are scoring a retrieval system for a code review memory.

Given a contributor's question, the answer known to be correct, and the \
comments retrieval returned, decide whether those comments support the \
correct answer.

Reply with JSON only:

{"verdict": "yes" | "partial" | "no", "reason": "one sentence"}

  - "yes": a contributor reading these comments would arrive at the correct \
answer.
  - "partial": the comments are relevant and point in the right direction but \
leave the key detail unstated.
  - "no": the comments do not support the answer.

Judge the comments, not the answer. Do not reward a comment for being about \
the same general topic. And when the question is one the corpus should not be \
able to answer, "no" is the correct and desirable verdict.
"""

JUDGE_USER = """\
Question: {question}

Correct answer: {expected}

Retrieved comments:
{retrieved}
"""


def flatten(body: str, width: int) -> str:
    return re.sub(r"\s+", " ", body).strip()[:width]


async def main_async(args: argparse.Namespace) -> int:
    settings = get_settings()
    spec = yaml.safe_load(Path(args.questions).read_text(encoding="utf-8"))
    questions = spec["questions"]

    if args.tag:
        questions = [q for q in questions if args.tag in (q.get("tags") or [])]
    if args.id:
        questions = [q for q in questions if q["id"] == args.id]
    if not questions:
        print("no questions matched the filters")
        return 1

    engine = create_engine(args.dsn)
    provider = OpenAIEmbeddings(
        settings.openai_api_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dim,
    )

    owner, _, name = spec["repo"].partition("/")
    async with engine.connect() as conn:
        repo_id = str(
            (
                await conn.execute(
                    text("SELECT id FROM repos WHERE owner = :o AND name = :n"),
                    {"o": owner, "n": name},
                )
            ).scalar_one()
        )
        embedded = (
            await conn.execute(
                text("SELECT count(*) FROM review_comments WHERE embedding IS NOT NULL")
            )
        ).scalar_one()

    print(f"repo {spec['repo']}, {embedded} comments embedded, k={args.k}\n")

    records = []
    answerable_hits = 0
    answerable_total = 0
    verdicts: Counter[str] = Counter()

    judge = None
    if not args.no_judge:
        judge = ChatClient(
            settings.openai_api_key, model=args.judge_model, max_spend=args.max_spend
        )

    try:
        for q in questions:
            result = await search_comments(
                engine,
                provider,
                repo_id=repo_id,
                query=q["question"],
                k=args.k,
                maintainers_only=args.maintainers_only,
            )
            retrieved_prs = [hit.pr_number for hit in result]
            expected = set(q.get("sources") or [])
            found = expected.intersection(retrieved_prs)

            if q.get("answerable", True):
                answerable_total += 1
                if found:
                    answerable_hits += 1
                status = "HIT " if found else "MISS"
            else:
                # Nothing to hit. What matters is whether the distances are far
                # enough for the agent to recognise it has no relevant memory.
                status = "n/a "

            verdict = None
            if judge is not None:
                retrieved_text = "\n\n".join(
                    f"[PR #{h.pr_number}, {h.author or 'unknown'}]\n{flatten(h.body, 500)}"
                    for h in list(result)[: args.k]
                )
                judgement = await judge.complete_json(
                    [
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {
                            "role": "user",
                            "content": JUDGE_USER.format(
                                question=q["question"],
                                expected=" ".join((q.get("expected") or "").split()),
                                retrieved=retrieved_text or "(nothing retrieved)",
                            ),
                        },
                    ]
                )
                verdict = judgement.get("verdict")
                verdicts[verdict or "unknown"] += 1

            nearest = result[0].distance if len(result) else float("nan")
            mark = f" judge={verdict}" if verdict else ""
            print(f"[{status}] {q['id']}{mark}")
            print(f"         {q['question']}")
            if expected:
                print(
                    f"         expected PRs {sorted(expected)}, "
                    f"found {sorted(found) if found else 'none'}"
                )
            print(f"         nearest distance {nearest:.4f}, PRs {retrieved_prs[: args.k]}")

            for hit in list(result)[: args.show]:
                marker = "M" if hit.is_maintainer else " "
                print(f"           {marker} {hit.distance:.4f} {hit.citation}")
                print(f"             {flatten(hit.body, args.width)}")
            print()

            records.append(
                {
                    "id": q["id"],
                    "question": q["question"],
                    "answerable": q.get("answerable", True),
                    "expected": sorted(expected),
                    "retrieved": retrieved_prs,
                    "found": sorted(found),
                    "nearest_distance": None if len(result) == 0 else result[0].distance,
                    "judge_verdict": verdict,
                    "reviewed": q.get("reviewed", False),
                }
            )
    finally:
        await provider.aclose()
        if judge is not None:
            await judge.aclose()
        await engine.dispose()

    unreviewed = sum(1 for q in questions if not q.get("reviewed", False))
    print("=" * 70)
    if answerable_total:
        print(
            f"recall@{args.k}: {answerable_hits}/{answerable_total} "
            f"({100 * answerable_hits / answerable_total:.0f}%)"
        )
    if verdicts:
        answerable_ids = {q["id"] for q in questions if q.get("answerable", True)}
        answered = sum(
            1
            for r in records
            if r["id"] in answerable_ids and r["judge_verdict"] in ("yes", "partial")
        )
        full = sum(1 for r in records if r["id"] in answerable_ids and r["judge_verdict"] == "yes")
        total = len(answerable_ids)
        print(
            f"answer rate: {full}/{total} fully ({100 * full / total:.0f}%), "
            f"{answered}/{total} at least partially ({100 * answered / total:.0f}%)"
        )

        # For the out-of-scope questions, "no" is the desirable answer: the
        # corpus should not be able to answer them.
        oos = [
            r for r in records if r["id"] not in answerable_ids and r["judge_verdict"] is not None
        ]
        if oos:
            correct = sum(1 for r in oos if r["judge_verdict"] == "no")
            print(f"out-of-scope correctly unanswered: {correct}/{len(oos)}")
        print(f"judge cost: ${judge.spent:.4f} over {judge.usage.calls} calls")

    print(f"{unreviewed} of {len(questions)} expected answers are still unreviewed by a human")

    if args.json:
        payload = {
            "run_at": datetime.now(UTC).isoformat(),
            "k": args.k,
            "embedded_comments": embedded,
            "recall_at_k": (answerable_hits / answerable_total) if answerable_total else None,
            "results": records,
        }
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")

    return 0


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Score retrieval against the eval set.")
    parser.add_argument("--dsn", default=settings.cockroach_dsn or None)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--show", type=int, default=2, help="Retrieved comments to print.")
    parser.add_argument("--width", type=int, default=220)
    parser.add_argument("--tag", default=None, help="Only questions carrying this tag.")
    parser.add_argument("--id", default=None, help="Only this question id.")
    parser.add_argument("--maintainers-only", action="store_true")
    parser.add_argument("--json", default=None, help="Write machine-readable results here.")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--max-spend", type=float, default=0.05)
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Report recall@k only. Free, and badly understates performance.",
    )
    args = parser.parse_args()

    if not args.dsn:
        parser.error("no DSN: pass --dsn or set COCKROACH_DSN in .env")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
