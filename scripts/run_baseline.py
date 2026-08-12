"""Measure the claim: does the memory beat the same model without it.

Everything else in this repository is evidence that the system is careful. None
of it is evidence that the system is *useful*, because the obvious rebuttal has
never been tested: `gpt-4o-mini` has read a great deal of pandas and might
answer these questions perfectly well on its own, in which case the corpus, the
cluster and the retrieval are an expensive way to reach the same place.

So the same 29 questions go to both, and three things are counted.

**Correct.** A judge compares each answer against the expected one from
`eval/questions.yaml`. This is the number people expect to matter, and it is the
least interesting of the three.

**Refused when it should.** Six of the questions are deliberately unanswerable
from pandas review history. A model that answers them has invented something,
however plausible it sounds, and a contributor cannot tell the difference. This
is the number the whole design exists to move.

**Citations that resolve.** Both systems are asked to cite pull requests.
Precedent's are verified against the evidence it retrieved, before the answer is
released; an answer whose citations do not check out is discarded rather than
shown. The baseline's are checked here against the corpus.

Read that third number carefully, because it cannot bear as much weight as it
looks like it can. The corpus holds 21,762 of pandas' roughly 66,000 pull
requests, so a citation that does not resolve is **not** proof of fabrication:
it may name a real pull request that was never ingested. What it does establish
is the asymmetry. Precedent cannot emit a citation that fails to resolve,
because verification happens before release and a failure suppresses the whole
answer. The baseline has no such mechanism and no way to acquire one, since it
is generating plausible integers rather than reading anything.

Costs about four cents for the full set. `--limit` cuts it for a dry run.

    python scripts/run_baseline.py --limit 4
    python scripts/run_baseline.py --json eval/baseline.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml
from sqlalchemy import text

from precedent.agent.answer import answer_question
from precedent.agent.retrieve import recall
from precedent.config import REPO_ROOT, get_settings
from precedent.db.engine import create_engine
from precedent.embed.provider import OpenAIEmbeddings
from precedent.extract.llm import ChatClient

DEFAULT_QUESTIONS = REPO_ROOT / "eval" / "questions.yaml"

# What the baseline is told. Deliberately generous: it is given the repository,
# the role, and an explicit instruction to cite, because the comparison is only
# worth anything if the baseline is asked for its best rather than set up to
# fail. It is not told to refuse when unsure, for the same reason Precedent's
# refusal counts as a result: if refusing has to be prompted for, it is the
# prompt doing the work rather than the memory.
BASELINE_SYSTEM = """\
You are helping a first-time contributor to the pandas project (pandas-dev/pandas).

Answer their question about the project's conventions as accurately as you can.
Where a convention was established in a specific pull request, cite it in the \
form [PR #12345] so the contributor can check it.

Reply with JSON only:

{"answer": "your answer, with [PR #12345] citations where you can give them"}
"""

JUDGE_SYSTEM = """\
You are scoring answers given to a pandas contributor.

You are given the question, the answer known to be correct, and an answer a \
system produced. Decide whether the produced answer would lead the contributor \
to the correct behaviour.

Ignore differences in wording, length and citation. Judge the substance.

Reply with JSON only:

{"verdict": "yes" | "partial" | "no", "reason": "one sentence"}

  - "yes": substantially the correct answer.
  - "partial": right direction, but incomplete or hedged enough to leave the \
contributor unsure.
  - "no": wrong, or a refusal to answer.
"""

# Does a cited pull request exist in the corpus at all, and did anyone discuss
# it. A number alone proves little, since pandas has 66,000 pull requests and
# almost any five-digit guess names a real one. What this catches is a citation
# pointing at a pull request that is not in the review history the answer is
# supposedly drawn from.
PR_EXISTS = text("""
    SELECT count(*) FROM review_comments
    WHERE repo_id = :repo_id AND pr_number = :pr
""")

CITE = re.compile(r"\[PR #(\d+)\]")


@dataclass
class Outcome:
    question_id: str
    answerable: bool
    system: str
    answered: bool
    text: str
    verdict: str = "no"
    cited: list[int] = field(default_factory=list)
    resolving: list[int] = field(default_factory=list)

    @property
    def refused(self) -> bool:
        return not self.answered


@dataclass
class Tally:
    name: str
    correct: int = 0
    partial: int = 0
    answerable_total: int = 0
    refused_when_should: int = 0
    unanswerable_total: int = 0
    cited: int = 0
    resolving: int = 0

    @property
    def correct_rate(self) -> float:
        return self.correct / max(self.answerable_total, 1)

    @property
    def abstention_rate(self) -> float:
        return self.refused_when_should / max(self.unanswerable_total, 1)

    @property
    def citation_rate(self) -> float:
        return self.resolving / max(self.cited, 1)


async def ask_baseline(chat, question: str) -> str:
    result = await chat.complete_json(
        [
            {"role": "system", "content": BASELINE_SYSTEM},
            {"role": "user", "content": question},
        ]
    )
    if not isinstance(result, dict):
        return ""
    return str(result.get("answer") or "")


async def judge(chat, question: str, expected: str, produced: str) -> str:
    result = await chat.complete_json(
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Correct answer: {expected}\n\n"
                    f"Answer produced: {produced or '(no answer)'}"
                ),
            },
        ]
    )
    verdict = (result or {}).get("verdict") if isinstance(result, dict) else None
    return verdict if verdict in ("yes", "partial", "no") else "no"


async def resolve_citations(conn, repo_id: str, text_body: str) -> tuple[list[int], list[int]]:
    cited = sorted({int(n) for n in CITE.findall(text_body or "")})
    resolving = []
    for pr in cited:
        count = (await conn.execute(PR_EXISTS, {"repo_id": repo_id, "pr": pr})).scalar() or 0
        if count:
            resolving.append(pr)
    return cited, resolving


async def run(args) -> int:
    settings = get_settings()
    data = yaml.safe_load(args.questions.read_text(encoding="utf-8"))
    questions = data["questions"][: args.limit] if args.limit else data["questions"]

    engine = create_engine(args.dsn)
    provider = OpenAIEmbeddings(
        settings.openai_api_key, model=settings.embedding_model, dimensions=settings.embedding_dim
    )
    chat = ChatClient(settings.openai_api_key, max_spend=args.max_spend)

    outcomes: list[Outcome] = []
    try:
        owner, _, name = settings.repo_slug.partition("/")
        async with engine.connect() as conn:
            repo_id = str(
                (
                    await conn.execute(
                        text("SELECT id FROM repos WHERE owner = :o AND name = :n"),
                        {"o": owner, "n": name},
                    )
                ).scalar_one()
            )

        print(f"{len(questions)} questions, two systems\n")
        for entry in questions:
            question = entry["question"]
            expected = (entry.get("expected") or "").strip()
            answerable = bool(entry.get("answerable", True))

            # Precedent: retrieval, then an answer built only from what came back.
            memory = await recall(engine, provider, repo_id=repo_id, question=question)
            answer = await answer_question(chat, memory)

            # Baseline: the same model, nothing else.
            bare = await ask_baseline(chat, question)

            async with engine.connect() as conn:
                bare_cited, bare_resolving = await resolve_citations(conn, repo_id, bare)

            for system, answered, body, cited, resolving in (
                ("precedent", answer.answered, answer.text, answer.cited_prs, answer.cited_prs),
                ("baseline", bool(bare.strip()), bare, bare_cited, bare_resolving),
            ):
                verdict = "no"
                if answerable and answered:
                    verdict = await judge(chat, question, expected, body)
                outcomes.append(
                    Outcome(
                        question_id=entry["id"],
                        answerable=answerable,
                        system=system,
                        answered=answered,
                        text=body,
                        verdict=verdict,
                        cited=list(cited),
                        resolving=list(resolving),
                    )
                )

            mark = "answerable" if answerable else "UNANSWERABLE"
            p = outcomes[-2]
            b = outcomes[-1]
            print(
                f"  {entry['id']:<28} {mark:<12} "
                f"precedent={'refused' if p.refused else p.verdict:<8} "
                f"baseline={'refused' if b.refused else b.verdict}"
            )
    finally:
        await provider.aclose()
        await chat.aclose()
        await engine.dispose()

    tallies = {name: Tally(name) for name in ("precedent", "baseline")}
    for outcome in outcomes:
        tally = tallies[outcome.system]
        if outcome.answerable:
            tally.answerable_total += 1
            tally.correct += outcome.verdict == "yes"
            tally.partial += outcome.verdict == "partial"
        else:
            tally.unanswerable_total += 1
            tally.refused_when_should += outcome.refused
        tally.cited += len(outcome.cited)
        tally.resolving += len(outcome.resolving)

    print(f"\n  {'':<10} {'correct':>18} {'refused when it should':>24} {'citations resolve':>20}")
    for tally in tallies.values():
        correct = f"{tally.correct}/{tally.answerable_total} ({tally.correct_rate:.0%})"
        refused = (
            f"{tally.refused_when_should}/{tally.unanswerable_total} ({tally.abstention_rate:.0%})"
        )
        citations = f"{tally.resolving}/{tally.cited} ({tally.citation_rate:.0%})"
        print(f"  {tally.name:<10} {correct:>18} {refused:>24} {citations:>20}")

    print(
        "\nCitations resolve against the ingested corpus, which is 21,762 of pandas'"
        "\nroughly 66,000 pull requests. A citation that does not resolve is therefore"
        "\nnot proven fabricated. What the column shows is that one system verifies"
        "\nbefore it speaks and the other has no way to."
    )
    print(f"\nspent ${chat.spent:.4f}")

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "generated_at": datetime.now(UTC).isoformat(),
                    "repo": settings.repo_slug,
                    "spend_usd": round(chat.spent, 4),
                    "summary": {
                        name: {
                            "correct": t.correct,
                            "partial": t.partial,
                            "answerable_total": t.answerable_total,
                            "refused_when_should": t.refused_when_should,
                            "unanswerable_total": t.unanswerable_total,
                            "citations": t.cited,
                            "citations_resolving": t.resolving,
                        }
                        for name, t in tallies.items()
                    },
                    "outcomes": [
                        {
                            "id": o.question_id,
                            "system": o.system,
                            "answerable": o.answerable,
                            "answered": o.answered,
                            "verdict": o.verdict,
                            "cited": o.cited,
                            "resolving": o.resolving,
                            "text": o.text,
                        }
                        for o in outcomes
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.json}")

    return 0


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=settings.cockroach_dsn)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--limit", type=int, default=None, help="First N questions only.")
    parser.add_argument("--max-spend", type=float, default=0.25)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    if not args.dsn:
        parser.error("no DSN: pass --dsn or set COCKROACH_DSN in .env")

    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
