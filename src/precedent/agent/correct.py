"""Apply a maintainer's correction so later answers reflect it.

This is the claim the whole project rests on. Retrieval with citations is a
search engine with good manners; what makes this a memory is that a maintainer
can say "no, that is wrong" once, and the next contributor to ask gets the
corrected answer, with the correction cited as the reason.

## The shape of the thing

A correction arrives attached to a specific answer, so it has something to act
on: the rules that answer was built from, recorded on the turn at the time.
From there:

  1. The correction text is stored as a `review_comments` row of kind
     `maintainer_correction`. Not in a side table. It is embedded and cited on
     exactly the same path as anything a maintainer said on a real pull
     request, which means every downstream consumer already handles it.
  2. A model turns it into a rule statement, because a correction is prose
     ("no, that only applies to the tests directory") and a rule has to stand
     on its own months later.
  3. The statement is compared against the rules the answer actually used. The
     one it contradicts is superseded and points at the replacement.
  4. The correction then sweeps its neighbourhood, retiring any other active
     rule it also contradicts. See below: without this step a correction can
     be true and still lose.

## Why it compares against the cited rules and not the nearest ones

`persist_rule` already supersedes on nearest-neighbour distance, and reusing
that alone would have been less code. It answers the wrong question. A
correction is aimed at a particular claim the agent made, and the rule that
produced that claim is not reliably the nearest rule to the correction's
wording: a maintainer writing "that is only true for new code" produces a
statement whose nearest neighbour may be some other rule about new code
entirely. Superseding that one would retire a correct rule and leave the wrong
one standing, which is worse than doing nothing.

So the cited rules are checked first, in the order the answer used them. Only
when none of them is the target does this fall back to the ordinary
nearest-neighbour path, which is the right behaviour for a correction that
turns out to be about something the answer did not actually cite.

## Why the correction is not simply believed

The judge decides whether the correction contradicts a cited rule, agrees with
it, or is about something else. All three happen:

  * **contradicts** is the interesting case, and supersedes.
  * **same** means the rule was right and the answer misused it. The correction
    becomes further evidence for that rule instead of a near-duplicate of it.
    Creating a second rule here would be the classic memory failure of learning
    the same thing twice and trusting it more each time.
  * **compatible** means the correction adds something rather than replacing
    anything, so it goes in as a new rule and nothing is retired.

## Why a correction can be refused

A correction has to say what is true, not only that the answer is wrong. Given
"no, that's wrong" and nothing else, the model drafting the rule has only the
question and the answer in front of it, so it restates the answer. That
restatement is then judged "same" as the rule it came from, and merged into it
as further evidence.

The result is the exact failure this path exists to prevent: a maintainer's
objection raising the confidence of the thing they objected to. It is not
hypothetical. The first vague correction this system received, "no actually its
something else", was absorbed as a second supporting comment for the rule being
disputed.

So the drafting step may return `usable: false`, and then nothing is written at
all.

## Why one supersession is not enough

The first correction this system took retired exactly the rule it was aimed at,
and left memory holding the correction alongside two near duplicates of the
thing that had just been corrected, both at higher confidence than the
correction itself. Deduplication had judged those "compatible" rather than
"same" when they were written, so they had survived as separate rules.

Nothing was technically wrong. The correction was recorded, the cited rule was
retired, and the next answer happened to come out right. But memory held a
maintainer's correction and the claim they had corrected at the same time, and
which one an answer used was down to retrieval order. That is not a memory that
learned anything.

So a correction sweeps: after superseding its target it compares itself against
the other active rules nearby and retires the ones it also contradicts.

The sweep is bounded tightly, and the first run showed why. Reaching out to the
usual `CONSIDER_DISTANCE` it retired the two intended duplicates and also
"always include the GitHub issue number in the pull request description", which
does not conflict with a rule about test comments at all. That pair is the
worked example of "compatible" in the judge's own prompt and the judge got it
backwards anyway. So the sweep stops at `MERGE_DISTANCE`, where the measured
duplicates were (0.445, 0.650, 0.708) and the false positive was not (0.763).
A sweep retires several rules off single verdicts with nobody reviewing the
result, so it needs a gate that does not depend on the judge being right.

Usage:

    python -m precedent.agent.correct <session-id> <turn> "no, ..." --as jbrockmendel

The session id and turn number are printed under every answer `ask` gives.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncEngine

from precedent.agent.session import Turn, load_turn
from precedent.config import get_settings
from precedent.db.engine import create_engine
from precedent.db.retry import with_retry
from precedent.embed.provider import EmbeddingProvider, OpenAIEmbeddings
from precedent.embed.vector import encode
from precedent.extract import contradiction
from precedent.extract.llm import ChatClient
from precedent.extract.persist import (
    MERGE_DISTANCE,
    NEIGHBOURS_IN_BAND,
    Candidate,
    insert_rule,
    merge_evidence,
    persist_rule,
    supersede,
)
from precedent.extract.schemas import DraftedRule, Verdict, validated

log = logging.getLogger(__name__)

# A correction did not happen on a pull request, but `review_comments.pr_number`
# is NOT NULL because every other row did. Zero is the sentinel, and rendering
# knows never to print it: an answer citing "[PR #0]" would be worse than one
# citing nothing.
NO_PR = 0


class UnusableCorrection(ValueError):
    """The correction says the answer is wrong without saying what is true.

    Carries what the maintainer would need to add, so the caller can say so
    rather than failing opaquely.
    """


SYSTEM = """\
A maintainer has corrected an answer this system gave a contributor. Turn the \
correction into a single convention the project can be held to.

You are given the contributor's question, the answer that was given, and the \
maintainer's correction. The correction is authoritative. The answer is not.

Rules:

  - State the convention as an instruction, in one sentence, the way it would \
be written in a contributing guide. Not "the earlier answer was wrong about \
X", but what a contributor should actually do.
  - Write it so it stands on its own. Someone reading it in six months will \
not have the question or the answer in front of them.
  - Do not add anything the maintainer did not say. If the correction only \
narrows part of the earlier answer, then the narrower thing is the whole \
convention. Inventing the rest is how a memory becomes untrustworthy.
  - Choose the scope that fits: repo, directory, file, api, testing, docs, \
style, or process. Set scope_pattern only for directory or file scope, as a \
SQL LIKE pattern such as 'pandas/tests/%'. Otherwise leave it null.

  - **Refuse if the correction does not say what the project actually does.** \
"No, that's wrong", "this is not right", "something else" and the like state \
that the answer is wrong without stating what is true. There is nothing to \
record. Set usable to false and say what you would need to be told. Do not \
fall back on the earlier answer, and do not invent the rest: a guess here \
enters memory as though a maintainer had said it.

Reply with JSON only:

{
  "usable": true or false,
  "needed": "what the maintainer would have to say, when usable is false",
  "statement": "the convention, one sentence, imperative",
  "rationale": "why, in one sentence, only if the maintainer gave a reason",
  "scope": "repo" | "directory" | "file" | "api" | "testing" | "docs" | "style" | "process",
  "scope_pattern": null
}
"""

USER = """\
The contributor asked:
{question}

The system answered:
{answer}

{maintainer} corrected it:
{correction}
"""

VALID_SCOPES = {"repo", "directory", "file", "api", "testing", "docs", "style", "process"}

INSERT_CORRECTION_COMMENT = text("""
    INSERT INTO review_comments (
        repo_id, github_node_id, kind, pr_number, author, author_association,
        is_maintainer, body, created_at, embedding, embedding_model, embedded_at
    ) VALUES (
        :repo_id, :node_id, CAST('maintainer_correction' AS comment_kind), :pr_number,
        :author, 'MAINTAINER_CORRECTION', true, :body, now(),
        CAST(:embedding AS VECTOR(1536)), :embedding_model, now()
    )
    RETURNING id
""")

LOAD_RULES = text("""
    SELECT id, statement, last_evidence_at, status, confidence
    FROM rules
    WHERE repo_id = :repo_id AND id IN :rule_ids
""").bindparams(bindparam("rule_ids", expanding=True))

RECORD_CORRECTION = text("""
    INSERT INTO corrections (
        repo_id, session_id, turn_number, comment_id, corrected_rule_id,
        replacement_rule_id, maintainer_login, applied_at
    ) VALUES (
        :repo_id, :session_id, :turn_number, :comment_id, :corrected_rule_id,
        :replacement_rule_id, :maintainer_login, now()
    )
    RETURNING id
""")


@dataclass(slots=True)
class CorrectionResult:
    correction_id: str
    comment_id: str
    statement: str
    rule_id: str
    outcome: str  # inserted | merged | superseded
    corrected_rule_id: str | None = None
    corrected_statement: str | None = None
    reason: str = ""
    # Other active rules the correction also contradicted. Usually near
    # duplicates of the corrected one that deduplication never merged.
    also_retired: list[str] = field(default_factory=list)

    @property
    def changed_memory(self) -> bool:
        """False only if the correction added nothing memory did not have."""
        return self.outcome in ("inserted", "superseded")


async def _statement_from_correction(chat, turn: Turn, maintainer: str, correction: str) -> dict:
    result = await chat.complete_json(
        [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": USER.format(
                    question=turn.question,
                    answer=turn.answer or "(no answer was recorded)",
                    maintainer=maintainer,
                    correction=correction,
                ),
            },
        ]
    )

    parsed = validated(DraftedRule, result, context="drafting a correction")
    statement = parsed.statement.strip()
    if not parsed.is_usable:
        # Nothing is written. A correction saying only that the answer is wrong
        # leaves the model with just the question and the earlier answer to work
        # from, so it restates the answer, and that restatement is then judged
        # "same" as the rule it came from and merged into it as evidence.
        #
        # That is not a near miss, it is the failure this whole path exists to
        # prevent: a maintainer's objection strengthening the thing they were
        # objecting to. It happened on the first vague correction this system
        # received, and the rule gained a second supporting comment from
        # somebody disputing it.
        needed = parsed.needed.strip()
        raise UnusableCorrection(
            "A correction has to say what the project actually does, not only that "
            "the answer is wrong. Nothing was recorded."
            + (f" Missing: {needed}." if needed else "")
        )

    # Scope is already constrained to the enum by validation; an unknown value
    # became "repo" there rather than needing checking again here.
    scope = parsed.scope
    pattern = parsed.scope_pattern
    if pattern and scope not in ("directory", "file"):
        # A pattern on a scope that does not use one would be stored and never
        # applied, which reads as a constraint the memory is not enforcing.
        log.info("dropping scope_pattern %r on %s-scoped rule", pattern, scope)
        pattern = None

    return {
        "statement": statement,
        "rationale": parsed.rationale.strip() or None,
        "scope": scope,
        "scope_pattern": pattern,
    }


async def _store_correction_text(
    engine: AsyncEngine,
    provider: EmbeddingProvider,
    *,
    repo_id: str,
    maintainer: str,
    correction: str,
) -> str:
    vector = encode((await provider.embed([correction]))[0])

    async def op() -> str:
        async with engine.begin() as conn:
            return str(
                (
                    await conn.execute(
                        INSERT_CORRECTION_COMMENT,
                        {
                            "repo_id": repo_id,
                            # Unique per correction rather than per turn: a
                            # maintainer may correct the same answer twice, and
                            # the second attempt is not a duplicate of the first.
                            "node_id": f"correction:{uuid.uuid4()}",
                            "pr_number": NO_PR,
                            "author": maintainer,
                            "body": correction,
                            "embedding": vector,
                            "embedding_model": getattr(provider, "model", None),
                        },
                    )
                ).scalar_one()
            )

    return await with_retry(op, description="store correction text")


async def _find_target(engine, chat, *, repo_id: str, turn: Turn, statement: str, today: str):
    """Which cited rule the correction is aimed at, and how it relates.

    Returns (rule_row, relation, reason), or (None, None, "") when the answer
    cited nothing or none of what it cited is relevant.
    """
    if not turn.cited_rule_ids:
        return None, None, ""

    async with engine.connect() as conn:
        rows = (
            (await conn.execute(LOAD_RULES, {"repo_id": repo_id, "rule_ids": turn.cited_rule_ids}))
            .mappings()
            .all()
        )

    # Back into the order the answer used them. The first rule is the one the
    # answer leaned on hardest, so it is the likeliest target and checking it
    # first usually means one model call rather than five.
    by_id = {str(r["id"]): r for r in rows}
    ordered = [by_id[rid] for rid in turn.cited_rule_ids if rid in by_id]

    for rule in ordered:
        if rule["status"] != "active":
            # Already superseded by something else since the answer was given.
            continue

        last = rule["last_evidence_at"]
        verdict = await chat.complete_json(
            contradiction.build_messages(
                old_statement=rule["statement"],
                new_statement=statement,
                old_date=last.date().isoformat() if last else "unknown",
                new_date=today,
            )
        )
        judged = validated(Verdict, verdict, context="comparing a correction to a cited rule")
        relation, reason = judged.relation, judged.reason

        if relation in ("contradicts", "same"):
            return rule, relation, reason

    return None, None, ""


async def sweep_contradicted(
    engine: AsyncEngine,
    chat,
    *,
    repo_id: str,
    statement: str,
    statement_vector: str,
    keep_rule_id: str,
    maintainer_login: str,
    already_retired: set[str] | None = None,
    max_comparisons: int = 8,
    max_distance: float = MERGE_DISTANCE,
) -> list[tuple[str, str, str]]:
    """Retire every other active rule the correction contradicts.

    Superseding only the rule the answer cited is not enough, and the first
    correction this system took proved it. A maintainer corrected "put the issue
    number at the top of the test"; that rule was retired, and two near
    duplicates saying the same superseded thing stayed active at higher
    confidence than the correction, because deduplication had judged them
    "compatible" rather than "same" when they were written.

    The result is a memory that holds a maintainer's correction and the thing
    they corrected at the same time, and answers from whichever the retrieval
    happens to rank first. So a correction sweeps its neighbourhood.

    Bounded two ways, and both bounds were earned.

    `max_comparisons` bounds cost: each comparison is a model call, and a dense
    cluster of similar rules would otherwise run up more than the correction is
    worth.

    `max_distance` bounds damage, and defaults to `MERGE_DISTANCE` rather than
    the wider `CONSIDER_DISTANCE` the ordinary write path uses. The first run of
    this sweep went out to 1.10 and retired three rules, of which two were the
    intended near duplicates and the third was "always include the GitHub issue
    number in the pull request description", which does not contradict a rule
    about where to put a comment in a test at all. Both can be followed. That
    exact pair is the worked example of "compatible" in the judge's own prompt,
    and the judge still called it a contradiction.

    Measured against the correction, the genuine duplicates sat at 0.445, 0.650
    and 0.708, and the false positive at 0.763. So the distance gate is the one
    that has to hold here, because the judge demonstrably does not. The margin
    is thin, which is the reason for a gate at all: a sweep supersedes several
    rules off single verdicts, so one bad verdict destroys correct guidance,
    and unlike the targeted path there is no maintainer looking at the result.
    """
    retired: list[tuple[str, str, str]] = []
    skip = set(already_retired or set())

    async with engine.connect() as conn:
        neighbours = (
            (
                await conn.execute(
                    NEIGHBOURS_IN_BAND,
                    {
                        "repo_id": repo_id,
                        "query": statement_vector,
                        "lo": 0.0,
                        "hi": max_distance,
                        "exclude_id": keep_rule_id,
                    },
                )
            )
            .mappings()
            .all()
        )

    for neighbour in neighbours[:max_comparisons]:
        rule_id = str(neighbour["id"])
        if rule_id in skip:
            continue

        verdict = await chat.complete_json(
            contradiction.build_messages(
                old_statement=neighbour["statement"],
                new_statement=statement,
                old_date=(
                    neighbour["last_evidence_at"].date().isoformat()
                    if neighbour["last_evidence_at"]
                    else "unknown"
                ),
                new_date=datetime.now(UTC).date().isoformat(),
            )
        )
        judged = validated(Verdict, verdict, context="sweeping for contradictions")
        if judged.relation != "contradicts":
            continue

        reason = judged.reason
        await supersede(
            engine,
            repo_id=repo_id,
            old_rule_id=rule_id,
            replacement_rule_id=keep_rule_id,
            reason=f"contradicted by {maintainer_login}'s correction: {reason}"
            if reason
            else f"contradicted by {maintainer_login}'s correction",
        )
        retired.append((rule_id, neighbour["statement"], reason))

    if retired:
        log.info("correction swept %d further contradicted rules", len(retired))
    return retired


async def apply_correction(
    engine: AsyncEngine,
    provider: EmbeddingProvider,
    chat,
    *,
    repo_id: str,
    session_id: str,
    turn_number: int,
    maintainer_login: str,
    correction: str,
) -> CorrectionResult:
    """Record a correction and change what memory will say next time."""
    turn = await load_turn(engine, repo_id=repo_id, session_id=session_id, turn_number=turn_number)
    if turn is None:
        raise LookupError(f"no turn {turn_number} in session {session_id}")

    drafted = await _statement_from_correction(chat, turn, maintainer_login, correction)
    statement = drafted["statement"]
    log.info("correction reads as: %s", statement)

    comment_id = await _store_correction_text(
        engine,
        provider,
        repo_id=repo_id,
        maintainer=maintainer_login,
        correction=correction,
    )

    now = datetime.now(UTC)
    candidate = Candidate(
        statement=statement,
        rationale=drafted["rationale"],
        scope=drafted["scope"],
        scope_pattern=drafted["scope_pattern"],
        comment_ids=[comment_id],
        distinct_prs=1,
        distinct_authors=1,
        # Dated now, which is what makes recency resolve the conflict in the
        # correction's favour without a special case: it is by definition the
        # most recent thing the project has said on the subject.
        first_evidence_at=now,
        last_evidence_at=now,
    )

    target, relation, reason = await _find_target(
        engine,
        chat,
        repo_id=repo_id,
        turn=turn,
        statement=statement,
        today=now.date().isoformat(),
    )

    statement_vector: str | None = None

    if target is not None and relation == "same":
        # The rule was right and the answer misused it. Strengthen the rule.
        rule_id = str(target["id"])
        await merge_evidence(engine, repo_id=repo_id, rule_id=rule_id, comment_ids=[comment_id])
        outcome, corrected_id = "merged", None
        log.info("correction agrees with cited rule %s; recorded as evidence", rule_id[:8])

    elif target is not None:
        # Inserted without comparing against memory, because the target is
        # already decided. Going through the ordinary write path would let
        # nearest-neighbour distance pick a different rule to retire than the
        # one the maintainer was actually correcting.
        result = await insert_rule(
            engine,
            provider,
            repo_id=repo_id,
            candidate=candidate,
            origin="correction",
        )
        rule_id = result.rule_id
        statement_vector = result.statement_vector
        corrected_id = str(target["id"])
        await supersede(
            engine,
            repo_id=repo_id,
            old_rule_id=corrected_id,
            replacement_rule_id=rule_id,
            reason=f"corrected by {maintainer_login}: {reason}"
            if reason
            else f"corrected by {maintainer_login}",
        )
        outcome = "superseded"

    else:
        # Nothing the answer cited is the target, so fall back to comparing
        # against memory as a whole. This is the ordinary write path.
        async def judge(**kwargs):
            return await chat.complete_json(contradiction.build_messages(**kwargs))

        result = await persist_rule(
            engine,
            provider,
            repo_id=repo_id,
            candidate=candidate,
            judge=judge,
            origin="correction",
        )
        rule_id = result.rule_id
        statement_vector = result.statement_vector
        outcome = result.outcome
        corrected_id = result.superseded_rule_id

    # A merge means memory already said this, so there is nothing to sweep: the
    # surviving rule is the one that was already there.
    swept: list[tuple[str, str, str]] = []
    if statement_vector and outcome != "merged":
        swept = await sweep_contradicted(
            engine,
            chat,
            repo_id=repo_id,
            statement=statement,
            statement_vector=statement_vector,
            keep_rule_id=rule_id,
            maintainer_login=maintainer_login,
            already_retired={corrected_id} if corrected_id else set(),
        )

    async def record() -> str:
        async with engine.begin() as conn:
            return str(
                (
                    await conn.execute(
                        RECORD_CORRECTION,
                        {
                            "repo_id": repo_id,
                            "session_id": session_id,
                            "turn_number": turn_number,
                            "comment_id": comment_id,
                            "corrected_rule_id": corrected_id,
                            "replacement_rule_id": rule_id if outcome != "merged" else None,
                            "maintainer_login": maintainer_login,
                        },
                    )
                ).scalar_one()
            )

    correction_id = await with_retry(record, description="record correction")

    return CorrectionResult(
        correction_id=correction_id,
        comment_id=comment_id,
        statement=statement,
        rule_id=rule_id,
        outcome=outcome,
        corrected_rule_id=corrected_id,
        corrected_statement=target["statement"] if target is not None else None,
        reason=reason,
        also_retired=[statement for _, statement, _ in swept],
    )


async def main_async(args: argparse.Namespace) -> int:
    settings = get_settings()
    engine = create_engine(args.dsn)
    provider = OpenAIEmbeddings(
        settings.openai_api_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dim,
    )
    chat = ChatClient(settings.openai_api_key, model=args.model, max_spend=args.max_spend)

    owner, _, name = args.repo.partition("/")
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

        result = await apply_correction(
            engine,
            provider,
            chat,
            repo_id=repo_id,
            session_id=args.session_id,
            turn_number=args.turn,
            maintainer_login=args.maintainer,
            correction=args.correction,
        )
    finally:
        await provider.aclose()
        await chat.aclose()
        await engine.dispose()

    print(f"recorded as: {result.statement}")
    print()
    if result.outcome == "superseded":
        print(f"retired: {result.corrected_statement}")
        print(f"replaced by rule {result.rule_id}")
        if result.reason:
            print(f"reason: {result.reason}")
    elif result.outcome == "merged":
        print("memory already held this. Recorded as further evidence for:")
        print(f"  {result.corrected_statement}")
    else:
        print(f"added as a new rule {result.rule_id}; nothing was retired")

    for statement in result.also_retired:
        print(f"also retired: {statement}")

    print(f"\n[correction {result.correction_id} | ${chat.spent:.4f}]")
    return 0


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Correct an answer the agent gave.")
    parser.add_argument("session_id")
    parser.add_argument("turn", type=int)
    parser.add_argument("correction", help="What the project actually does.")
    parser.add_argument(
        "--as",
        dest="maintainer",
        required=True,
        help="The maintainer making the correction. Recorded with it.",
    )
    parser.add_argument("--dsn", default=settings.cockroach_dsn or None)
    parser.add_argument("--repo", default=settings.repo_slug)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max-spend", type=float, default=0.02)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if not args.dsn:
        parser.error("no DSN: pass --dsn or set COCKROACH_DSN in .env")

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)-7s | %(message)s",
        stream=sys.stderr,
    )
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
