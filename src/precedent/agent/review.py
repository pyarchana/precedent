"""Say what the project already decided, before anyone has to ask.

Everything else in this system answers a question. This does not: a pull request
opens, nobody addresses the agent, and it decides on its own whether the
project's accumulated conventions have anything to say about the files that
changed. Asking is the easy case, because the question tells you what to look
for. Here the only input is a list of paths.

## Why this does not retrieve by similarity

The first version of this searched the rules by vector, the way `/ask` does,
using a description of the change as the query. It commented on 38 of 40 real
pull requests. `scripts/try_review.py --sample --sensitivity` shows why, and the
reason is not a badly chosen threshold.

Across 25 pull requests, the distance from the query to the *nearest* rule and to
the *tenth nearest* differed by a median of 0.10, whichever way the query was
phrased: the title alone, the title with directories, or the whole thing written
out as a question. Four formulations, all flat. Every rule sits at roughly the
same distance from any description of a change, because every rule is a general
statement about the same codebase, and a description of a change to that codebase
is similar to all of them at once.

A threshold over a signal that flat is drawing an arbitrary line through a blob.
Set at 1.05 it spoke on 95% of pull requests; at 0.95, on 18%; there is no value
in between that means anything, because the number it is thresholding does not
measure what it is supposed to measure.

## What it retrieves on instead

Paths, which are facts. A rule anchored to `doc/source/whatsnew/%` applies to a
pull request that changes a whatsnew file and does not apply to one that does
not. That is checkable, and it is why every line of the posted comment can name
the file that triggered it rather than a similarity score.

Two things are therefore excluded. A convention scoped `repo`, `style`,
`process` or `api` is true of every pull request ever opened, so it is not
evidence that this one needs a comment, and distance is now known to be unable to
rank them. And an anchor covering most of the repository is not an anchor:
`pandas/%` matches 75% of known paths, which says only that the change is to
pandas, which was already known. See `MAX_ANCHOR_SHARE`.

Similarity keeps a smaller and honest job: among the rules the paths have already
admitted, the closest to the pull request's title goes first. Ranking inside a
set that is already relevant is where a weak signal is still worth having.

## What this did and did not fix

Honestly, on the same 40 pull requests: the vector version spoke on 95%, and this
one speaks on 92%. Two separate bugs were real and worth fixing on their own
terms, and neither moved that number much.

What did change is what gets said. Before, the comment was drawn from whichever
`repo`-scoped platitudes happened to embed nearest, with no reason to give.
Now every cited rule is anchored to a file the contributor actually changed, and
the comment names that file. The remaining 92% is not the agent reaching: it is
that most pandas pull requests do touch `pandas/core` or a whatsnew file, and
this project genuinely does hold settled conventions about both.

The rate is left where the evidence puts it rather than tuned down to look
discerning. If it should be quieter, the honest lever is `MAX_ANCHOR_SHARE`, and
`scripts/try_review.py --sample --sensitivity` prints what each value costs.

## Silence

The third outcome still matters. Silence is recorded as a decision (migration
0008) so that "found nothing" stays distinguishable from "never ran".
"""

from __future__ import annotations

import logging
import posixpath
from collections import defaultdict
from dataclasses import dataclass, field
from fnmatch import fnmatch

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from precedent.agent.retrieve import EVIDENCE_FOR_RULES, RetrievedRule
from precedent.embed.provider import EmbeddingProvider
from precedent.embed.vector import encode
from precedent.memory.search import SearchHit

log = logging.getLogger(__name__)

# A convention the project barely evidenced is not worth interrupting somebody's
# first contribution with.
MIN_CONFIDENCE = 0.45

# Three is what a contributor reads. Ten is what they collapse.
MAX_RULES = 3

# How many pull requests to cite per rule. One is enough to check; two shows it
# was not a one-off.
CITATIONS_PER_RULE = 2

TEST_MARKERS = ("/tests/", "tests/", "test_", "_test.py", "conftest.py")
DOC_MARKERS = ("doc/", "docs/", ".rst", ".md", "whatsnew")
CODE_SUFFIXES = (".py", ".pyx", ".pxd", ".pyi", ".c", ".h", ".cpp")

# Scopes with a path hook, which are the only ones that can start a comment.
# `repo`, `style`, `process` and `api` are excluded on purpose: they are true of
# every pull request ever opened, so they are not evidence that this one needs
# anything said about it, and distance is measurably unable to tell which of them
# is relevant. See the module docstring.
ANCHORED_SCOPES = ("directory", "file", "testing", "docs")

# Candidate rules pulled before the pattern filter runs. The whole `rules` table
# is a few hundred rows, so this is not a limit that bites; it is a guard against
# a future repository with thousands.
CANDIDATE_LIMIT = 500

# An anchor matching more than this share of the repository is not an anchor.
#
# Measured, and for once there is a real gap to site it in. Across the 34
# distinct patterns in memory, selectivity against 2,181 known paths falls into
# two groups with nothing whatsoever between them: 54 rules anchor on under 10%
# of the repository (`pandas/tests/indexing/test_loc.py` at 0.0%,
# `doc/source/whatsnew/%` at 5.1%, `pandas/core/%` at 9.5%) and 25 anchor on
# over 40% (`pandas/tests/%` at 42.3%, `pandas/**/*.py` at 63.0%, `pandas/%` at
# 75.3%). Any value from 0.10 to 0.42 gives the identical answer, so the exact
# number carries no weight; `scripts/try_review.py --sensitivity` prints it.
#
# The second group is what "always add a docstring, anywhere in pandas" looks
# like as a pattern. True, and useless as a reason to comment on one particular
# pull request.
MAX_ANCHOR_SHARE = 0.25

# Distinct paths sampled to measure an anchor against. Enough to make 10% and
# 40% clearly different numbers; the shape of the distribution does not need
# more.
CORPUS_SAMPLE = 4000

REPOSITORY_PATHS = text("""
    SELECT DISTINCT file_path
    FROM review_comments
    WHERE repo_id = :repo_id AND file_path IS NOT NULL
    LIMIT :limit
""")

# Selected on scope and confidence, ordered by similarity to the pull request
# title. The vector work is deliberately outside the WHERE clause's reach: this
# does not use `idx_rules_embedding`, and does not need to, because it scans a
# few hundred rules rather than the 86,000 comments the index exists for.
CANDIDATES = text(f"""
    SELECT id, statement, rationale, scope::STRING AS scope, scope_pattern,
           confidence, evidence_count, origin,
           embedding <-> CAST(:query AS VECTOR(1536)) AS distance
    FROM rules
    WHERE repo_id = :repo_id
      AND status = 'active'
      AND confidence >= :min_confidence
      AND scope IN ({", ".join(f"CAST('{s}' AS rule_scope)" for s in ANCHORED_SCOPES)})
    ORDER BY distance
    LIMIT :limit
""")


@dataclass(slots=True)
class Applicable:
    rule: RetrievedRule
    reason: str
    """Why this rule was let through: which fact about the paths admitted it."""


@dataclass(slots=True)
class ReviewDecision:
    pr_number: int
    paths: list[str]
    considered: list[RetrievedRule] = field(default_factory=list)
    selected: list[Applicable] = field(default_factory=list)
    body: str | None = None
    silent_reason: str | None = None

    @property
    def will_speak(self) -> bool:
        return bool(self.body)


@dataclass(frozen=True, slots=True)
class Touched:
    """What kinds of thing a pull request changes."""

    tests: bool
    docs: bool
    code: bool
    directories: tuple[str, ...]

    @property
    def only_docs(self) -> bool:
        return self.docs and not self.code and not self.tests


def classify(paths: list[str]) -> Touched:
    lowered = [p.lower() for p in paths]
    return Touched(
        tests=any(marker in p for p in lowered for marker in TEST_MARKERS),
        docs=any(marker in p for p in lowered for marker in DOC_MARKERS),
        code=any(
            p.endswith(CODE_SUFFIXES) and not any(m in p for m in TEST_MARKERS) for p in lowered
        ),
        directories=tuple(sorted({posixpath.dirname(p) for p in paths if posixpath.dirname(p)})),
    )


def normalise_pattern(pattern: str | None) -> str:
    """Put a stored pattern into glob syntax.

    The patterns in memory are overwhelmingly **SQL LIKE** patterns, not globs:
    `pandas/tests/%`, `pandas/core/%`, `pandas/tests/%/conftest.py`. Nothing
    asked the extraction prompt for that and nothing documented it; it is simply
    what the model wrote, presumably because the rules were destined for a
    database.

    Matching them with `fnmatch`, where `%` is an ordinary character, meant 68 of
    the 79 path-anchored rules matched nothing whatsoever. The failure was
    invisible from the outside: the only rules that ever fired were the two
    written `pandas/**/*.py`, which match most of the repository, so the agent
    looked like it was working while running on its two least specific rules.
    """
    if not pattern:
        return ""
    cleaned = pattern.strip().strip("`").lstrip("./")
    # `%` is LIKE's "any run of characters", which is `*` here. `_` is LIKE's
    # single character, but it is also legitimate in Python filenames far more
    # often than it is a wildcard, so it is left alone.
    return cleaned.replace("%", "*")


def pattern_matches(pattern: str | None, paths: list[str]) -> bool:
    """Whether a rule's own path pattern covers anything in this change.

    Patterns arrive in whatever shape the comment suggested: a LIKE pattern, a
    glob, a bare directory, or a filename. All of them have to work, because
    rejecting the ones that are not already globs is what silently disabled most
    of the anchored rules.
    """
    cleaned = normalise_pattern(pattern)
    if not cleaned:
        return False

    if any(character in cleaned for character in "*?["):
        return any(fnmatch(path, cleaned) or fnmatch(path, f"*/{cleaned}") for path in paths)

    prefix = cleaned.rstrip("/")
    return any(
        path == prefix
        or path.startswith(f"{prefix}/")
        or f"/{prefix}/" in path
        or posixpath.basename(path) == prefix
        for path in paths
    )


def anchor_share(pattern: str | None, corpus: tuple[str, ...]) -> float:
    """What fraction of the repository a pattern covers.

    The measure of whether an anchor is an anchor. `pandas/tests/indexing/
    test_loc.py` says something about a pull request; `pandas/%` says only that
    the change is to pandas, which was already known.
    """
    if not corpus:
        return 0.0
    return sum(1 for path in corpus if pattern_matches(pattern, [path])) / len(corpus)


def applies_to(rule: RetrievedRule, paths: list[str], touched: Touched) -> str | None:
    """The reason this rule is relevant to these paths, or None if it is not.

    Everything this can return names a fact about the change, which is what makes
    it quotable back to the contributor. "This applies because you changed
    pandas/tests/indexing/test_loc.py" is checkable against their own diff.
    "This applies because it was 1.02 away" is not.

    A scope with no pattern is not an anchor. `testing` on its own means "this
    pull request changes tests", which is true of nearly every pull request
    pandas receives, and is the scope-level version of the same mistake
    `MAX_ANCHOR_SHARE` catches at the pattern level.
    """
    if rule.scope not in ANCHORED_SCOPES:
        # Applies to every pull request equally, which is the same as applying
        # to none of them in particular.
        return None

    if not pattern_matches(rule.scope_pattern, paths):
        return None

    # Name the file that matched rather than the pattern that matched it. The
    # contributor can check a path against their own diff; they have never seen
    # the pattern and have no reason to trust it.
    matched = next((path for path in paths if pattern_matches(rule.scope_pattern, [path])), None)
    return f"it changes {matched}" if matched else f"path matches {rule.scope_pattern}"


def describe(title: str | None, paths: list[str], touched: Touched) -> str:
    """The query the rules are ordered by, standing in for the question nobody asked.

    This decides nothing. Its only job is putting the most fitting of the
    already-applicable rules first, so the wording matters much less than it
    looks like it should: four formulations were measured and all ranked about
    equally flat.
    """
    parts: list[str] = []
    if title:
        parts.append(title.strip().rstrip("."))

    areas = list(touched.directories[:6]) or ["the repository root"]
    parts.append(f"Changes files in {', '.join(areas)}")

    kinds = [
        name
        for name, present in (("tests", touched.tests), ("documentation", touched.docs))
        if present
    ]
    if kinds:
        parts.append(f"Touches {' and '.join(kinds)}")

    parts.append(f"{len(paths)} files changed")
    return ". ".join(parts) + "."


def _rank(applicable: Applicable) -> tuple:
    """Order for selection: what a maintainer said, then evidence, then fit.

    A rule a maintainer stated outright outranks one inferred from a pattern
    even when the inferred one is a slightly closer match, because the cost of
    being wrong is borne by the contributor either way and only one of the two
    is something the project actually committed to.
    """
    rule = applicable.rule
    return (not rule.is_stated, -rule.confidence, rule.distance)


def select(rules: list[RetrievedRule], paths: list[str], touched: Touched) -> list[Applicable]:
    candidates: list[Applicable] = []
    for rule in rules:
        if rule.confidence < MIN_CONFIDENCE:
            continue
        reason = applies_to(rule, paths, touched)
        if reason:
            candidates.append(Applicable(rule=rule, reason=reason))

    candidates.sort(key=_rank)
    return candidates[:MAX_RULES]


def _citations(rule: RetrievedRule) -> str:
    """Links to where the convention was established, deduplicated by pull request."""
    seen: dict[int, str | None] = {}
    for hit in rule.citations:
        seen.setdefault(hit.pr_number, hit.url)
        if len(seen) >= CITATIONS_PER_RULE:
            break

    if not seen:
        return ""

    links = [f"[#{number}]({url})" if url else f"#{number}" for number, url in seen.items()]
    established = " and ".join(links)
    return f"  \n  Established in {established}."


def render(selected: list[Applicable], trigger: str) -> str:
    """The comment itself.

    Not written by a model. The statements were already distilled by one and
    tied to real pull requests, and regenerating them as prose would put a
    second chance to invent something between the evidence and the contributor,
    in exchange for nicer sentences. The one thing this comment has going for it
    is that every claim in it is checkable.
    """
    count = {1: "one convention", 2: "two conventions", 3: "three conventions"}.get(
        len(selected), f"{len(selected)} conventions"
    )

    lines = [
        "**From this project's review history**",
        "",
        (
            f"Nobody asked me. I read the files this pull request changes and found {count} "
            "the project has settled before, each linked to where it was settled."
        ),
        "",
    ]

    for item in selected:
        rule = item.rule
        lines.append(f"- **{rule.statement.rstrip('.')}.**{_citations(rule)}")
        if rule.rationale:
            lines.append(f"  \n  {rule.rationale.rstrip('.')}.")

    lines += [
        "",
        (
            f"If this is wrong or out of date, a maintainer can say so: reply `{trigger} "
            "<the convention as it actually is>` and I will hold the correction from then on."
        ),
    ]
    return "\n".join(lines)


# Cached per repository. The path list changes only when the corpus is
# reingested, and re-reading a few thousand rows on every pull request to
# recompute a number that does not move would be paying for nothing.
_corpus: dict[str, tuple[str, ...]] = {}


async def repository_paths(engine: AsyncEngine, repo_id: str) -> tuple[str, ...]:
    """Distinct paths the repository is known to contain, for measuring anchors."""
    if repo_id not in _corpus:
        async with engine.connect() as conn:
            rows = await conn.execute(
                REPOSITORY_PATHS, {"repo_id": repo_id, "limit": CORPUS_SAMPLE}
            )
        _corpus[repo_id] = tuple(row[0] for row in rows)
        log.info("anchor corpus for %s: %d paths", repo_id, len(_corpus[repo_id]))
    return _corpus[repo_id]


async def candidates(
    engine: AsyncEngine,
    provider: EmbeddingProvider,
    *,
    repo_id: str,
    query: str,
) -> list[RetrievedRule]:
    """Every active rule with a specific path anchor, ordered by fit to the change."""
    vector = encode((await provider.embed([query]))[0])
    corpus = await repository_paths(engine, repo_id)

    async with engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    CANDIDATES,
                    {
                        "repo_id": repo_id,
                        "query": vector,
                        "min_confidence": MIN_CONFIDENCE,
                        "limit": CANDIDATE_LIMIT,
                    },
                )
            )
            .mappings()
            .all()
        )

    # Memoised across rules, because 22 of them share `pandas/core/%` and there
    # is no reason to walk the corpus 22 times for one answer.
    shares: dict[str, float] = {}
    kept: list[RetrievedRule] = []
    dropped = 0
    for row in rows:
        pattern = row["scope_pattern"]
        if not pattern:
            dropped += 1
            continue
        if pattern not in shares:
            shares[pattern] = anchor_share(pattern, corpus)
        if shares[pattern] > MAX_ANCHOR_SHARE:
            dropped += 1
            continue

        kept.append(
            RetrievedRule(
                id=str(row["id"]),
                statement=row["statement"],
                rationale=row["rationale"],
                scope=str(row["scope"]),
                scope_pattern=pattern,
                confidence=float(row["confidence"]),
                evidence_count=row["evidence_count"],
                distance=float(row["distance"]),
                origin=str(row["origin"]),
            )
        )

    log.info("%d rules with a specific anchor, %d too broad or unanchored", len(kept), dropped)
    return kept


async def attach_citations(
    engine: AsyncEngine, *, repo_id: str, selected: list[Applicable]
) -> None:
    """Fetch the comments each selected rule was learned from.

    Only for the rules that made it, rather than for everything considered. A
    citation has to be evidence *for the rule being stated*, so it is keyed on
    rule id rather than searched for, and there is no reason to pay for the
    hundreds that were not chosen.
    """
    if not selected:
        return

    by_rule: dict[str, list[SearchHit]] = defaultdict(list)
    async with engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    EVIDENCE_FOR_RULES,
                    {
                        "repo_id": repo_id,
                        "rule_ids": [item.rule.id for item in selected],
                        "per_rule": CITATIONS_PER_RULE,
                    },
                )
            )
            .mappings()
            .all()
        )

    for row in rows:
        by_rule[str(row["rule_id"])].append(
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
                distance=0.0,
            )
        )

    for item in selected:
        item.rule.citations = by_rule.get(item.rule.id, [])


async def review(
    engine: AsyncEngine,
    provider: EmbeddingProvider,
    *,
    repo_id: str,
    pr_number: int,
    title: str | None,
    paths: list[str],
    trigger: str = "@precedent",
) -> ReviewDecision:
    """Decide what, if anything, to say about a pull request."""
    decision = ReviewDecision(pr_number=pr_number, paths=paths)

    if not paths:
        decision.silent_reason = "the pull request changes no files"
        return decision

    touched = classify(paths)
    decision.considered = await candidates(
        engine, provider, repo_id=repo_id, query=describe(title, paths, touched)
    )
    decision.selected = select(decision.considered, paths, touched)

    if not decision.selected:
        decision.silent_reason = (
            f"nothing applies: {len(decision.considered)} anchored rules, none matching these paths"
        )
        log.info("#%s: staying quiet, %s", pr_number, decision.silent_reason)
        return decision

    await attach_citations(engine, repo_id=repo_id, selected=decision.selected)
    decision.body = render(decision.selected, trigger)
    log.info(
        "#%s: %d of %d rules apply (%s)",
        pr_number,
        len(decision.selected),
        len(decision.considered),
        ", ".join(item.rule.scope for item in decision.selected),
    )
    return decision
