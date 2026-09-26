"""System-level invariants: properties that must hold however the parts change.

The other test modules check units. Each one is right about its own function and
none of them would notice if two correct functions were wired together wrongly,
which is where the expensive failures in this project have actually come from: a
correction that retired its target and left two duplicates active, a citation
whose label pointed at a different repository, a webhook path that worked
perfectly and was entered twice.

So these are written as claims about the whole system rather than about a
function, and each one names what a contributor would see if it broke. They are
deliberately blunt. A test here failing is not a regression in a detail, it is
the system doing the thing it advertises it cannot do.

Four invariants:

  1. A retired rule never reaches an answer.
  2. A citation never appears in a released answer unless its source was
     retrieved.
  3. A redelivered webhook never produces a second comment.
  4. A correction that says nothing usable never changes memory.

Nothing here touches the cluster. The database is faked at the engine boundary,
which is the same seam the rest of the suite uses.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from precedent.agent.answer import NO_MEMORY, UNVERIFIABLE, answer_question
from precedent.agent.correct import UnusableCorrection, apply_correction
from precedent.agent.retrieve import Recall, RetrievedRule, rank_rules, recall
from precedent.agent.review import ReviewDecision
from precedent.agent.session import Turn
from precedent.api.github_app import AlreadyHandled, act_on_pull_request
from precedent.memory.search import SearchHit

NOW = datetime(2026, 9, 26, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeChat:
    """Returns queued replies and records every call, so silence is assertable."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    async def complete_json(self, messages):
        self.calls.append(messages)
        if not self.replies:
            raise AssertionError("chat called more times than the test expected")
        return self.replies.pop(0)


class FakeProvider:
    """An embedding provider that records what it was asked to embed."""

    model = "text-embedding-3-small"

    def __init__(self):
        self.calls: list[list[str]] = []

    async def embed(self, texts):
        self.calls.append(list(texts))
        return [[0.0] * 1536 for _ in texts]


class Result:
    def __init__(self, rows=(), scalar=None):
        self._rows = list(rows)
        self._scalar = scalar

    def mappings(self):
        return self

    def all(self):
        return self._rows

    def scalar_one_or_none(self):
        return self._scalar


def _hit(**overrides) -> SearchHit:
    base = {
        "id": "c1",
        "pr_number": 65052,
        "kind": "review_thread",
        "body": "Put the issue number next to the assertion.",
        "author": "maint",
        "is_maintainer": True,
        "file_path": None,
        "url": None,
        "created_at": NOW,
        "distance": 0.3,
    }
    base.update(overrides)
    return SearchHit(**base)


def _retrieved(**overrides) -> RetrievedRule:
    base = {
        "id": "r1",
        "statement": "Add a whatsnew note for every bug fix.",
        "rationale": None,
        "scope": "process",
        "scope_pattern": None,
        "confidence": 0.8,
        "evidence_count": 4,
        "distance": 0.4,
        "citations": [_hit()],
    }
    base.update(overrides)
    return RetrievedRule(**base)


# ---------------------------------------------------------------------------
# 1. A retired rule never reaches an answer
# ---------------------------------------------------------------------------


def _rule_row(rule_id, distance, *, origin="extracted", status="active", confidence=0.8):
    return {
        "id": rule_id,
        "statement": f"statement for {rule_id}",
        "rationale": None,
        "scope": "process",
        "scope_pattern": None,
        "confidence": confidence,
        "evidence_count": 3,
        "status": status,
        "origin": origin,
        "distance": distance,
    }


class RecallEngine:
    """Serves whatever rule rows the test names, and records the bound parameters.

    Deliberately does not apply the status filter itself. The point of these
    tests is what happens when the database hands back a row it should not have,
    so the fake is the failure being simulated.
    """

    def __init__(self, rule_rows):
        self.rule_rows = rule_rows
        self.params: list[dict] = []

    def connect(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, statement, params=None):
        self.params.append(dict(params or {}))
        # Only the rule search is served. Evidence and the comment fallback come
        # back empty, so what reaches Recall is whatever survived ranking.
        if "FROM rules" in str(statement):
            return Result(self.rule_rows)
        return Result([])


class TestARetiredRuleNeverReachesAnAnswer:
    """What a superseded rule means, and why ranking is where this can break.

    Supersession is the whole correction mechanism: a maintainer says the
    project does X, the rule saying Y is retired, and the answer changes. If a
    retired rule can still be retrieved, the correction silently did nothing,
    and the contributor gets the abandoned convention with a citation under it.

    `rank_rules` is the dangerous place rather than the query, because promotion
    exists to lift a maintainer's statement past the distance ordering, and a
    correction that was itself later corrected is both stated and retired. That
    combination promotes the retired rule over the one that replaced it.
    """

    def test_ranking_drops_a_retired_rule_the_query_should_not_have_returned(self):
        rows = [
            _rule_row("retired", 0.10, status="superseded"),
            _rule_row("active", 0.90),
        ]
        assert [r["id"] for r in rank_rules(rows, 5)] == ["active"]

    def test_a_retired_correction_is_not_promoted_over_the_rule_that_replaced_it(self):
        # The worst shape this can take. Both are corrections, so both are
        # eligible for promotion; the retired one is nearer, so distance alone
        # would put the abandoned convention first.
        rows = [
            _rule_row("was_corrected", 0.20, origin="correction", status="superseded"),
            _rule_row("replacement", 0.80, origin="correction"),
        ]
        chosen = [r["id"] for r in rank_rules(rows, 5)]
        assert chosen == ["replacement"]

    def test_rows_without_a_status_are_still_treated_as_active(self):
        # Callers that do not select the column must keep working.
        rows = [{"id": "a", "distance": 0.4, "origin": "extracted"}]
        assert [r["id"] for r in rank_rules(rows, 5)] == ["a"]

    @pytest.mark.asyncio
    async def test_a_retired_rule_does_not_survive_into_recall(self):
        engine = RecallEngine([_rule_row("retired", 0.1, status="superseded")])
        got = await recall(
            engine,
            FakeProvider(),
            repo_id="repo",
            question="where does the note go?",
            comment_k=0,
        )
        assert [r.id for r in got.rules] == []

    def test_the_status_filter_sits_outside_the_vector_ordering(self):
        # Any metadata predicate in the same SELECT as the vector ordering
        # defeats the index, so the filter has to run over the candidate pool
        # rather than beside `ORDER BY embedding <-> ...`. This is the shape
        # that keeps both properties at once, and it is easy to undo by
        # "tidying" the query.
        from precedent.agent.retrieve import SEARCH_RULES

        sql = str(SEARCH_RULES)
        inner = sql.split("FROM rules")[1].split("ORDER BY")[0]
        assert "status" not in inner, "the status filter moved inside the vector subquery"
        assert "status = 'active'" in sql


# ---------------------------------------------------------------------------
# 2. A citation never appears in a released answer unless its source was retrieved
# ---------------------------------------------------------------------------


class TestACitationAlwaysResolves:
    """The one claim the evaluation actually settled.

    Against the same 29 questions, the bare model cited 24 pull requests and not
    one of them existed. Precedent cited 43 and every one resolved. That is the
    difference the whole system is for, and it rests entirely on the answer
    being discarded rather than flagged when verification fails.
    """

    @pytest.mark.asyncio
    async def test_an_invented_pull_request_discards_the_whole_answer(self):
        chat = FakeChat(
            {
                "answered": True,
                "answer": "Put it at the top of the test [PR #99999].",
                "confidence": "high",
            }
        )
        answer = await answer_question(chat, Recall(question="q", rules=[_retrieved()]))

        assert not answer.answered
        assert answer.text == UNVERIFIABLE
        assert answer.invented_prs == [99999]

    @pytest.mark.asyncio
    async def test_the_discarded_prose_is_never_shown(self):
        # Returning the text with a flag beside it was the old behaviour. The
        # flag is not what a reader sees; the fluent paragraph is.
        chat = FakeChat(
            {"answered": True, "answer": "Confident nonsense [PR #99999].", "confidence": "high"}
        )
        answer = await answer_question(chat, Recall(question="q", rules=[_retrieved()]))
        assert "Confident nonsense" not in answer.text

    @pytest.mark.asyncio
    async def test_a_citation_that_was_retrieved_survives(self):
        chat = FakeChat(
            {"answered": True, "answer": "Next to the assertion [PR #65052].", "confidence": "high"}
        )
        answer = await answer_question(chat, Recall(question="q", rules=[_retrieved()]))

        assert answer.answered
        assert answer.cited_prs == [65052]
        assert answer.is_trustworthy

    @pytest.mark.asyncio
    async def test_the_correction_sentinel_cannot_be_cited_as_a_pull_request(self):
        # Corrections carry pr_number 0 because the column is NOT NULL. If the
        # sentinel were admitted to the set of available numbers, "[PR #0]"
        # would verify: a citation that looks checkable and links nowhere.
        rule = _retrieved(
            origin="correction",
            citations=[_hit(kind="maintainer_correction", pr_number=0, author="maint")],
        )
        chat = FakeChat({"answered": True, "answer": "Do it [PR #0].", "confidence": "high"})
        answer = await answer_question(chat, Recall(question="q", rules=[rule]))

        assert not answer.answered
        assert answer.invented_prs == [0]

    @pytest.mark.asyncio
    async def test_crediting_a_maintainer_who_corrected_nothing_discards_the_answer(self):
        rule = _retrieved(
            origin="correction",
            citations=[_hit(kind="maintainer_correction", pr_number=0, author="maint")],
        )
        chat = FakeChat(
            {
                "answered": True,
                "answer": "Do it [correction by someone_else].",
                "confidence": "high",
            }
        )
        answer = await answer_question(chat, Recall(question="q", rules=[rule]))

        assert not answer.answered
        assert answer.invented_corrections == ["someone_else"]

    @pytest.mark.asyncio
    async def test_empty_memory_never_reaches_the_model_at_all(self):
        # A model handed nothing can still write something confident, so the
        # cheapest guard is not to ask. This also makes the refusal free.
        chat = FakeChat()
        answer = await answer_question(chat, Recall(question="q"))

        assert not answer.answered
        assert answer.text == NO_MEMORY
        assert chat.calls == []


# ---------------------------------------------------------------------------
# 3. A redelivered webhook never produces a second comment
# ---------------------------------------------------------------------------


class ClaimEngine:
    """Simulates `pr_reviews`, including ON CONFLICT DO NOTHING.

    Faithful about the one detail that matters: `RELEASE_REVIEW` only deletes a
    claim that never reached a decision, so a recorded review cannot be released
    and re-entered.
    """

    def __init__(self):
        self.claims: set[tuple[str, int]] = set()
        self.decided: set[tuple[str, int]] = set()
        self.records: list[dict] = []

    def begin(self):
        return self

    def connect(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, statement, params=None):
        sql = str(statement)
        params = dict(params or {})
        key = (params.get("source_repo"), params.get("pr_number"))

        if "INSERT INTO pr_reviews" in sql:
            if key in self.claims:
                return Result(scalar=None)
            self.claims.add(key)
            return Result(scalar=params["pr_number"])

        if "DELETE FROM pr_reviews" in sql:
            if key not in self.decided:
                self.claims.discard(key)
            return Result()

        if "UPDATE pr_reviews" in sql:
            self.decided.add(key)
            self.records.append(params)
            return Result()

        return Result()


class FakeGitHub:
    def __init__(self, paths=("pandas/core/groupby/groupby.py",), fail_on_comment=False):
        self.paths = list(paths)
        self.fail_on_comment = fail_on_comment
        self.comments: list[tuple[str, int, str]] = []

    async def changed_files(self, repo, pr_number):
        return self.paths

    async def comment(self, repo, pr_number, body):
        if self.fail_on_comment:
            raise RuntimeError("GitHub returned 502")
        self.comments.append((repo, pr_number, body))
        return f"https://github.com/{repo}/pull/{pr_number}#issuecomment-1"


def _speaks(**overrides):
    base = {
        "pr_number": 7,
        "paths": ["pandas/core/groupby/groupby.py"],
        "body": "three conventions",
    }
    base.update(overrides)
    return ReviewDecision(**base)


PARSED = {"repo": "pyarchana/demo", "pr_number": 7, "title": "Fix groupby apply"}


async def _act(engine, github, monkeypatch, decision=None, fail_review=False):
    async def fake_review(*args, **kwargs):
        if fail_review:
            raise RuntimeError("retrieval failed")
        return decision if decision is not None else _speaks()

    monkeypatch.setattr("precedent.api.github_app.review", fake_review)
    return await act_on_pull_request(
        engine,
        FakeProvider(),
        github,
        repo_id="repo",
        parsed=PARSED,
        trigger="@precedent",
        source_repo="pandas-dev/pandas",
    )


class TestAWebhookRetryNeverCommentsTwice:
    """GitHub's webhook timeout is 10 seconds and this path exceeds it cold.

    Retries are therefore routine rather than exceptional, which is why the
    claim is taken before the work and not after. The failure this prevents is
    not an internal inconsistency, it is two identical bot comments on a first
    time contributor's pull request, and nothing takes that back.
    """

    @pytest.mark.asyncio
    async def test_the_first_delivery_comments_once(self, monkeypatch):
        engine, github = ClaimEngine(), FakeGitHub()
        decision = await _act(engine, github, monkeypatch)

        assert decision.will_speak
        assert len(github.comments) == 1
        assert len(engine.records) == 1

    @pytest.mark.asyncio
    async def test_a_redelivery_of_the_same_event_posts_nothing(self, monkeypatch):
        engine, github = ClaimEngine(), FakeGitHub()
        await _act(engine, github, monkeypatch)

        with pytest.raises(AlreadyHandled):
            await _act(engine, github, monkeypatch)

        assert len(github.comments) == 1

    @pytest.mark.asyncio
    async def test_the_claim_is_taken_before_the_diff_is_read(self, monkeypatch):
        # Ordering is the whole mechanism. Claiming after the work would leave
        # the window open for exactly as long as the work takes, which is the
        # window GitHub retries inside.
        engine = ClaimEngine()

        class Watchful(FakeGitHub):
            async def changed_files(self, repo, pr_number):
                assert engine.claims, "the diff was read before the claim was taken"
                return self.paths

        await _act(engine, Watchful(), monkeypatch)

    @pytest.mark.asyncio
    async def test_a_failure_during_review_gives_the_claim_back(self, monkeypatch):
        # This direction is safe: the work never reached a comment, so a retry
        # cannot duplicate one, and refusing the retry would lose the review to
        # a transient error.
        engine, github = ClaimEngine(), FakeGitHub()
        with pytest.raises(RuntimeError):
            await _act(engine, github, monkeypatch, fail_review=True)

        assert engine.claims == set()

        await _act(engine, github, monkeypatch)
        assert len(github.comments) == 1

    @pytest.mark.asyncio
    async def test_a_failure_while_posting_does_not_give_the_claim_back(self, monkeypatch):
        # The opposite direction, and deliberately not symmetrical. The comment
        # may well have been created before the error surfaced, so retrying
        # risks a duplicate. Losing a comment is recoverable by asking again;
        # posting twice is not.
        engine = ClaimEngine()
        with pytest.raises(RuntimeError):
            await _act(engine, FakeGitHub(fail_on_comment=True), monkeypatch)

        assert engine.claims, "the claim was released after a possible partial post"

        with pytest.raises(AlreadyHandled):
            await _act(engine, FakeGitHub(), monkeypatch)

    @pytest.mark.asyncio
    async def test_staying_silent_still_consumes_the_claim(self, monkeypatch):
        # Silence is a decision, and it has to be recorded as one. A retry that
        # found no claim would run the whole retrieval again to reach the same
        # silence, on every redelivery, for nothing.
        engine, github = ClaimEngine(), FakeGitHub()
        quiet = ReviewDecision(pr_number=7, paths=[], silent_reason="no confident rules")

        decision = await _act(engine, github, monkeypatch, decision=quiet)
        assert not decision.will_speak
        assert github.comments == []

        with pytest.raises(AlreadyHandled):
            await _act(engine, github, monkeypatch)


# ---------------------------------------------------------------------------
# 4. A correction that says nothing usable never changes memory
# ---------------------------------------------------------------------------


class WriteRecordingEngine:
    """Records every statement executed, so "changed nothing" is checkable."""

    def __init__(self):
        self.statements: list[str] = []

    def begin(self):
        return self

    def connect(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, statement, params=None):
        self.statements.append(str(statement))
        return Result()


class TestAnUnusableCorrectionChangesNothing:
    """ "No, that's wrong" is not a correction, and treating it as one backfires.

    With nothing but the question and the answer to work from, the drafting
    model restates the answer. That restatement is then judged the same as the
    rule it came from and merged in as further evidence, so a maintainer's
    objection raises the confidence of the thing they objected to.

    That happened. "no actually its something else" left the disputed rule with
    one more supporting comment, contributed by the person disputing it.

    The guard is an ordering as much as a check: drafting happens before any
    write. If the two were ever swapped, a refused correction would still leave
    its text in `review_comments` as kind `maintainer_correction`, and that
    table is retrieved and cited on the same path as everything else. The
    contentless objection would become citable evidence.
    """

    @pytest.fixture
    def turn(self, monkeypatch):
        loaded = Turn("s", 1, "where does the number go?", "at the top", cited_rule_ids=["r1"])

        async def fake_load_turn(engine, **kwargs):
            return loaded

        monkeypatch.setattr("precedent.agent.correct.load_turn", fake_load_turn)
        return loaded

    @pytest.mark.asyncio
    async def test_nothing_is_written(self, turn):
        engine, provider = WriteRecordingEngine(), FakeProvider()
        chat = FakeChat({"usable": False, "needed": "say where the number goes"})

        with pytest.raises(UnusableCorrection):
            await apply_correction(
                engine,
                provider,
                chat,
                repo_id="repo",
                session_id="s",
                turn_number=1,
                maintainer_login="maint",
                correction="no actually its something else",
            )

        assert engine.statements == []

    @pytest.mark.asyncio
    async def test_the_objection_is_never_embedded(self, turn):
        # An embedding call is both spend and the first half of storing the
        # text. Neither should happen for something that was refused.
        engine, provider = WriteRecordingEngine(), FakeProvider()
        chat = FakeChat({"usable": False})

        with pytest.raises(UnusableCorrection):
            await apply_correction(
                engine,
                provider,
                chat,
                repo_id="repo",
                session_id="s",
                turn_number=1,
                maintainer_login="maint",
                correction="wrong",
            )

        assert provider.calls == []

    @pytest.mark.asyncio
    async def test_the_refusal_says_what_would_make_it_usable(self, turn):
        # A maintainer who is told "no" and not told why will type the same
        # thing again. The model's own words are used when it gives them.
        engine, provider = WriteRecordingEngine(), FakeProvider()
        chat = FakeChat({"usable": False, "needed": "say where the number goes"})

        with pytest.raises(UnusableCorrection) as caught:
            await apply_correction(
                engine,
                provider,
                chat,
                repo_id="repo",
                session_id="s",
                turn_number=1,
                maintainer_login="maint",
                correction="nope",
            )

        assert "say where the number goes" in str(caught.value)
