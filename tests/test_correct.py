"""Tests for the correction loop.

The database work is exercised against the cluster by hand; what is tested here
is the decision logic, because that is where a correction can go wrong quietly.
Two failures matter more than the rest:

  * A correction absorbed as *further evidence for* the rule it contradicts.
    That does not merely fail to fix the error, it raises the confidence of
    the wrong answer, and the contributor who asks tomorrow gets it stated
    more firmly than the contributor who asked today.
  * A correction superseding a rule the answer never used, because the model's
    wording happened to sit nearer to something else.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from precedent.agent.answer import (
    Answer,
    citation_label,
    correction_author,
    extract_correction_citations,
    render_material,
)
from precedent.agent.correct import (
    UnusableCorrection,
    _find_target,
    _statement_from_correction,
    sweep_contradicted,
)
from precedent.agent.retrieve import Recall, RetrievedRule
from precedent.agent.session import Turn, _uuid_array
from precedent.extract.confidence import STATED_DIRECTLY_FLOOR, score
from precedent.memory.search import SearchHit

NOW = datetime(2026, 8, 7, tzinfo=UTC)


def _hit(**overrides) -> SearchHit:
    base = {
        "id": "c1",
        "pr_number": 12345,
        "kind": "review_thread",
        "body": "Use a fixture here.",
        "author": "maint",
        "is_maintainer": True,
        "file_path": None,
        "url": None,
        "created_at": NOW,
        "distance": 0.4,
    }
    base.update(overrides)
    return SearchHit(**base)


def _rule(**overrides) -> RetrievedRule:
    base = {
        "id": "r1",
        "statement": "Add a whatsnew note for every bug fix.",
        "rationale": None,
        "scope": "process",
        "scope_pattern": None,
        "confidence": 0.8,
        "evidence_count": 4,
        "distance": 0.5,
    }
    base.update(overrides)
    return RetrievedRule(**base)


class FakeChat:
    """Returns queued replies and remembers what it was asked."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    async def complete_json(self, messages):
        self.calls.append(messages)
        if not self.replies:
            raise AssertionError("chat called more times than the test expected")
        return self.replies.pop(0)


class TestConfidenceFloor:
    def test_a_correction_is_not_weak_evidence(self):
        # One author, one occasion, no history. Scored on the ordinary terms
        # this lands near the floor, which would have the agent hedge on
        # something a maintainer stated outright.
        inferred = score(distinct_authors=1, distinct_prs=1, last_evidence_at=NOW, now=NOW)
        stated = score(
            distinct_authors=1,
            distinct_prs=1,
            last_evidence_at=NOW,
            now=NOW,
            stated_directly=True,
        )
        assert inferred.confidence < 0.4
        assert stated.confidence == STATED_DIRECTLY_FLOOR

    def test_a_well_evidenced_convention_can_still_outrank_a_correction(self):
        strong = score(
            distinct_authors=5,
            distinct_prs=8,
            first_evidence_at=NOW - timedelta(days=1500),
            last_evidence_at=NOW,
            now=NOW,
        )
        assert strong.confidence > STATED_DIRECTLY_FLOOR

    def test_the_floor_never_lowers_a_score(self):
        strong = score(
            distinct_authors=5,
            distinct_prs=8,
            first_evidence_at=NOW - timedelta(days=1500),
            last_evidence_at=NOW,
            now=NOW,
        )
        floored = score(
            distinct_authors=5,
            distinct_prs=8,
            first_evidence_at=NOW - timedelta(days=1500),
            last_evidence_at=NOW,
            now=NOW,
            stated_directly=True,
        )
        assert floored.confidence == strong.confidence


class TestCitationLabels:
    def test_a_correction_is_never_rendered_as_a_pull_request(self):
        # Corrections carry the zero sentinel. "[PR #0]" would be a citation
        # that looks checkable and is not.
        label = citation_label(_hit(kind="maintainer_correction", pr_number=0, author="jbrock"))
        assert "PR #" not in label
        assert label == "[correction by jbrock, 2026-08-07]"

    def test_an_ordinary_comment_keeps_its_pull_request_citation(self):
        assert citation_label(_hit()) == "[PR #12345]"

    def test_the_full_label_is_recognised(self):
        hit = _hit(kind="maintainer_correction", pr_number=0, author="jbrock")
        label = citation_label(hit)
        assert extract_correction_citations(f"Do the thing {label}.") == ["jbrock"]

    def test_a_label_without_its_date_still_verifies(self):
        # Models routinely drop the date. Requiring the whole rendered string
        # to match reported correctly cited answers as fabricating citations,
        # which discredits the check on exactly the answers that used a
        # correction properly.
        assert extract_correction_citations("Do it [correction by jbrock].") == ["jbrock"]

    def test_an_answer_citing_a_correction_is_trustworthy(self):
        recall = Recall(
            question="q",
            rules=[
                _rule(
                    origin="correction",
                    citations=[_hit(kind="maintainer_correction", pr_number=0, author="jbrock")],
                )
            ],
        )
        supplied = [c for r in recall.rules for c in r.citations]
        available = {correction_author(c) for c in supplied}
        assert extract_correction_citations("x [correction by jbrock]")[0] in available

    def test_crediting_someone_who_corrected_nothing_is_still_caught(self):
        assert extract_correction_citations("x [correction by nobody, 2026-01-01]") == ["nobody"]

    def test_a_corrected_rule_says_so_in_the_material(self):
        material = render_material(
            Recall(question="q", rules=[_rule(origin="correction", confidence=0.85)])
        )
        assert "established by maintainer correction" in material
        assert "weakly evidenced" not in material


class TestFabricatedCorrections:
    def test_an_invented_correction_makes_an_answer_untrustworthy(self):
        answer = Answer(
            question="q",
            answered=True,
            text="Do the thing [correction by nobody, 2026-01-01].",
            invented_corrections=["[correction by nobody, 2026-01-01]"],
        )
        assert not answer.is_trustworthy

    def test_a_clean_answer_is_trustworthy(self):
        assert Answer(question="q", answered=True, text="ok [PR #1]").is_trustworthy


class TestStatementFromCorrection:
    @pytest.mark.asyncio
    async def test_a_correction_becomes_a_standalone_instruction(self):
        chat = FakeChat(
            {
                "statement": "Put the whatsnew note in the file for the next release.",
                "rationale": "The current release is already cut.",
                "scope": "docs",
                "scope_pattern": None,
            }
        )
        drafted = await _statement_from_correction(
            chat, Turn("s", 1, "where does the note go?", "in the current file"), "maint", "no, ..."
        )
        assert drafted["statement"].startswith("Put the whatsnew note")
        assert drafted["scope"] == "docs"

    @pytest.mark.asyncio
    async def test_an_unknown_scope_falls_back_rather_than_failing(self):
        chat = FakeChat({"statement": "Do the thing.", "scope": "vibes"})
        drafted = await _statement_from_correction(
            chat, Turn("s", 1, "q", "a"), "maint", "correction"
        )
        assert drafted["scope"] == "repo"

    @pytest.mark.asyncio
    async def test_a_pattern_on_a_scope_that_cannot_use_one_is_dropped(self):
        # Storing it would read as a constraint that is never applied.
        chat = FakeChat(
            {"statement": "Do the thing.", "scope": "process", "scope_pattern": "pandas/core/%"}
        )
        drafted = await _statement_from_correction(
            chat, Turn("s", 1, "q", "a"), "maint", "correction"
        )
        assert drafted["scope_pattern"] is None

    @pytest.mark.asyncio
    async def test_an_empty_statement_is_an_error_not_a_blank_rule(self):
        chat = FakeChat({"statement": "  ", "scope": "repo"})
        with pytest.raises(ValueError):
            await _statement_from_correction(chat, Turn("s", 1, "q", "a"), "maint", "correction")


class TestUnusableCorrections:
    """A correction must say what is true, not only that the answer is wrong.

    Given "no, that's wrong" alone, the drafting model has nothing but the
    question and the answer to work from, so it restates the answer. That
    restatement is judged "same" as the rule it came from and merged in as
    further evidence, which means a maintainer's objection ends up raising the
    confidence of the thing they objected to.

    That is not hypothetical. "no actually its something else" did exactly
    that, and the disputed rule gained a second supporting comment from the
    person disputing it.
    """

    @pytest.mark.asyncio
    async def test_a_contentless_correction_is_refused(self):
        chat = FakeChat({"usable": False, "needed": "say where the number goes"})
        with pytest.raises(UnusableCorrection) as caught:
            await _statement_from_correction(
                chat,
                Turn("s", 1, "where does it go?", "at the top"),
                "maint",
                "no actually its something else.",
            )
        assert "say where the number goes" in str(caught.value)

    @pytest.mark.asyncio
    async def test_the_refusal_explains_what_is_missing_even_when_the_model_does_not(self):
        chat = FakeChat({"usable": False})
        with pytest.raises(UnusableCorrection) as caught:
            await _statement_from_correction(chat, Turn("s", 1, "q", "a"), "maint", "wrong")
        assert str(caught.value).strip()

    @pytest.mark.asyncio
    async def test_a_usable_correction_is_unaffected(self):
        chat = FakeChat(
            {"usable": True, "statement": "Put it next to the assertion.", "scope": "testing"}
        )
        drafted = await _statement_from_correction(chat, Turn("s", 1, "q", "a"), "maint", "no, ...")
        assert drafted["statement"] == "Put it next to the assertion."

    @pytest.mark.asyncio
    async def test_a_response_omitting_usable_is_still_accepted(self):
        # The flag was added after the prompt shipped; absence must not be read
        # as refusal, or every correction would fail closed.
        chat = FakeChat({"statement": "Do the thing.", "scope": "repo"})
        drafted = await _statement_from_correction(chat, Turn("s", 1, "q", "a"), "maint", "no...")
        assert drafted["statement"] == "Do the thing."


class FakeEngine:
    """Serves one fixed set of rule rows to `_find_target`."""

    def __init__(self, rows):
        self.rows = rows

    def connect(self):
        rows = self.rows

        class Conn:
            async def __aenter__(self_inner):
                return self_inner

            async def __aexit__(self_inner, *exc):
                return False

            async def execute(self_inner, statement, params):
                class Result:
                    def mappings(self_r):
                        return self_r

                    def all(self_r):
                        return rows

                return Result()

        return Conn()


def _row(rule_id: str, statement: str, status: str = "active"):
    return {
        "id": rule_id,
        "statement": statement,
        "last_evidence_at": NOW - timedelta(days=400),
        "status": status,
        "confidence": 0.7,
    }


class TestFindTarget:
    @pytest.mark.asyncio
    async def test_the_contradicted_rule_is_found(self):
        engine = FakeEngine([_row("r1", "Put the note in the current release file.")])
        chat = FakeChat({"relation": "contradicts", "reason": "different file"})
        turn = Turn("s", 1, "q", "a", cited_rule_ids=["r1"])

        rule, relation, reason = await _find_target(
            engine,
            chat,
            repo_id="repo",
            turn=turn,
            statement="Use the next release file.",
            today="2026-08-07",
        )
        assert rule["id"] == "r1"
        assert relation == "contradicts"
        assert reason == "different file"

    @pytest.mark.asyncio
    async def test_a_correction_that_agrees_returns_same_not_contradicts(self):
        # This is the case that must not become a new rule: the rule was right
        # and the answer misused it.
        engine = FakeEngine([_row("r1", "Add a whatsnew note for every bug fix.")])
        chat = FakeChat({"relation": "same", "reason": "identical convention"})
        turn = Turn("s", 1, "q", "a", cited_rule_ids=["r1"])

        _, relation, _ = await _find_target(
            engine,
            chat,
            repo_id="repo",
            turn=turn,
            statement="Every bug fix needs a whatsnew note.",
            today="2026-08-07",
        )
        assert relation == "same"

    @pytest.mark.asyncio
    async def test_rules_are_checked_in_the_order_the_answer_used_them(self):
        # The first rule is the one the answer leaned on hardest, so checking
        # it first usually costs one model call instead of five.
        engine = FakeEngine(
            [_row("r2", "second rule"), _row("r1", "first rule")]  # returned out of order
        )
        chat = FakeChat({"relation": "contradicts", "reason": "x"})
        turn = Turn("s", 1, "q", "a", cited_rule_ids=["r1", "r2"])

        rule, _, _ = await _find_target(
            engine, chat, repo_id="repo", turn=turn, statement="s", today="2026-08-07"
        )
        assert rule["id"] == "r1"
        assert len(chat.calls) == 1

    @pytest.mark.asyncio
    async def test_a_rule_superseded_since_the_answer_is_not_targeted_again(self):
        engine = FakeEngine([_row("r1", "old rule", status="superseded")])
        chat = FakeChat()  # no call should be made
        turn = Turn("s", 1, "q", "a", cited_rule_ids=["r1"])

        rule, relation, _ = await _find_target(
            engine, chat, repo_id="repo", turn=turn, statement="s", today="2026-08-07"
        )
        assert rule is None
        assert relation is None
        assert not chat.calls

    @pytest.mark.asyncio
    async def test_an_unrelated_correction_targets_nothing(self):
        engine = FakeEngine([_row("r1", "Add a whatsnew note.")])
        chat = FakeChat({"relation": "compatible", "reason": "different subject"})
        turn = Turn("s", 1, "q", "a", cited_rule_ids=["r1"])

        rule, relation, _ = await _find_target(
            engine,
            chat,
            repo_id="repo",
            turn=turn,
            statement="Use pytest.raises with match.",
            today="2026-08-07",
        )
        assert rule is None
        assert relation is None

    @pytest.mark.asyncio
    async def test_an_answer_that_cited_no_rules_costs_nothing(self):
        chat = FakeChat()
        rule, _, _ = await _find_target(
            FakeEngine([]),
            chat,
            repo_id="repo",
            turn=Turn("s", 1, "q", "a"),
            statement="s",
            today="2026-08-07",
        )
        assert rule is None
        assert not chat.calls


class SweepEngine:
    """Serves neighbours to `sweep_contradicted` and records supersessions."""

    def __init__(self, rows):
        self.rows = rows
        self.superseded: list[str] = []

    def connect(self):
        return _Ctx(self.rows)

    def begin(self):
        return _Ctx(self.rows, record=self.superseded)


class _Ctx:
    def __init__(self, rows, record=None):
        self.rows = rows
        self.record = record

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, statement, params=None):
        if self.record is not None and params and "rule_id" in params:
            self.record.append(params["rule_id"])

        rows = self.rows

        class Result:
            def mappings(self_r):
                return self_r

            def all(self_r):
                return rows

        return Result()


def _neighbour(rule_id: str, statement: str, distance: float = 0.5):
    return {
        "id": rule_id,
        "statement": statement,
        "last_evidence_at": NOW - timedelta(days=300),
        "distance": distance,
    }


class TestSweep:
    """A correction that retires only its target can still lose.

    The first real correction did exactly that: the cited rule was retired and
    two near duplicates of it stayed active at higher confidence, so memory
    held the correction and the corrected claim at once.
    """

    @pytest.mark.asyncio
    async def test_near_duplicates_of_the_corrected_rule_are_retired(self):
        engine = SweepEngine(
            [
                _neighbour("r2", "Put the issue number at the top of each test."),
                _neighbour("r3", "Reference the issue number at the start of the test function."),
            ]
        )
        chat = FakeChat(
            {"relation": "contradicts", "reason": "different location"},
            {"relation": "contradicts", "reason": "different location"},
        )

        retired = await sweep_contradicted(
            engine,
            chat,
            repo_id="repo",
            statement="Put the issue number next to the assertion.",
            statement_vector="[0]",
            keep_rule_id="r1",
            maintainer_login="maint",
        )
        assert [r[0] for r in retired] == ["r2", "r3"]
        assert engine.superseded == ["r2", "r3"]

    @pytest.mark.asyncio
    async def test_a_compatible_neighbour_survives(self):
        # Being about the same subject is not being in conflict. Retiring these
        # would make a correction destroy unrelated guidance.
        engine = SweepEngine([_neighbour("r2", "Put the issue number in the PR description.")])
        chat = FakeChat({"relation": "compatible", "reason": "different place entirely"})

        retired = await sweep_contradicted(
            engine,
            chat,
            repo_id="repo",
            statement="Put it next to the assertion.",
            statement_vector="[0]",
            keep_rule_id="r1",
            maintainer_login="maint",
        )
        assert retired == []
        assert engine.superseded == []

    @pytest.mark.asyncio
    async def test_the_rule_already_retired_is_not_visited_twice(self):
        engine = SweepEngine([_neighbour("r2", "old rule")])
        chat = FakeChat()  # no comparison should be made

        retired = await sweep_contradicted(
            engine,
            chat,
            repo_id="repo",
            statement="s",
            statement_vector="[0]",
            keep_rule_id="r1",
            maintainer_login="maint",
            already_retired={"r2"},
        )
        assert retired == []
        assert not chat.calls

    @pytest.mark.asyncio
    async def test_comparisons_are_bounded(self):
        # Each comparison is a paid call, so a dense cluster of similar rules
        # must not be able to run the cost up without limit.
        engine = SweepEngine([_neighbour(f"r{i}", f"rule {i}") for i in range(2, 20)])
        chat = FakeChat(*([{"relation": "compatible", "reason": ""}] * 3))

        await sweep_contradicted(
            engine,
            chat,
            repo_id="repo",
            statement="s",
            statement_vector="[0]",
            keep_rule_id="r1",
            maintainer_login="maint",
            max_comparisons=3,
        )
        assert len(chat.calls) == 3


class TestUuidArray:
    def test_ids_are_bound_not_interpolated(self):
        sql, params = _uuid_array(["a-b-c", "d-e-f"], "rid")
        assert sql == "ARRAY[:rid_0, :rid_1]::UUID[]"
        assert params == {"rid_0": "a-b-c", "rid_1": "d-e-f"}

    def test_an_empty_list_is_a_typed_empty_array(self):
        # string_to_array on an empty string yields {''}, which fails the cast.
        sql, params = _uuid_array([], "rid")
        assert sql == "ARRAY[]::UUID[]"
        assert params == {}
