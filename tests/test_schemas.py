"""Tests for what happens when a model returns something unexpected.

Every one of these decides whether to write to memory or to state something to
a contributor. The direction of failure is the whole point: a garbled response
is not evidence for either, so it has to end in doing nothing rather than in
doing something arbitrary.
"""

from __future__ import annotations

from precedent.extract.schemas import (
    AnswerOutput,
    DraftedRule,
    ExtractedRule,
    Verdict,
    validated,
)


class TestTheCoercionBug:
    """`bool("false")` is `True`, and this cost a refusal.

    A model writing the string "false" where the prompt asked for the literal
    turned a declined answer into a delivered one, in the one system whose
    entire argument is that it refuses when it does not know.
    """

    def test_the_string_false_is_not_true(self):
        assert bool("false") is True  # the trap
        assert validated(AnswerOutput, {"answered": "false"}, context="t").answered is False

    def test_the_string_true_is_still_true(self):
        assert validated(AnswerOutput, {"answered": "true"}, context="t").answered is True

    def test_a_declined_draft_stays_declined(self):
        # `result.get("usable") is False` was never satisfied by "false", so a
        # correction the model refused to draft was written anyway.
        draft = validated(DraftedRule, {"usable": "false", "statement": "x"}, context="t")
        assert draft.is_usable is False


class TestFailingInert:
    def test_a_missing_payload_becomes_a_refusal(self):
        assert validated(AnswerOutput, None, context="t").answered is False

    def test_a_non_object_payload_becomes_a_refusal(self):
        assert validated(AnswerOutput, "sorry, I cannot help", context="t").answered is False
        assert validated(AnswerOutput, ["a", "list"], context="t").answered is False

    def test_an_unknown_relation_retires_nothing(self):
        # "compatible" is the outcome that changes no rules.
        assert validated(Verdict, {"relation": "maybe"}, context="t").relation == "compatible"

    def test_a_garbled_verdict_retires_nothing(self):
        assert validated(Verdict, {}, context="t").relation == "compatible"
        assert validated(Verdict, "contradicts", context="t").relation == "compatible"

    def test_an_empty_draft_writes_nothing(self):
        assert validated(DraftedRule, {}, context="t").is_usable is False

    def test_a_cluster_with_no_convention_is_not_one(self):
        assert validated(ExtractedRule, {}, context="t").is_convention is False


class TestStrictnessIsAimed:
    """Fields that decide are strict. Fields that only label degrade.

    Discarding a maintainer's correction because the model wrote "vibes"
    instead of "testing" throws away the part that mattered to protect the part
    that did not.
    """

    def test_an_unknown_scope_keeps_the_statement(self):
        draft = validated(
            DraftedRule, {"statement": "Do the thing.", "scope": "vibes"}, context="t"
        )
        assert draft.statement == "Do the thing."
        assert draft.scope == "repo"
        assert draft.is_usable is True

    def test_an_unknown_confidence_keeps_the_answer(self):
        parsed = validated(
            AnswerOutput,
            {"answered": True, "answer": "yes [PR #1]", "confidence": "extremely"},
            context="t",
        )
        assert parsed.answered is True
        assert parsed.answer == "yes [PR #1]"
        assert parsed.confidence == "low"

    def test_a_known_scope_survives_untouched(self):
        assert validated(
            DraftedRule, {"statement": "x", "scope": "testing"}, context="t"
        ).scope == ("testing")

    def test_labels_are_matched_case_insensitively(self):
        assert validated(
            DraftedRule, {"statement": "x", "scope": " Testing "}, context="t"
        ).scope == ("testing")


class TestWellFormedResponsesAreUnaffected:
    def test_an_answer_passes_through(self):
        parsed = validated(
            AnswerOutput,
            {"answered": True, "answer": "do X [PR #1]", "confidence": "high", "missing": ""},
            context="t",
        )
        assert (parsed.answered, parsed.confidence, parsed.answer) == (True, "high", "do X [PR #1]")

    def test_a_draft_passes_through(self):
        draft = validated(
            DraftedRule,
            {"usable": True, "statement": "Put it next to the assertion.", "scope": "testing"},
            context="t",
        )
        assert draft.is_usable and draft.scope == "testing"

    def test_a_verdict_passes_through(self):
        verdict = validated(
            Verdict, {"relation": "contradicts", "reason": "different place"}, context="t"
        )
        assert verdict.relation == "contradicts"
        assert verdict.reason == "different place"

    def test_a_draft_without_the_usable_flag_is_usable(self):
        # The flag postdates the prompts. Absence must not read as refusal, or
        # every response written before it would fail closed.
        assert validated(DraftedRule, {"statement": "Do X."}, context="t").is_usable is True
