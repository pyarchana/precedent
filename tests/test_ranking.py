"""Tests for letting a maintainer's statement in ahead of a closer inference.

The central claim failed a live test. A maintainer taught the project "place
whatsnew entries in the current release file, not the next one", and asking
"which whatsnew file should my bug fix note go in?" answered from five inferred
rules without mentioning it. The correction sat at distance 1.038, inside the
threshold but thirteenth, because rules distilled from patterns phrase the topic
more like the question does.

A correction that does not surface is the one thing this system promises cannot
happen, so distance is no longer allowed to settle that tie alone.
"""

from __future__ import annotations

from precedent.agent.retrieve import MAX_PROMOTED_STATED, rank_rules


def row(rule_id: str, distance: float, origin: str = "extracted") -> dict:
    return {"id": rule_id, "distance": distance, "origin": origin}


class TestOrdinaryRetrievalIsUnchanged:
    def test_with_no_stated_rules_it_is_the_nearest_k(self):
        rows = [row(str(i), 0.5 + i / 100) for i in range(10)]
        assert [r["id"] for r in rank_rules(rows, 5)] == ["0", "1", "2", "3", "4"]

    def test_it_never_returns_more_than_asked_for(self):
        rows = [row(str(i), 0.5 + i / 100, "taught") for i in range(10)]
        assert len(rank_rules(rows, 5)) == 5

    def test_fewer_rules_than_asked_for_is_fine(self):
        assert len(rank_rules([row("a", 0.5)], 5)) == 1

    def test_no_rules_at_all_is_fine(self):
        assert rank_rules([], 5) == []


class TestPromotion:
    def test_a_correction_outside_the_top_k_is_let_in(self):
        rows = [row(str(i), 0.7 + i / 100) for i in range(12)]
        rows.append(row("correction", 1.038, "correction"))
        chosen = [r["id"] for r in rank_rules(rows, 5)]
        assert "correction" in chosen

    def test_a_taught_rule_is_let_in_too(self):
        # Stating a convention and correcting an answer are different
        # operations with different origins, and both are things a maintainer
        # actually said.
        rows = [row(str(i), 0.7 + i / 100) for i in range(12)]
        rows.append(row("taught", 1.038, "taught"))
        assert "taught" in [r["id"] for r in rank_rules(rows, 5)]

    def test_it_displaces_the_furthest_inference_not_the_nearest(self):
        rows = [row(str(i), 0.7 + i / 100) for i in range(12)]
        rows.append(row("taught", 1.2, "taught"))
        chosen = [r["id"] for r in rank_rules(rows, 5)]
        assert "0" in chosen and "1" in chosen
        assert "4" not in chosen

    def test_the_result_is_still_ordered_by_distance(self):
        # The prompt reads nearest first, so promotion must not scramble it.
        rows = [row(str(i), 0.7 + i / 100) for i in range(12)]
        rows.append(row("taught", 1.038, "taught"))
        distances = [r["distance"] for r in rank_rules(rows, 5)]
        assert distances == sorted(distances)

    def test_corrections_cannot_crowd_out_the_matching_rules(self):
        # A repository with many corrections must not answer every question
        # from corrections. They take at most MAX_PROMOTED_STATED slots.
        rows = [row(f"s{i}", 1.10 + i / 100, "correction") for i in range(6)]
        rows += [row(f"e{i}", 0.70 + i / 100) for i in range(6)]
        chosen = rank_rules(rows, 5)
        assert sum(1 for r in chosen if r["origin"] == "correction") == MAX_PROMOTED_STATED

    def test_a_stated_rule_already_in_range_is_not_duplicated(self):
        rows = [row("taught", 0.5, "taught")] + [row(str(i), 0.6 + i / 100) for i in range(9)]
        chosen = rank_rules(rows, 5)
        assert len(chosen) == len({r["id"] for r in chosen}) == 5
