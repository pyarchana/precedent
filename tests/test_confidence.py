"""Tests for rule confidence scoring.

These pin down the ordering the score is meant to express, rather than exact
numbers, because the weights are a judgement that the day 12 audit may revise.
What must not change is which rules outrank which.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from precedent.extract.confidence import score

NOW = datetime(2026, 8, 3, tzinfo=UTC)


def years_ago(n: float) -> datetime:
    return NOW - timedelta(days=n * 365.25)


class TestBounds:
    def test_no_evidence_scores_zero(self):
        assert score(distinct_authors=0, distinct_prs=0, now=NOW).confidence == 0.0

    def test_maximal_evidence_approaches_one(self):
        s = score(
            distinct_authors=9,
            distinct_prs=20,
            first_evidence_at=years_ago(3),
            last_evidence_at=NOW,
            now=NOW,
        )
        assert s.confidence == 1.0

    def test_always_within_range(self):
        for authors, prs in ((0, 0), (1, 1), (100, 100), (-5, -5)):
            s = score(
                distinct_authors=authors,
                distinct_prs=prs,
                first_evidence_at=years_ago(1),
                last_evidence_at=NOW,
                now=NOW,
            )
            assert 0.0 <= s.confidence <= 1.0


class TestIndependence:
    def test_many_voices_beat_one_voice_repeating(self):
        # The distinction the whole score exists to make: four maintainers
        # agreeing is a convention, one maintainer four times is a preference.
        many = score(
            distinct_authors=4,
            distinct_prs=4,
            first_evidence_at=years_ago(2),
            last_evidence_at=NOW,
            now=NOW,
        )
        one = score(
            distinct_authors=1,
            distinct_prs=4,
            first_evidence_at=years_ago(2),
            last_evidence_at=NOW,
            now=NOW,
        )
        assert many.confidence > one.confidence

    def test_independence_outweighs_repetition(self):
        # Five authors across three PRs should beat one author across eight.
        broad = score(distinct_authors=5, distinct_prs=3, now=NOW)
        deep = score(distinct_authors=1, distinct_prs=8, now=NOW)
        assert broad.confidence > deep.confidence


class TestPersistence:
    def test_guidance_repeated_over_years_beats_a_single_month(self):
        durable = score(
            distinct_authors=3,
            distinct_prs=5,
            first_evidence_at=years_ago(3),
            last_evidence_at=NOW,
            now=NOW,
        )
        brief = score(
            distinct_authors=3,
            distinct_prs=5,
            first_evidence_at=NOW - timedelta(days=20),
            last_evidence_at=NOW,
            now=NOW,
        )
        assert durable.confidence > brief.confidence

    def test_missing_dates_do_not_crash(self):
        s = score(distinct_authors=3, distinct_prs=5, now=NOW)
        assert s.persistence == 0.0
        assert s.recency == 0.0


class TestRecency:
    def test_stale_evidence_scores_lower(self):
        # A rule last mentioned a decade ago may since have been reversed.
        fresh = score(
            distinct_authors=3,
            distinct_prs=5,
            first_evidence_at=years_ago(3),
            last_evidence_at=NOW,
            now=NOW,
        )
        stale = score(
            distinct_authors=3,
            distinct_prs=5,
            first_evidence_at=years_ago(13),
            last_evidence_at=years_ago(10),
            now=NOW,
        )
        assert fresh.confidence > stale.confidence

    def test_decays_by_half_over_the_half_life(self):
        s = score(
            distinct_authors=0,
            distinct_prs=0,
            last_evidence_at=years_ago(4),
            now=NOW,
        )
        assert abs(s.recency - 0.5) < 0.01


class TestExplanation:
    def test_breakdown_is_readable(self):
        s = score(
            distinct_authors=4,
            distinct_prs=6,
            first_evidence_at=years_ago(2),
            last_evidence_at=NOW,
            now=NOW,
        )
        text = s.explain()
        assert "independence" in text and "recency" in text
