"""Scoring how much to trust an extracted rule.

Not by maintainer status: the corpus is maintainer comments only, so weighting
by it is a scale factor that looks like signal and carries none. What separates
a convention from a strongly held opinion is independence and durability.

  * **Independent voices**, the heaviest term. Four maintainers saying a thing
    is a convention; one maintainer saying it four times is a preference.
  * **Separate occasions**, by distinct pull request, since several comments in
    one review are one conversation.
  * **Persistence.** Guidance repeated across years has survived turnover.
  * **Recency.** Copy-on-Write alone silently invalidated a lot of older advice.

The weights below are a judgement, not a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

# Saturation points. Beyond these, more evidence does not increase belief:
# the difference between five maintainers and eight is not meaningful.
AUTHORS_SATURATE_AT = 5
PRS_SATURATE_AT = 8

# A convention restated over two years is as durable as one restated over ten.
PERSISTENCE_SATURATE_YEARS = 2.0

# Evidence older than this is treated as possibly stale rather than wrong.
RECENCY_HALF_LIFE_YEARS = 4.0

WEIGHTS = {
    "independence": 0.40,
    "repetition": 0.30,
    "persistence": 0.15,
    "recency": 0.15,
}

# A correction has none of the properties measured above, so scoring it on them
# gives a number that is backwards rather than merely imprecise. Floored, not
# scored. Below 1.0 so a convention five maintainers held for four years wins.
STATED_DIRECTLY_FLOOR = 0.85

# Origins where a maintainer said it themselves rather than it being inferred.
# A correction retires a wrong answer, a teaching adds to memory; same authority.
STATED_ORIGINS = frozenset({"correction", "taught"})


@dataclass(slots=True)
class ConfidenceBreakdown:
    """The score with its parts, so a low score can be explained rather than argued with."""

    confidence: float
    independence: float
    repetition: float
    persistence: float
    recency: float

    def explain(self) -> str:
        return (
            f"{self.confidence:.2f} "
            f"(independence {self.independence:.2f}, repetition {self.repetition:.2f}, "
            f"persistence {self.persistence:.2f}, recency {self.recency:.2f})"
        )


def _years_between(earlier: datetime, later: datetime) -> float:
    return max(0.0, (later - earlier).total_seconds() / (365.25 * 24 * 3600))


def score(
    *,
    distinct_authors: int,
    distinct_prs: int,
    first_evidence_at: datetime | None = None,
    last_evidence_at: datetime | None = None,
    now: datetime | None = None,
    stated_directly: bool = False,
) -> ConfidenceBreakdown:
    """Confidence in [0, 1] for a rule, from the shape of its evidence.

    `stated_directly` marks a rule a maintainer asserted rather than one
    inferred from a pattern, and applies a floor instead of a score.
    """
    now = now or datetime.now(UTC)

    # A single voice is not zero confidence, but it is close: the rule may be
    # true and simply under-witnessed.
    independence = min(max(distinct_authors, 0), AUTHORS_SATURATE_AT) / AUTHORS_SATURATE_AT
    repetition = min(max(distinct_prs, 0), PRS_SATURATE_AT) / PRS_SATURATE_AT

    if first_evidence_at and last_evidence_at:
        span = _years_between(first_evidence_at, last_evidence_at)
        persistence = min(span, PERSISTENCE_SATURATE_YEARS) / PERSISTENCE_SATURATE_YEARS
    else:
        persistence = 0.0

    if last_evidence_at:
        age = _years_between(last_evidence_at, now)
        recency = 0.5 ** (age / RECENCY_HALF_LIFE_YEARS)
    else:
        recency = 0.0

    confidence = (
        WEIGHTS["independence"] * independence
        + WEIGHTS["repetition"] * repetition
        + WEIGHTS["persistence"] * persistence
        + WEIGHTS["recency"] * recency
    )

    if stated_directly:
        confidence = max(confidence, STATED_DIRECTLY_FLOOR)

    return ConfidenceBreakdown(
        confidence=round(min(1.0, max(0.0, confidence)), 4),
        independence=round(independence, 4),
        repetition=round(repetition, 4),
        persistence=round(persistence, 4),
        recency=round(recency, 4),
    )
