"""Scoring how much to trust an extracted rule.

The plan called for weighting by maintainer status. That turned out to be
already spent: the deployed corpus is maintainer comments only, so every rule
scores identically on it. Weighting by something constant is just a scale
factor, and would have looked like signal while carrying none.

What is left is independence and durability, which are the properties that
actually distinguish a convention from a strongly held opinion:

  * **Independent voices.** Four maintainers saying the same thing is a
    convention. One maintainer saying it four times is a preference. This is
    the heaviest term.
  * **Separate occasions.** Distinct pull requests, because several comments
    inside one review are a single conversation.
  * **Persistence.** Guidance repeated across years has survived turnover and
    argument. Guidance confined to one month may be a passing concern.
  * **Recency.** A rule last mentioned in 2015 may since have been reversed.
    Copy-on-Write alone silently invalidated a great deal of older advice.

The weights are a judgement, not a measurement, and they are declared in one
place so the audit on plan day 12 can argue with them.
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

# A maintainer correcting a specific wrong answer is not weak evidence. It is
# the strongest evidence this system can get: stated deliberately, in context,
# by someone with authority over the answer.
#
# It also has none of the properties measured above, so measuring it produces a
# number that is not merely imprecise but backwards. Corrections are therefore
# floored rather than scored. The floor sits below 1.0 because a correction is
# still one person on one occasion, and a convention restated by five
# maintainers across four years should still be able to outrank it.
STATED_DIRECTLY_FLOOR = 0.85


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
