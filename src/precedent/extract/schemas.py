"""Shapes for what models return, and what to do when they return something else.

A prompt asking for JSON gets JSON almost always, and the almost is the
problem. `complete_json` hands back a plain dict, and every caller then reads
it with `.get`, which quietly accepts anything:

    result = {"answered": "false"}
    bool(result.get("answered"))       # True

That is not a hypothetical. A model that writes `"false"` instead of `false`
turns a refusal into an answer, and refusing is the behaviour this whole
project is built around. The same shape of bug sits in the correction path,
where `result.get("usable") is False` is not satisfied by the string `"false"`,
so a correction the model declined to draft would be written anyway.

## Failing towards doing nothing

Every model here validates to the **inert** outcome when the payload is
malformed:

  * an answer becomes a refusal, not an answer
  * a drafted rule becomes unusable, so nothing is written
  * a contradiction verdict becomes "compatible", so nothing is retired

That direction is deliberate. Each of these decides whether to write to memory
or to state something to a contributor, and a garbled response is not evidence
for either. The cost of failing this way is a request that does nothing; the
cost of failing the other way is memory that quietly contains a claim nobody
made.
"""

from __future__ import annotations

import logging
from typing import Literal, TypeVar, get_args

from pydantic import BaseModel, Field, ValidationError, field_validator

log = logging.getLogger(__name__)

# PEP 695 syntax would read better and is 3.12 only. The project supports 3.11.
T = TypeVar("T", bound=BaseModel)

SCOPES = Literal["repo", "directory", "file", "api", "testing", "docs", "style", "process"]
_SCOPE_VALUES = frozenset(get_args(SCOPES))
_CONFIDENCE_VALUES = frozenset({"high", "medium", "low"})


def _one_of(allowed: frozenset[str], fallback: str, value: object) -> str:
    """Keep a recognised label, replace anything else with a safe one.

    Strictness has to be aimed. `usable` and `relation` decide whether memory is
    written to, so a malformed one invalidates the whole response and the
    caller does nothing. `scope` and `confidence` only label a result that is
    otherwise fine, and discarding a maintainer's correction because the model
    wrote "vibes" instead of "testing" throws away the part that mattered in
    order to protect the part that did not.
    """
    if isinstance(value, str) and value.strip().lower() in allowed:
        return value.strip().lower()
    if value is not None:
        log.info("unrecognised label %r; using %r", value, fallback)
    return fallback


class ModelOutput(BaseModel):
    """Base for every model response, handling the one thing they all get right.

    A prompt that says to include a rationale "only if they gave a reason" is
    asking the model to omit it, and JSON's way of omitting a value is `null`.
    Pydantic rejects `null` for a plain `str`, and because validation is
    all-or-nothing, one absent optional field discarded the entire response.

    That cost a live maintainer correction. `@precedent whatsnew entries go in
    the file for the next release, not the current one` produced exactly the
    payload the prompt asked for, `usable: true` with a clean statement and
    `rationale: null`, and was thrown away six times out of six. The webhook
    answered 200 with "no convention stated", so it looked like the model had
    declined rather than like the schema had refused a correct answer.

    This is the same shape as the `scope: "vibes"` bug: strictness protecting a
    field nobody depends on by discarding the field everything depends on. A
    `null` where a string was expected means empty, not malformed.
    """

    @field_validator("*", mode="before")
    @classmethod
    def _null_text_is_empty(cls, value, info):
        if value is not None:
            return value
        field = cls.model_fields.get(info.field_name)
        # Only plain `str` fields. `scope_pattern: str | None` means it, and a
        # null scope must still fall through to the label validator.
        return "" if field is not None and field.annotation is str else value


class AnswerOutput(ModelOutput):
    """What the answering prompt is asked to return."""

    answered: bool = False
    answer: str = ""
    confidence: Literal["high", "medium", "low"] = "low"
    missing: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def _known_confidence(cls, value):
        return _one_of(_CONFIDENCE_VALUES, "low", value)


class DraftedRule(ModelOutput):
    """A convention drafted from a correction or a maintainer's comment."""

    # Absent means usable. The flag was added after the prompts shipped, and
    # treating its absence as refusal would fail every well-formed response
    # that predates it.
    usable: bool = True
    needed: str = ""
    statement: str = ""
    rationale: str = ""
    scope: SCOPES = "repo"
    scope_pattern: str | None = None

    @field_validator("scope", mode="before")
    @classmethod
    def _known_scope(cls, value):
        return _one_of(_SCOPE_VALUES, "repo", value)

    @property
    def is_usable(self) -> bool:
        return self.usable and bool(self.statement.strip())


class Verdict(ModelOutput):
    """How two rules relate."""

    relation: Literal["same", "contradicts", "compatible"] = "compatible"
    reason: str = ""


class ExtractedRule(ModelOutput):
    """A convention distilled from a cluster of comments."""

    is_convention: bool = False
    statement: str = ""
    rationale: str = ""
    scope: SCOPES = "repo"
    scope_pattern: str | None = None
    reason: str = Field(default="", description="Why not, when is_convention is false")

    @field_validator("scope", mode="before")
    @classmethod
    def _known_scope(cls, value):
        return _one_of(_SCOPE_VALUES, "repo", value)


def validated(model: type[T], payload: object, *, context: str) -> T:
    """Coerce a model response into `model`, or return its inert default.

    Never raises. A malformed response is a thing that happens, not an
    exception the request should die on, and the defaults on every model above
    are chosen so that "inert" means "change nothing".
    """
    if not isinstance(payload, dict):
        log.warning("%s: expected a JSON object, got %s", context, type(payload).__name__)
        return model()
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        # Logged with the payload, because the useful debugging question is
        # always which field the model got wrong rather than that it did.
        log.warning(
            "%s: model returned an unusable shape (%s); treating it as inert. payload=%r",
            context,
            exc.errors()[0].get("msg", "invalid") if exc.errors() else "invalid",
            payload,
        )
        return model()
