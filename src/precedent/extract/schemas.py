"""Shapes for what models return, and what to do when they return something else.

`complete_json` hands back a plain dict, and reading it with `.get` accepts
anything: `bool({"answered": "false"}.get("answered"))` is True, which turns a
refusal into an answer.

So a malformed payload validates to the **inert** outcome. An answer becomes a
refusal, a drafted rule becomes unusable, a contradiction verdict becomes
"compatible". Each of these decides whether to write to memory or to state
something to a contributor, and a garbled response is not evidence for either.
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
    written to; `scope` and `confidence` only label a result that is otherwise
    fine, so "vibes" instead of "testing" must not discard a good correction.
    """
    if isinstance(value, str) and value.strip().lower() in allowed:
        return value.strip().lower()
    if value is not None:
        log.info("unrecognised label %r; using %r", value, fallback)
    return fallback


class ModelOutput(BaseModel):
    """Base for every model response. A `null` string means empty, not malformed.

    A prompt saying to give a rationale "only if they gave a reason" is asking
    for `null`, Pydantic rejects that for a plain `str`, and validation is
    all-or-nothing. One absent optional field discarded a live maintainer
    correction six times out of six, reported as "no convention stated".
    """

    @field_validator("*", mode="before")
    @classmethod
    def _null_text_is_empty(cls, value, info):
        if value is not None:
            return value
        field = cls.model_fields.get(info.field_name)
        # Plain `str` only. `str | None` means it, and a null scope has to fall
        # through to the label validator.
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

    # Absent means usable. The flag postdates the prompts, so reading its
    # absence as refusal would fail every well-formed response before it.
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

    Never raises. Every default above is chosen so that inert means change
    nothing.
    """
    if not isinstance(payload, dict):
        log.warning("%s: expected a JSON object, got %s", context, type(payload).__name__)
        return model()
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        # With the payload: the useful question is which field it got wrong.
        log.warning(
            "%s: model returned an unusable shape (%s); treating it as inert. payload=%r",
            context,
            exc.errors()[0].get("msg", "invalid") if exc.errors() else "invalid",
            payload,
        )
        return model()
