"""Conversion between Python float lists and CockroachDB's VECTOR wire format.

The driver hands VECTOR columns back as text and expects text going in, so
every read and write has to cross this boundary. Keeping it in one place means
the rest of the codebase can deal in lists of floats.
"""

from __future__ import annotations

from collections.abc import Sequence


def encode(values: Sequence[float]) -> str:
    """Render a vector as the `[1,2,3]` literal CockroachDB parses."""
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


def decode(raw: str | None) -> list[float] | None:
    """Parse a vector as returned by the driver."""
    if raw is None:
        return None
    inner = str(raw).strip()
    if inner.startswith("[") and inner.endswith("]"):
        inner = inner[1:-1]
    if not inner:
        return []
    return [float(part) for part in inner.split(",")]
