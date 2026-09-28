"""Request instrumentation: JSON logs, per-stage timings, outcome counters.

`configure_logging()` installs the formatter, `request()` wraps a handler and
emits one JSON line per request, `snapshot()` backs `/metrics`.

Counters are per Lambda execution environment and lost when it recycles, so
`/metrics` is not a global total. Durable spend lives in `api_usage`.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("precedent.request")

# Kept per route so one busy endpoint cannot evict another's history.
SAMPLE_SIZE = 200

_started = time.time()
_counts: Counter[str] = Counter()
_samples: dict[str, deque[float]] = {}


class JsonFormatter(logging.Formatter):
    """Everything a record carries, on one line, with `extra` merged in."""

    _BUILTIN = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
        "message",
        "asctime",
        "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update({k: v for k, v in record.__dict__.items() if k not in self._BUILTIN})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        try:
            return json.dumps(payload, default=str)
        except Exception:  # noqa: BLE001 - a log line must never raise
            return json.dumps(
                {"ts": payload["ts"], "level": payload["level"], "msg": payload["msg"]}
            )


def configure_logging(level: int = logging.INFO) -> None:
    """JSON under Lambda or when asked, plain text otherwise.

    Idempotent, because Lambda may import this more than once per container.
    """
    as_json = os.environ.get("LOG_FORMAT", "").lower() == "json" or bool(
        os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
    )
    root = logging.getLogger()
    for existing in root.handlers:
        if getattr(existing, "_precedent", False):
            return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter() if as_json else logging.Formatter("%(levelname)s %(name)s: %(message)s")
    )
    handler._precedent = True  # type: ignore[attr-defined]
    root.handlers = [h for h in root.handlers if not getattr(h, "_precedent", False)] + [handler]
    root.setLevel(level)


@dataclass
class Request:
    """Timings and fields for one request, emitted as a single line at the end."""

    route: str
    stages: dict[str, float] = field(default_factory=dict)
    fields: dict[str, Any] = field(default_factory=dict)
    result: str = "ok"

    @contextmanager
    def stage(self, name: str):
        """Time a stage. A failure is still timed, and still re-raised."""
        began = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = round((time.perf_counter() - began) * 1000, 1)

    def note(self, **fields: Any) -> None:
        self.fields.update(fields)

    def outcome(self, name: str) -> None:
        """Why this request ended as it did. Distinguishes refusal from failure."""
        self.result = name


@contextmanager
def request(route: str, **fields: Any):
    """Wrap a request: time it, name its outcome, emit one line, count it."""
    rec = Request(route=route)
    rec.note(**fields)
    began = time.perf_counter()
    try:
        yield rec
    except Exception as exc:
        rec.result = f"error:{type(exc).__name__}"
        raise
    finally:
        total = round((time.perf_counter() - began) * 1000, 1)
        _counts[f"{route}:{rec.result}"] += 1
        _samples.setdefault(route, deque(maxlen=SAMPLE_SIZE)).append(total)
        log.info(
            "%s %s %.0fms",
            route,
            rec.result,
            total,
            extra={
                "route": route,
                "outcome": rec.result,
                "total_ms": total,
                "stages": rec.stages,
                **rec.fields,
            },
        )


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def snapshot() -> dict[str, Any]:
    """What this process has seen. Not a global total; see the module docstring."""
    latency = {}
    for route, seen in _samples.items():
        values = list(seen)
        latency[route] = {
            "n": len(values),
            "p50_ms": _percentile(values, 0.50),
            "p95_ms": _percentile(values, 0.95),
            "max_ms": max(values) if values else 0.0,
        }
    outcomes: dict[str, dict[str, int]] = {}
    for key, n in _counts.items():
        route, _, result = key.partition(":")
        outcomes.setdefault(route, {})[result] = n
    return {
        "scope": "this execution environment only, reset when it is recycled",
        "uptime_s": round(time.time() - _started, 1),
        "outcomes": outcomes,
        "latency": latency,
    }


def reset() -> None:
    """For tests."""
    _counts.clear()
    _samples.clear()
