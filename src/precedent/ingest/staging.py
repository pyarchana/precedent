"""Raw-response staging and cursor checkpointing.

Raw GitHub JSON is written to disk *before* anything parses it, so the whole
transform can be replayed after a schema change without re-hitting the API.
Local disk is the staging area; `scripts/sync_raw_to_s3.py` mirrors it to S3.
That ordering is deliberate: it means the ingest does not block on an AWS
account existing.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_write(path: Path, payload: bytes) -> None:
    """Write via a temp file + replace so a crash never leaves a half file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class RawStore:
    """Append-only store of raw API pages, one gzipped JSON object per page."""

    def __init__(self, root: Path, repo_slug: str) -> None:
        self.root = root / repo_slug.replace("/", "__") / "pr_pages"
        self.root.mkdir(parents=True, exist_ok=True)

    def page_path(self, page_number: int) -> Path:
        return self.root / f"page_{page_number:06d}.json.gz"

    def has_page(self, page_number: int) -> bool:
        return self.page_path(page_number).exists()

    def write_page(
        self,
        page_number: int,
        data: dict[str, Any],
        *,
        cursor: str | None,
        variables: dict[str, Any],
    ) -> Path:
        envelope = {
            "schema": "precedent.raw.pr_page/1",
            "page_number": page_number,
            "fetched_at": _utcnow(),
            "request_cursor": cursor,
            "request_variables": variables,
            "data": data,
        }
        blob = gzip.compress(json.dumps(envelope, separators=(",", ":")).encode("utf-8"))
        path = self.page_path(page_number)
        _atomic_write(path, blob)
        return path

    def read_page(self, page_number: int) -> dict[str, Any]:
        with gzip.open(self.page_path(page_number), "rt", encoding="utf-8") as fh:
            return json.load(fh)

    def iter_pages(self):
        for path in sorted(self.root.glob("page_*.json.gz")):
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                yield path, json.load(fh)


@dataclass
class Checkpoint:
    """Resume state. Written after every successful page."""

    repo_slug: str
    cursor: str | None = None
    page_number: int = 0
    prs_seen: int = 0
    completed: bool = False
    started_at: str = field(default_factory=_utcnow)
    updated_at: str = field(default_factory=_utcnow)
    oldest_pr_number: int | None = None
    newest_pr_number: int | None = None
    # PRs whose nested connections were truncated; these need a second pass.
    truncated_prs: list[int] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path, repo_slug: str) -> Checkpoint:
        if not path.exists():
            return cls(repo_slug=repo_slug)
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("repo_slug") != repo_slug:
            raise RuntimeError(
                f"checkpoint at {path} is for {raw.get('repo_slug')}, not {repo_slug}. "
                "Use a different --raw-dir or delete the checkpoint."
            )
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self, path: Path) -> None:
        self.updated_at = _utcnow()
        _atomic_write(path, json.dumps(asdict(self), indent=2).encode("utf-8"))
