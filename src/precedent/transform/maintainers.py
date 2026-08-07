"""Load the derived maintainer list.

Kept out of `normalize.py` so that module stays free of file access and remains
testable as pure functions. The list itself is produced by
`scripts/derive_maintainers.py` and checked in at `config/maintainers.yaml`,
with the counts that justified each entry, so a reader can disagree with it.

Loading is deliberately strict about one thing and forgiving about another. A
missing file is fine and yields an empty set, because a fresh repository has no
list yet and classification by association still works. A file that exists but
names a different repository is an error, because silently applying pandas'
maintainers to another project would corrupt the corpus in a way nothing
downstream could detect.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from precedent.config import REPO_ROOT

log = logging.getLogger(__name__)

DEFAULT_PATH = REPO_ROOT / "config" / "maintainers.yaml"


def load_maintainers(repo_slug: str, path: Path | None = None) -> frozenset[str]:
    """Lowercased logins to treat as maintainers regardless of association."""
    path = path or DEFAULT_PATH
    if not path.is_file():
        log.warning(
            "no maintainer list at %s; falling back to authorAssociation alone, "
            "which misses anyone who has left the project",
            path,
        )
        return frozenset()

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    listed = data.get("repo")
    if listed and listed != repo_slug:
        raise ValueError(
            f"{path} holds maintainers for {listed}, not {repo_slug}. "
            "Run scripts/derive_maintainers.py for this repository."
        )

    logins = frozenset(
        entry["login"].lower() for entry in data.get("maintainers") or [] if entry.get("login")
    )
    log.info("loaded %d known maintainers for %s", len(logins), repo_slug)
    return logins
