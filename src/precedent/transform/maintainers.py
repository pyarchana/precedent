"""Load the derived maintainer list, produced by `scripts/derive_maintainers.py`.

Kept out of `normalize.py` so that module stays free of file access. A missing
file yields an empty set, since a fresh repository has none and classification
by association still works. A file naming a different repository raises, because
applying pandas' maintainers elsewhere corrupts the corpus undetectably.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from precedent.config import REPO_ROOT

log = logging.getLogger(__name__)

# Two layouts. A checkout nests the package under `src/`; the Lambda package
# unpacks it straight into `/var/task/`, where `REPO_ROOT` resolves to `/`.
# Getting this wrong does not raise, it silently drops every former maintainer.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

CANDIDATE_PATHS = (
    REPO_ROOT / "config" / "maintainers.yaml",
    PACKAGE_ROOT.parent / "config" / "maintainers.yaml",
)


def find_maintainer_list() -> Path:
    """The first candidate that exists, or the checkout path for the error message."""
    return next((p for p in CANDIDATE_PATHS if p.is_file()), CANDIDATE_PATHS[0])


def load_maintainers(repo_slug: str, path: Path | None = None) -> frozenset[str]:
    """Lowercased logins to treat as maintainers regardless of association."""
    path = path or find_maintainer_list()
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
