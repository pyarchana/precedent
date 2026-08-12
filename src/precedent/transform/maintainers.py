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

# Two layouts, because the file is read in both and they nest differently.
#
# In a checkout the package sits at `src/precedent/`, so the repository root is
# two levels above it and the list is at `<root>/config/maintainers.yaml`.
#
# In the Lambda package there is no `src/`: the package is unpacked directly
# into `/var/task/precedent/`, and `REPO_ROOT`, computed as `parents[2]` of
# `config.py`, resolves to `/`. So the deployed list is looked for beside the
# package instead. Getting this wrong does not raise; it returns an empty set
# and quietly drops every maintainer who has left the project.
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
