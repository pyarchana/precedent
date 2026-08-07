"""Work out who actually reviews this repository, rather than who still can.

GitHub's `authorAssociation` is computed from **current** repository permissions
at the moment you query it, not from the permissions the author held when they
wrote the comment. Anyone who has since stepped back from a project has their
entire history reported as CONTRIBUTOR.

On pandas this is not a rounding error. `jreback` wrote 74,077 comments between
2012 and 2025, more than twice anyone else, and every one of them comes back as
CONTRIBUTOR because he is no longer a member of the organisation. Twelve people
with over a thousand comments each are misreported the same way, 102,454
comments between them, about a third of the corpus. Filtering to maintainers on
that field silently discards the most experienced reviewers a project ever had,
which is the exact opposite of what the filter is for.

So maintainer status is derived from behaviour instead. The signal used here is
**formal review decisions on other people's pull requests**: submitting a review
with CHANGES_REQUESTED or APPROVED is the act of gating someone else's work.

Measured against pandas, the association field alone finds 32 maintainers
holding 47.8% of the review comments. Adding the behavioural rule finds 39
holding 76.5%. Seven people account for that 29-point gap.

There is no clean gap in the distribution to site a threshold in, and pretending
otherwise would be the kind of unsupported precision this project is meant to
avoid. The defence of 50 is not that it sits in a valley but that the answer is
almost unaffected by it: from 20 to 120, coverage moves only between 77.8% and
74.0%, because comment volume is so skewed towards people no threshold in that
range would exclude. `--sensitivity` prints that table.

The numbers behind each entry are written out with the list, so it can be argued
with rather than trusted.

    python scripts/derive_maintainers.py --report
    python scripts/derive_maintainers.py --sensitivity
    python scripts/derive_maintainers.py --write
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from precedent.config import REPO_ROOT, get_settings
from precedent.transform.normalize import MAINTAINER_ASSOCIATIONS, is_bot

log = logging.getLogger("precedent.maintainers")

DEFAULT_OUTPUT = REPO_ROOT / "config" / "maintainers.yaml"

# Review states that gate someone else's work. COMMENTED is excluded: it is
# what anyone leaves when passing by, and counting it would readmit exactly the
# drive-by contributors this is meant to distinguish maintainers from.
GATING_STATES = {"CHANGES_REQUESTED", "APPROVED"}

# Set from the distribution this script prints. See the report: gating reviews
# per person fall away sharply, and everyone above this line is someone the
# project trusted to accept or block a change, repeatedly, for years.
MIN_GATED_PRS = 50
MIN_TENURE_DAYS = 365


@dataclass
class Activity:
    login: str
    associations: set[str] = field(default_factory=set)
    comments: int = 0
    gated_prs: set[int] = field(default_factory=set)
    approvals: int = 0
    changes_requested: int = 0
    first_at: str | None = None
    last_at: str | None = None

    def saw(self, when: str | None) -> None:
        if not when:
            return
        if self.first_at is None or when < self.first_at:
            self.first_at = when
        if self.last_at is None or when > self.last_at:
            self.last_at = when

    @property
    def tenure_days(self) -> int:
        if not (self.first_at and self.last_at):
            return 0
        try:
            # GitHub returns Zulu times, which fromisoformat handles from 3.11
            # and which stay timezone-aware so the subtraction is meaningful.
            return (
                datetime.fromisoformat(self.last_at) - datetime.fromisoformat(self.first_at)
            ).days
        except ValueError:
            return 0

    @property
    def by_association(self) -> bool:
        """Still holds a role GitHub reports today."""
        return bool(self.associations & MAINTAINER_ASSOCIATIONS)

    @property
    def by_behaviour(self) -> bool:
        return len(self.gated_prs) >= MIN_GATED_PRS and self.tenure_days >= MIN_TENURE_DAYS

    @property
    def is_maintainer(self) -> bool:
        return self.by_association or self.by_behaviour


def _login(node) -> str | None:
    author = (node or {}).get("author")
    if not author or is_bot(author):
        return None
    return author.get("login")


def scan(pages_dir: Path) -> dict[str, Activity]:
    people: dict[str, Activity] = defaultdict(lambda: Activity(login=""))
    files = sorted(pages_dir.glob("*.json.gz"))
    log.info("scanning %d staged pages", len(files))

    for n, path in enumerate(files, 1):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)

        # Pages are stored with an envelope around the GraphQL response.
        root = payload.get("data") or payload
        nodes = ((root.get("repository") or {}).get("pullRequests") or {}).get("nodes") or []

        for pr in nodes:
            if not pr:
                continue
            pr_number = pr.get("number")
            pr_author = _login(pr)

            def touch(login: str, when: str | None, assoc: str | None) -> None:
                who = people[login]
                who.login = login
                who.comments += 1
                if assoc:
                    who.associations.add(assoc.upper())
                who.saw(when)

            if pr_author:
                touch(pr_author, pr.get("createdAt"), pr.get("authorAssociation"))

            for review in (pr.get("reviews") or {}).get("nodes") or []:
                login = _login(review)
                if not login:
                    continue
                touch(login, review.get("submittedAt"), review.get("authorAssociation"))
                # Reviewing your own pull request is not gatekeeping.
                if (
                    login != pr_author
                    and (review.get("state") or "").upper() in GATING_STATES
                    and pr_number is not None
                ):
                    people[login].gated_prs.add(pr_number)
                    if (review.get("state") or "").upper() == "APPROVED":
                        people[login].approvals += 1
                    else:
                        people[login].changes_requested += 1

            for thread in (pr.get("reviewThreads") or {}).get("nodes") or []:
                for comment in (thread.get("comments") or {}).get("nodes") or []:
                    login = _login(comment)
                    if login:
                        touch(login, comment.get("createdAt"), comment.get("authorAssociation"))

            for comment in (pr.get("comments") or {}).get("nodes") or []:
                login = _login(comment)
                if login:
                    touch(login, comment.get("createdAt"), comment.get("authorAssociation"))

        if n % 500 == 0:
            log.info("%d/%d pages, %d people so far", n, len(files), len(people))

    return dict(people)


def report(people: dict[str, Activity]) -> None:
    ranked = sorted(people.values(), key=lambda a: len(a.gated_prs), reverse=True)

    print(f"{'login':<24}{'gated':>7}{'appr':>7}{'req':>6}{'comments':>10}{'yrs':>6}  flags")
    print("-" * 78)
    for a in ranked[:45]:
        flags = []
        if a.by_association:
            flags.append("association")
        if a.by_behaviour:
            flags.append("behaviour")
        print(
            f"{a.login:<24}{len(a.gated_prs):>7}{a.approvals:>7}{a.changes_requested:>6}"
            f"{a.comments:>10}{a.tenure_days / 365.25:>6.1f}  {', '.join(flags) or '-'}"
        )

    recovered = [a for a in people.values() if a.by_behaviour and not a.by_association]
    print()
    print(f"{sum(1 for a in people.values() if a.is_maintainer)} maintainers in total")
    print(
        f"{len(recovered)} recovered by behaviour that the association field misses, "
        f"holding {sum(a.comments for a in recovered):,} comments"
    )
    for a in sorted(recovered, key=lambda a: a.comments, reverse=True)[:20]:
        print(
            f"  {a.login:<24}{a.comments:>8,} comments, {len(a.gated_prs):>5} gated PRs, "
            f"associations seen: {sorted(a.associations) or ['none']}"
        )


def sensitivity(people: dict[str, Activity]) -> None:
    """How much the threshold actually changes the corpus.

    Worth printing rather than asserting. There is no clean gap in gated-review
    counts to put a threshold in, so the honest defence of the number is not
    that it sits in a valley but that the answer barely moves across a wide
    range of it.
    """
    total = sum(a.comments for a in people.values())
    print(f"\ncorpus comments across all authors: {total:,}\n")

    # What the association field alone gives, which is what the corpus was
    # built on. This is the number the whole exercise exists to move.
    assoc = [a for a in people.values() if a.by_association]
    assoc_held = sum(a.comments for a in assoc)
    print(f"{'min_gated':>10}{'maintainers':>13}{'comments':>13}{'% of corpus':>13}")
    print("-" * 49)
    print(f"{'assoc only':>10}{len(assoc):>13}{assoc_held:>13,}{100 * assoc_held / total:>12.1f}%")
    for t in (20, 25, 35, 50, 80, 120):
        chosen = [
            a
            for a in people.values()
            if a.by_association or (len(a.gated_prs) >= t and a.tenure_days >= MIN_TENURE_DAYS)
        ]
        held = sum(a.comments for a in chosen)
        print(f"{t:>10}{len(chosen):>13}{held:>13,}{100 * held / total:>12.1f}%")

    for t in (25, 35):
        moved = sorted(
            a.login
            for a in people.values()
            if not a.by_association
            and a.tenure_days >= MIN_TENURE_DAYS
            and t <= len(a.gated_prs) < MIN_GATED_PRS
        )
        print(f"\nlowering {MIN_GATED_PRS} to {t} would add: {moved or 'nobody'}")


def write(people: dict[str, Activity], repo: str, path: Path) -> None:
    entries = []
    for a in sorted(people.values(), key=lambda a: a.comments, reverse=True):
        if not a.is_maintainer:
            continue
        entries.append(
            {
                "login": a.login,
                "basis": "association" if a.by_association else "behaviour",
                "gated_prs": len(a.gated_prs),
                "approvals": a.approvals,
                "changes_requested": a.changes_requested,
                "comments": a.comments,
                "first_seen": (a.first_at or "")[:10],
                "last_seen": (a.last_at or "")[:10],
                "associations_reported": sorted(a.associations),
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Derived by scripts/derive_maintainers.py. Do not hand-edit counts.\n"
        "#\n"
        "# GitHub reports authorAssociation from current permissions, so anyone who\n"
        "# has stepped back from the project reads as CONTRIBUTOR across their whole\n"
        "# history. Entries with basis 'behaviour' are the ones that field misses.\n"
        + yaml.safe_dump(
            {
                "repo": repo,
                "thresholds": {
                    "min_gated_prs": MIN_GATED_PRS,
                    "min_tenure_days": MIN_TENURE_DAYS,
                },
                "maintainers": entries,
            },
            sort_keys=False,
            width=100,
        ),
        encoding="utf-8",
    )
    log.info("wrote %d maintainers to %s", len(entries), path)


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Derive maintainer status from behaviour.")
    parser.add_argument("--repo", default=settings.repo_slug)
    parser.add_argument("--pages", default=None, help="Directory of staged pages.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", action="store_true", help="Print the distribution.")
    parser.add_argument(
        "--sensitivity", action="store_true", help="Show how much the threshold matters."
    )
    parser.add_argument("--write", action="store_true", help="Write the maintainer list.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s", stream=sys.stdout
    )

    pages = (
        Path(args.pages)
        if args.pages
        else (REPO_ROOT / "data" / "raw" / args.repo.replace("/", "__") / "pr_pages")
    )
    if not pages.is_dir():
        parser.error(f"no staged pages at {pages}")

    people = scan(pages)
    if args.report or not (args.write or args.sensitivity):
        report(people)
    if args.sensitivity:
        sensitivity(people)
    if args.write:
        write(people, args.repo, Path(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
