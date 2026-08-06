"""Turn a raw GitHub PR page into review_comments rows.

Pure functions with no database and no network, so the rules about what counts
as a comment and who counts as a maintainer can be tested directly. Everything
that talks to CockroachDB lives in `load.py`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any

# GitHub's authorAssociation values that mean "this person speaks for the project".
# CONTRIBUTOR means "has had a PR merged", which is not the same thing and is
# deliberately excluded: the whole point of confidence weighting is that a
# maintainer saying something counts for more.
MAINTAINER_ASSOCIATIONS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})

# Bots that comment on pandas PRs. Matching is exact on lowercased login, in
# addition to the generic "[bot]" suffix and GitHub's own Bot typename.
KNOWN_BOTS = frozenset(
    {
        "codecov-io",
        "codecov-commenter",
        "meeseeksdev",
        "meeseeksmachine",
        "pep8speaks",
        "pandas-dev-bot",
        "stale",
        "sourcery-ai",
        "coveralls",
    }
)


# Lines from the pull request template. A description that is mostly these is
# a filled-in form, not a maintainer teaching anyone anything.
#
# This was found by auditing extracted rules. 2,585 comments, one per pull
# request and about 4% of the corpus, are this template. Six of the top fifty
# rules were built on it, producing statements like "always include the GitHub
# issue number in the pull request description" whose supporting evidence is a
# checklist rather than anything a reviewer said. Filtering at the cluster
# level did not catch it: in a cluster of eighty items the template mixes with
# real comments and the distances no longer look degenerate.
TEMPLATE_MARKERS = (
    "replace xxxx with the",
    "tests added and passed if fixing a bug",
    "all code checks passed",
    "added type annotations to new arguments",
    "added an entry in the latest doc/source/whatsnew",
)

# One marker can appear in a genuine comment quoting the template. Two or more
# means the body is the form itself.
TEMPLATE_THRESHOLD = 2


class RejectReason:
    BOT_AUTHOR = "bot_author"
    EMPTY_BODY = "empty_body"
    NO_AUTHOR = "no_author"
    PR_TEMPLATE = "pr_template"


def is_pr_template(body: str) -> bool:
    """True when the body is a filled-in pull request template."""
    lowered = body.lower()
    return sum(marker in lowered for marker in TEMPLATE_MARKERS) >= TEMPLATE_THRESHOLD


@dataclass(slots=True)
class CommentRecord:
    github_node_id: str
    kind: str
    pr_number: int
    body: str
    created_at: datetime
    author: str | None
    author_association: str
    is_maintainer: bool
    thread_id: str | None = None
    in_reply_to: str | None = None
    file_path: str | None = None
    line: int | None = None
    diff_hunk: str | None = None
    url: str | None = None


@dataclass(slots=True)
class RejectRecord:
    github_node_id: str
    reason: str
    pr_number: int | None
    author: str | None


def is_bot(author: dict[str, Any] | None) -> bool:
    if author is None:
        return False
    if author.get("__typename") == "Bot":
        return True
    login = (author.get("login") or "").lower()
    return login.endswith("[bot]") or login in KNOWN_BOTS


def is_maintainer(association: str | None) -> bool:
    return (association or "").upper() in MAINTAINER_ASSOCIATIONS


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _author_login(author: dict[str, Any] | None) -> str | None:
    return author.get("login") if author else None


def normalize_page(page: dict[str, Any]) -> tuple[list[CommentRecord], list[RejectRecord]]:
    """Flatten one staged page into comment rows plus a record of what was dropped."""
    kept: list[CommentRecord] = []
    dropped: list[RejectRecord] = []

    for pr in page["repository"]["pullRequests"]["nodes"]:
        if pr is None:
            continue
        for record in _pr_records(pr):
            if isinstance(record, RejectRecord):
                dropped.append(record)
            else:
                kept.append(record)

    return kept, dropped


def _emit(
    node: dict[str, Any],
    *,
    node_id: str,
    kind: str,
    pr_number: int,
    body: str | None,
    created_at: str | None,
    **extra: Any,
) -> CommentRecord | RejectRecord:
    author = node.get("author")
    login = _author_login(author)

    if is_bot(author):
        return RejectRecord(node_id, RejectReason.BOT_AUTHOR, pr_number, login)

    text = (body or "").strip()
    if not text:
        return RejectRecord(node_id, RejectReason.EMPTY_BODY, pr_number, login)

    if is_pr_template(text):
        return RejectRecord(node_id, RejectReason.PR_TEMPLATE, pr_number, login)

    association = node.get("authorAssociation") or "NONE"
    return CommentRecord(
        github_node_id=node_id,
        kind=kind,
        pr_number=pr_number,
        body=text,
        created_at=_parse_ts(created_at),
        author=login,
        author_association=association,
        is_maintainer=is_maintainer(association),
        url=node.get("url"),
        **extra,
    )


def _pr_records(pr: dict[str, Any]) -> Iterator[CommentRecord | RejectRecord]:
    number = pr["number"]

    # The PR description. The query does not select the pull request's own node
    # id, so this one is synthesised. It is deterministic, which is all the
    # upsert needs.
    yield _emit(
        pr,
        node_id=f"pr:{number}:body",
        kind="pr_body",
        pr_number=number,
        body=pr.get("bodyText"),
        created_at=pr.get("createdAt"),
    )

    for review in (pr.get("reviews") or {}).get("nodes") or []:
        if review is None:
            continue
        # Approvals with no text carry no knowledge; they fall out as empty_body.
        yield _emit(
            review,
            node_id=review["id"],
            kind="review_summary",
            pr_number=number,
            body=review.get("bodyText"),
            created_at=review.get("submittedAt") or pr.get("createdAt"),
        )

    for thread in (pr.get("reviewThreads") or {}).get("nodes") or []:
        if thread is None:
            continue
        for comment in (thread.get("comments") or {}).get("nodes") or []:
            if comment is None:
                continue
            yield _emit(
                comment,
                node_id=comment["id"],
                kind="review_thread",
                pr_number=number,
                body=comment.get("bodyText"),
                created_at=comment.get("createdAt"),
                thread_id=thread.get("id"),
                in_reply_to=(comment.get("replyTo") or {}).get("id"),
                file_path=comment.get("path") or thread.get("path"),
                line=comment.get("originalLine") or thread.get("line"),
                diff_hunk=comment.get("diffHunk"),
            )

    for comment in (pr.get("comments") or {}).get("nodes") or []:
        if comment is None:
            continue
        yield _emit(
            comment,
            node_id=comment["id"],
            kind="issue_comment",
            pr_number=number,
            body=comment.get("bodyText"),
            created_at=comment.get("createdAt"),
        )
