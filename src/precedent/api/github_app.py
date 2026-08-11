"""Learn from what maintainers say on pull requests, where they already are.

The web interface asks a maintainer to open a separate site, click a role
toggle, and type their own name into a box. Nobody with fifty reviews a week is
going to do that, and the name they type is unverified anyway, so the resulting
citation says "correction by jbrockmendel" on nothing more than someone's word.

A GitHub App removes both problems at once, and it removes them rather than
patching them.

## Identity comes from GitHub, not from a login

Webhook deliveries are signed with a shared secret using HMAC SHA-256. Verifying
that signature proves the payload came from GitHub and has not been altered, so
`sender.login` is trustworthy without this application ever handling a password,
running an OAuth flow, or holding a session.

`author_association` arrives in the same payload and says whether the sender is
an OWNER, MEMBER or COLLABORATOR on that repository. That is GitHub's own answer
to "may this person speak for the project", which is exactly the question the
correction path needs answered and the exact question a text box cannot answer.

The one thing it does not answer is whether a *former* maintainer still counts,
because the field reflects current permissions. That is the same defect
`derive_maintainers.py` exists to correct, so the known-maintainer list is
consulted as well.

## Teaching, not correcting

A maintainer writing "@precedent the whatsnew note goes in the next release
file" has not corrected anything. There is no earlier answer, no cited rule,
nothing to retire. They are stating a convention, which is a different operation
from a correction and gets its own origin. See migration 0005.

The comment is stored as an ordinary `review_comments` row carrying its real
pull request number, so it is embedded, retrieved and cited on the same path as
anything else said in review. What differs is that it becomes a rule
immediately, with verified authorship, instead of waiting to be noticed by
clustering.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from precedent.db.retry import with_retry
from precedent.embed.provider import EmbeddingProvider
from precedent.embed.vector import encode
from precedent.extract import contradiction
from precedent.extract.persist import Candidate, persist_rule
from precedent.transform.normalize import MAINTAINER_ASSOCIATIONS

log = logging.getLogger(__name__)

# Events worth reading. Everything else GitHub sends is noise for this purpose.
HANDLED_EVENTS = frozenset({"issue_comment", "pull_request_review_comment"})

STORE_COMMENT = text("""
    INSERT INTO review_comments (
        repo_id, github_node_id, kind, pr_number, author, author_association,
        is_maintainer, file_path, body, url, created_at,
        embedding, embedding_model, embedded_at
    ) VALUES (
        :repo_id, :node_id, CAST('maintainer_correction' AS comment_kind), :pr_number,
        :author, :association, true, :file_path, :body, :url, now(),
        CAST(:embedding AS VECTOR(1536)), :embedding_model, now()
    )
    ON CONFLICT (repo_id, github_node_id) DO NOTHING
    RETURNING id
""")


class SignatureInvalid(Exception):
    """The delivery did not come from GitHub, or was altered in transit."""


class AlreadyHandled(Exception):
    """This exact comment has been learned from before.

    GitHub retries a delivery it believes timed out, so the same comment can
    arrive several times. Distinct from "there was nothing to learn": both do
    nothing, but reporting a retry as an empty comment would hide the retry.
    """


@dataclass(slots=True)
class TaughtRule:
    statement: str
    rule_id: str
    outcome: str
    comment_id: str
    author: str
    pr_number: int
    superseded_statement: str | None = None


def verify_signature(secret: str, body: bytes, header: str | None) -> None:
    """Check GitHub's HMAC over the raw body.

    The raw bytes matter. Parsing the JSON and re-serialising it changes
    whitespace and key order, and the signature is over what was actually sent.

    `compare_digest` rather than `==` because a plain comparison returns early
    on the first differing byte, and the time it takes to fail leaks how much of
    a guess was right.
    """
    if not secret:
        raise SignatureInvalid("no webhook secret is configured")
    if not header or not header.startswith("sha256="):
        raise SignatureInvalid("missing or malformed signature header")

    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, header):
        raise SignatureInvalid("signature does not match")


def extract_instruction(body: str, trigger: str) -> str | None:
    """The text after the trigger, or None when the comment is not addressed to us.

    Quoted lines are dropped first. GitHub's reply button quotes the message
    being replied to, so without this the agent would read its own earlier
    output back as though a maintainer had written it.
    """
    unquoted = "\n".join(line for line in body.splitlines() if not line.lstrip().startswith(">"))
    match = re.search(rf"{re.escape(trigger)}\s*[:,]?\s*(.+)", unquoted, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return " ".join(match.group(1).split()).strip() or None


def speaks_for_project(
    association: str | None, login: str | None, known_maintainers: frozenset[str]
) -> bool:
    """Whether GitHub, or our own evidence, says this person speaks for the repo.

    `author_association` reflects permissions held *now*, so a maintainer who
    has stepped back reads as CONTRIBUTOR across their whole history. The
    derived list covers exactly that gap; see scripts/derive_maintainers.py.
    """
    if (association or "").upper() in MAINTAINER_ASSOCIATIONS:
        return True
    return bool(login) and login.lower() in known_maintainers


def parse_event(event: str, payload: dict) -> dict | None:
    """Flatten the two comment event shapes into one, or None if uninteresting."""
    if event not in HANDLED_EVENTS or payload.get("action") != "created":
        return None

    comment = payload.get("comment") or {}
    # An issue_comment fires for issues too. Only pull requests have review.
    if event == "issue_comment" and not (payload.get("issue") or {}).get("pull_request"):
        return None

    number = (payload.get("issue") or payload.get("pull_request") or {}).get("number")
    if number is None:
        return None

    return {
        "node_id": comment.get("node_id") or f"gh:{comment.get('id')}",
        "body": comment.get("body") or "",
        "author": ((comment.get("user") or {}).get("login")),
        "association": comment.get("author_association"),
        "url": comment.get("html_url"),
        "pr_number": number,
        "file_path": comment.get("path"),
    }


async def _store(
    engine: AsyncEngine,
    provider: EmbeddingProvider,
    *,
    repo_id: str,
    parsed: dict,
    instruction: str,
) -> str | None:
    vector = encode((await provider.embed([instruction]))[0])

    async def op() -> str | None:
        async with engine.begin() as conn:
            return (
                await conn.execute(
                    STORE_COMMENT,
                    {
                        "repo_id": repo_id,
                        "node_id": parsed["node_id"],
                        "pr_number": parsed["pr_number"],
                        "author": parsed["author"],
                        "association": parsed["association"] or "NONE",
                        "file_path": parsed["file_path"],
                        "body": instruction,
                        "url": parsed["url"],
                        "embedding": vector,
                        "embedding_model": getattr(provider, "model", None),
                    },
                )
            ).scalar_one_or_none()

    row = await with_retry(op, description="store taught comment")
    return str(row) if row else None


SYSTEM = """\
A maintainer has stated a convention while reviewing a pull request. Turn it \
into a rule the project can be held to.

Rules:

  - State it as an instruction, in one sentence, the way a contributing guide \
would put it.
  - Write it so it stands alone. Whoever reads it in six months will not have \
the pull request in front of them.
  - Add nothing the maintainer did not say. If they described a narrow case, \
the narrow case is the whole rule.
  - Refuse if there is no convention here. Questions, opinions about one \
specific change, and thinking aloud are not conventions. Set usable to false.

Reply with JSON only:

{
  "usable": true or false,
  "statement": "the convention, one sentence, imperative",
  "rationale": "why, one sentence, only if they gave a reason",
  "scope": "repo" | "directory" | "file" | "api" | "testing" | "docs" | "style" | "process",
  "scope_pattern": null
}
"""

VALID_SCOPES = {"repo", "directory", "file", "api", "testing", "docs", "style", "process"}


async def teach(
    engine: AsyncEngine,
    provider: EmbeddingProvider,
    chat,
    *,
    repo_id: str,
    parsed: dict,
    instruction: str,
) -> TaughtRule | None:
    """Turn a maintainer's pull request comment into a rule."""
    drafted = await chat.complete_json(
        [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": (
                    f"On pull request #{parsed['pr_number']}"
                    + (f", on {parsed['file_path']}" if parsed.get("file_path") else "")
                    + f", {parsed['author']} wrote:\n\n{instruction}"
                ),
            },
        ]
    )

    statement = (drafted.get("statement") or "").strip()
    if drafted.get("usable") is False or not statement:
        log.info("nothing to learn from %s on #%s", parsed["author"], parsed["pr_number"])
        return None

    scope = (drafted.get("scope") or "repo").strip().lower()
    if scope not in VALID_SCOPES:
        scope = "repo"
    pattern = drafted.get("scope_pattern") if scope in ("directory", "file") else None

    comment_id = await _store(
        engine, provider, repo_id=repo_id, parsed=parsed, instruction=instruction
    )
    if comment_id is None:
        # ON CONFLICT DO NOTHING returned no row, so this delivery is a repeat.
        # Learning the same thing twice would inflate a rule's evidence on
        # nothing but network flakiness.
        raise AlreadyHandled(parsed["node_id"])

    now = datetime.now(UTC)

    async def judge(**kwargs):
        return await chat.complete_json(contradiction.build_messages(**kwargs))

    result = await persist_rule(
        engine,
        provider,
        repo_id=repo_id,
        candidate=Candidate(
            statement=statement,
            rationale=(drafted.get("rationale") or "").strip() or None,
            scope=scope,
            scope_pattern=pattern,
            comment_ids=[comment_id],
            distinct_prs=1,
            distinct_authors=1,
            first_evidence_at=now,
            last_evidence_at=now,
        ),
        judge=judge,
        origin="taught",
    )

    log.info(
        "learned from %s on #%s (%s): %s",
        parsed["author"],
        parsed["pr_number"],
        result.outcome,
        statement[:70],
    )
    return TaughtRule(
        statement=statement,
        rule_id=result.rule_id,
        outcome=result.outcome,
        comment_id=comment_id,
        author=parsed["author"],
        pr_number=parsed["pr_number"],
    )
