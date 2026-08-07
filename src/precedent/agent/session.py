"""Working memory: the conversation a correction later refers back to.

A correction is not a standalone fact. It is always "that answer, the one you
gave me, was wrong, and here is what is actually true". To act on it the system
has to be able to find the answer being corrected and, more importantly, the
rules that answer was built from. Without that, a maintainer's correction is
just another opinion floating free of the thing it contradicts, and the system
has no way to know which of two hundred and fifty rules to retire.

So every answer is written down before it is shown: the question, the text, and
the ids of the material it used. The rule ids are stored on the turn rather
than looked up again later, because the point of the whole exercise is that
memory changes. A correction arriving after those rules have been superseded
must still see which rules the answer actually used, not which rules the same
question would retrieve today.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from precedent.db.retry import with_retry

log = logging.getLogger(__name__)

OPEN_SESSION = text("""
    INSERT INTO sessions (repo_id, contributor_login)
    VALUES (:repo_id, :login)
    RETURNING id
""")

TOUCH_SESSION = text("""
    UPDATE sessions SET last_active_at = now()
    WHERE repo_id = :repo_id AND id = :session_id
""")

NEXT_TURN = text("""
    SELECT coalesce(max(turn_number), 0) + 1
    FROM session_turns
    WHERE repo_id = :repo_id AND session_id = :session_id
""")

LOAD_TURN = text("""
    SELECT question, answer, cited_rule_ids, cited_comment_ids, answered_from_memory
    FROM session_turns
    WHERE repo_id = :repo_id AND session_id = :session_id AND turn_number = :turn_number
""")


@dataclass(slots=True)
class Turn:
    session_id: str
    turn_number: int
    question: str
    answer: str | None = None
    cited_rule_ids: list[str] = None  # type: ignore[assignment]
    cited_comment_ids: list[str] = None  # type: ignore[assignment]
    answered_from_memory: bool = True

    def __post_init__(self) -> None:
        if self.cited_rule_ids is None:
            self.cited_rule_ids = []
        if self.cited_comment_ids is None:
            self.cited_comment_ids = []

    @property
    def reference(self) -> str:
        """What a maintainer quotes back when correcting this answer."""
        return f"{self.session_id}/{self.turn_number}"


async def open_session(
    engine: AsyncEngine, *, repo_id: str, contributor_login: str | None = None
) -> str:
    async def op() -> str:
        async with engine.begin() as conn:
            return str(
                (
                    await conn.execute(
                        OPEN_SESSION, {"repo_id": repo_id, "login": contributor_login}
                    )
                ).scalar_one()
            )

    session_id = await with_retry(op, description="open session")
    log.info("opened session %s for %s", session_id[:8], contributor_login or "anonymous")
    return session_id


def _uuid_array(values, prefix: str) -> tuple[str, dict[str, str]]:
    """Build a literal UUID array with bound parameters.

    The ids arrive as strings and the columns are UUID[]. asyncpg will not
    adapt a Python list of strings to that type, and interpolating the ids into
    the statement would be a SQL injection waiting for the day one of them
    comes from somewhere less trusted than our own database.
    """
    if not values:
        return "ARRAY[]::UUID[]", {}
    names = [f"{prefix}_{i}" for i in range(len(values))]
    sql = "ARRAY[" + ", ".join(f":{n}" for n in names) + "]::UUID[]"
    return sql, {n: str(v) for n, v in zip(names, values, strict=True)}


async def record_turn(
    engine: AsyncEngine,
    *,
    repo_id: str,
    session_id: str,
    question: str,
    answer: str,
    rule_ids=(),
    comment_ids=(),
    answered_from_memory: bool = True,
) -> int:
    """Write an answer down and return its turn number."""
    rules_sql, rule_params = _uuid_array(rule_ids, "rid")
    comments_sql, comment_params = _uuid_array(comment_ids, "cid")

    statement = text(f"""
        INSERT INTO session_turns (
            repo_id, session_id, turn_number, question, answer,
            cited_rule_ids, cited_comment_ids, answered_from_memory
        ) VALUES (
            :repo_id, :session_id, :turn_number, :question, :answer,
            {rules_sql}, {comments_sql}, :answered_from_memory
        )
    """)

    async def op() -> int:
        # Numbering and insert share a transaction. CockroachDB's serializable
        # isolation is what makes that safe against a second turn recorded at
        # the same moment: one of the two transactions retries rather than both
        # claiming the same number.
        async with engine.begin() as conn:
            turn_number = int(
                (
                    await conn.execute(NEXT_TURN, {"repo_id": repo_id, "session_id": session_id})
                ).scalar_one()
            )
            await conn.execute(
                statement,
                {
                    "repo_id": repo_id,
                    "session_id": session_id,
                    "turn_number": turn_number,
                    "question": question,
                    "answer": answer,
                    "answered_from_memory": answered_from_memory,
                    **rule_params,
                    **comment_params,
                },
            )
            await conn.execute(TOUCH_SESSION, {"repo_id": repo_id, "session_id": session_id})
            return turn_number

    return await with_retry(op, description="record turn")


async def load_turn(
    engine: AsyncEngine, *, repo_id: str, session_id: str, turn_number: int
) -> Turn | None:
    async with engine.connect() as conn:
        row = (
            (
                await conn.execute(
                    LOAD_TURN,
                    {
                        "repo_id": repo_id,
                        "session_id": session_id,
                        "turn_number": turn_number,
                    },
                )
            )
            .mappings()
            .first()
        )

    if row is None:
        return None

    return Turn(
        session_id=session_id,
        turn_number=turn_number,
        question=row["question"],
        answer=row["answer"],
        cited_rule_ids=[str(x) for x in (row["cited_rule_ids"] or [])],
        cited_comment_ids=[str(x) for x in (row["cited_comment_ids"] or [])],
        answered_from_memory=row["answered_from_memory"],
    )
