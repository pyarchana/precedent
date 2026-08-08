"""HTTP interface to the memory: ask, correct, and inspect what changed.

Thin on purpose. Every decision this serves already lives in `agent/`, and
duplicating any of it here would mean the demo and the command line could drift
into disagreeing about what memory says. The endpoints translate JSON to those
calls and back.

## What this module actually adds

Two things the command line never needed, both consequences of the URL being
public:

**A spend ceiling across the process, not per request.** The project runs on a
single unreplenishable top-up. A per-request cap bounds nothing when anyone can
send a thousand requests, so the budget is shared by every caller and the
answer once it is gone is a plain 503 rather than a degraded answer from the
base model. Falling back to unsourced generation when the money runs out would
break the one guarantee this system makes.

**A rate limit.** Same reason, faster failure mode. It is per client address and
deliberately crude; anything more would need shared state this does not have.

Corrections carry a separate switch. They write to memory, and a public demo
that anyone can rewrite is a demo that will be rewritten, so
`api_corrections_enabled` turns them off without a redeploy.

    uvicorn precedent.api.app:app --reload
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import text

from precedent.agent.answer import answer_question, citation_label
from precedent.agent.correct import apply_correction
from precedent.agent.retrieve import recall
from precedent.agent.session import open_session, record_turn
from precedent.config import get_settings
from precedent.db.engine import create_engine
from precedent.embed.provider import OpenAIEmbeddings
from precedent.extract.llm import BudgetExhausted, ChatClient, QuotaExhausted

log = logging.getLogger("precedent.api")


@dataclass
class Deps:
    """Built once at startup. Creating an engine per request would open a new
    pool to a cluster in another region on every call."""

    engine: object = None
    provider: object = None
    chat: object = None
    repo_id: str = ""


deps = Deps()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if not settings.cockroach_dsn:
        raise RuntimeError("COCKROACH_DSN is not set")

    deps.engine = create_engine(settings.cockroach_dsn)
    deps.provider = OpenAIEmbeddings(
        settings.openai_api_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dim,
    )
    deps.chat = ChatClient(settings.openai_api_key, max_spend=settings.api_budget_usd)

    owner, _, name = settings.repo_slug.partition("/")
    async with deps.engine.connect() as conn:
        deps.repo_id = str(
            (
                await conn.execute(
                    text("SELECT id FROM repos WHERE owner = :o AND name = :n"),
                    {"o": owner, "n": name},
                )
            ).scalar_one()
        )
    log.info("ready: repo %s, budget $%.2f", settings.repo_slug, settings.api_budget_usd)

    yield

    await deps.provider.aclose()
    await deps.chat.aclose()
    await deps.engine.dispose()


app = FastAPI(
    title="Precedent",
    description="Agentic memory over a repository's code review history.",
    lifespan=lifespan,
)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _settings.api_cors_origins.split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Request timestamps per client. Bounded by the window, so it cannot grow
# without limit for a single caller, but the number of distinct callers is not
# bounded and this would need replacing before anything other than a demo.
_seen: dict[str, deque[float]] = defaultdict(deque)


def enforce_rate_limit(request: Request) -> None:
    limit = get_settings().api_rate_limit_per_minute
    if limit <= 0:
        return
    who = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = _seen[who]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= limit:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit is {limit} requests a minute. This demo runs on a fixed budget.",
        )
    window.append(now)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    session_id: str | None = None
    contributor: str | None = Field(default=None, max_length=100)


class Citation(BaseModel):
    label: str
    pr_number: int | None
    author: str | None
    url: str | None
    kind: str
    excerpt: str


class RuleView(BaseModel):
    id: str
    statement: str
    rationale: str | None
    confidence: float
    evidence_count: int
    is_correction: bool
    is_weak: bool
    citations: list[Citation]


class AskResponse(BaseModel):
    answer: str
    answered: bool
    confidence: str
    session_id: str
    turn: int
    rules: list[RuleView]
    cited_prs: list[int]
    trustworthy: bool
    spent_usd: float


def _citation(hit) -> Citation:
    body = " ".join(hit.body.split())
    return Citation(
        label=citation_label(hit),
        # Corrections carry a zero sentinel that is not a real pull request.
        pr_number=hit.pr_number if hit.kind != "maintainer_correction" else None,
        author=hit.author,
        url=hit.url,
        kind=hit.kind,
        excerpt=body[:400],
    )


@app.get("/health")
async def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "repo": settings.repo_slug,
        "spent_usd": round(getattr(deps.chat, "spent", 0.0), 4),
        "budget_usd": settings.api_budget_usd,
        "corrections_enabled": settings.api_corrections_enabled,
    }


@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, request: Request) -> AskResponse:
    enforce_rate_limit(request)

    memory = await recall(deps.engine, deps.provider, repo_id=deps.repo_id, question=body.question)

    try:
        result = await answer_question(deps.chat, memory)
    except BudgetExhausted as exc:
        # Deliberately not degraded into an unsourced answer. An answer this
        # system cannot attribute is worse than no answer.
        raise HTTPException(
            status_code=503,
            detail="This demo's model budget is spent. Memory is still readable at /rules.",
        ) from exc
    except QuotaExhausted as exc:
        raise HTTPException(status_code=503, detail="The model account has no credit.") from exc

    session_id = body.session_id or await open_session(
        deps.engine, repo_id=deps.repo_id, contributor_login=body.contributor
    )
    turn = await record_turn(
        deps.engine,
        repo_id=deps.repo_id,
        session_id=session_id,
        question=body.question,
        answer=result.text,
        rule_ids=result.rule_ids,
        comment_ids=result.comment_ids,
        answered_from_memory=result.answered,
    )

    return AskResponse(
        answer=result.text,
        answered=result.answered,
        confidence=result.confidence,
        session_id=session_id,
        turn=turn,
        rules=[
            RuleView(
                id=r.id,
                statement=r.statement,
                rationale=r.rationale,
                confidence=round(r.confidence, 3),
                evidence_count=r.evidence_count,
                is_correction=r.is_correction,
                is_weak=r.is_weak,
                citations=[_citation(c) for c in r.citations],
            )
            for r in memory.rules
        ],
        cited_prs=result.cited_prs,
        trustworthy=result.is_trustworthy,
        spent_usd=round(deps.chat.spent, 4),
    )


class CorrectRequest(BaseModel):
    session_id: str
    turn: int
    correction: str = Field(min_length=5, max_length=2000)
    maintainer: str = Field(min_length=1, max_length=100)


class CorrectResponse(BaseModel):
    statement: str
    outcome: str
    rule_id: str
    corrected_rule_id: str | None
    corrected_statement: str | None
    reason: str
    changed_memory: bool
    spent_usd: float


@app.post("/correct", response_model=CorrectResponse)
async def correct(body: CorrectRequest, request: Request) -> CorrectResponse:
    settings = get_settings()
    if not settings.api_corrections_enabled:
        raise HTTPException(status_code=403, detail="Corrections are disabled on this deployment.")
    enforce_rate_limit(request)

    try:
        result = await apply_correction(
            deps.engine,
            deps.provider,
            deps.chat,
            repo_id=deps.repo_id,
            session_id=body.session_id,
            turn_number=body.turn,
            maintainer_login=body.maintainer,
            correction=body.correction,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except BudgetExhausted as exc:
        raise HTTPException(status_code=503, detail="This demo's model budget is spent.") from exc

    return CorrectResponse(
        statement=result.statement,
        outcome=result.outcome,
        rule_id=result.rule_id,
        corrected_rule_id=result.corrected_rule_id,
        corrected_statement=result.corrected_statement,
        reason=result.reason,
        changed_memory=result.changed_memory,
        spent_usd=round(deps.chat.spent, 4),
    )


RECENT_RULES = text("""
    SELECT id, statement, rationale, confidence, evidence_count, origin,
           status::STRING AS status, supersession_reason, updated_at
    FROM rules
    WHERE repo_id = :repo_id AND status = 'active'
    ORDER BY confidence DESC, evidence_count DESC
    LIMIT :limit
""")

# Superseded rules, newest first. The demo needs these: "we used to say X" is
# the part that shows memory changing rather than merely being queried.
SUPERSEDED_RULES = text("""
    SELECT old.id, old.statement, old.supersession_reason, old.superseded_at,
           new.statement AS replaced_by
    FROM rules old
    LEFT JOIN rules new ON new.repo_id = old.repo_id AND new.id = old.superseded_by
    WHERE old.repo_id = :repo_id AND old.status = 'superseded'
    ORDER BY old.superseded_at DESC
    LIMIT :limit
""")


@app.get("/rules")
async def rules(limit: int = 20) -> dict:
    """Read-only, unmetered, and available after the budget is gone."""
    limit = max(1, min(limit, 100))
    async with deps.engine.connect() as conn:
        active = (
            (await conn.execute(RECENT_RULES, {"repo_id": deps.repo_id, "limit": limit}))
            .mappings()
            .all()
        )
        retired = (
            (await conn.execute(SUPERSEDED_RULES, {"repo_id": deps.repo_id, "limit": limit}))
            .mappings()
            .all()
        )

    return {
        "active": [
            {
                "id": str(r["id"]),
                "statement": r["statement"],
                "rationale": r["rationale"],
                "confidence": round(float(r["confidence"]), 3),
                "evidence_count": r["evidence_count"],
                "origin": r["origin"],
            }
            for r in active
        ],
        "superseded": [
            {
                "id": str(r["id"]),
                "statement": r["statement"],
                "reason": r["supersession_reason"],
                "replaced_by": r["replaced_by"],
                "at": r["superseded_at"].isoformat() if r["superseded_at"] else None,
            }
            for r in retired
        ],
    }
