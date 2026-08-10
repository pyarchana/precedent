"""HTTP interface to the memory: ask, correct, and inspect what changed.

Thin on purpose. Every decision this serves already lives in `agent/`, and
duplicating any of it here would mean the demo and the command line could drift
into disagreeing about what memory says. The endpoints translate JSON to those
calls and back.

## What this module actually adds

Two things the command line never needed, both consequences of the URL being
public:

**A spend ceiling that outlives the process.** The project runs on a single
unreplenishable top-up. A per-request cap bounds nothing when anyone can send a
thousand requests, so the budget is shared by every caller, and once it is gone
the answer is a plain 503 rather than a degraded answer from the base model.
Falling back to unsourced generation when the money runs out would break the one
guarantee this system makes.

The counter lives in the database rather than in memory, and that is not
over-engineering. The first version kept it on the model client, which is
correct under uvicorn and useless under Lambda: Mangum runs the ASGI lifespan on
every invocation, so the client and its counter were rebuilt per request and the
ceiling restarted at zero each time. Even with that fixed, Lambda runs several
execution environments at once and each would believe it had the whole budget.
A ceiling that resets is not a ceiling.

**A rate limit.** Same reason, faster failure mode. It is per client address and
per process, so it shares the weakness the budget no longer has; it is a speed
bump rather than a guarantee, and the durable budget is what actually bounds the
damage.

Corrections carry a separate switch. They write to memory, and a public demo
that anyone can rewrite is a demo that will be rewritten, so
`api_corrections_enabled` turns them off without a redeploy.

    uvicorn precedent.api.app:app --reload
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import text

from precedent.agent.answer import answer_question, citation_label
from precedent.agent.correct import UnusableCorrection, apply_correction
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
    # Why startup failed, when it did. Kept so the function can say what is
    # wrong instead of returning an opaque 502 that requires CloudWatch to read.
    error: str = ""


deps = Deps()


# True when running inside Lambda, where the lifespan contract is different.
IN_LAMBDA = bool(os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))


async def ensure_deps() -> None:
    """Build the shared clients once, and only once.

    Idempotent because Mangum runs the ASGI lifespan on **every invocation**,
    not once per execution environment. Rebuilding here would open a fresh
    connection pool to a cluster in another region on every request, which was
    measured at over three seconds even for a static page.

    Worse, it would reset the model client, and with it the spend counter, so
    the budget ceiling would restart at zero on every request and cap nothing
    at all.
    """
    if deps.engine is not None:
        return

    settings = get_settings()
    try:
        if not settings.cockroach_dsn:
            raise RuntimeError("COCKROACH_DSN is not set")
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        engine = create_engine(settings.cockroach_dsn)
        owner, _, name = settings.repo_slug.partition("/")
        async with engine.connect() as conn:
            deps.repo_id = str(
                (
                    await conn.execute(
                        text("SELECT id FROM repos WHERE owner = :o AND name = :n"),
                        {"o": owner, "n": name},
                    )
                ).scalar_one()
            )

        deps.provider = OpenAIEmbeddings(
            settings.openai_api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dim,
        )
        deps.chat = ChatClient(settings.openai_api_key, max_spend=settings.api_budget_usd)
        # Assigned last, so a partial failure leaves this None and the next
        # invocation retries rather than serving a half-built application.
        deps.engine = engine
        deps.error = ""
        log.info("ready: repo %s, budget $%.2f", settings.repo_slug, settings.api_budget_usd)
    except Exception as exc:
        # Deliberately not re-raised. Under Mangum a lifespan failure becomes a
        # bare 502 with the reason only in CloudWatch, so a misconfigured
        # deployment looks identical to a broken one. Recording it here lets
        # /health say which environment variable is missing, which is the
        # difference between a two minute fix and an afternoon.
        deps.error = f"{type(exc).__name__}: {exc}"
        log.exception("startup failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await ensure_deps()
    yield
    # Nothing is torn down under Lambda. The lifespan ends when the invocation
    # ends, and disposing the pool here would throw away the connection the
    # next request is about to want.
    if not IN_LAMBDA and deps.engine is not None:
        await deps.provider.aclose()
        await deps.chat.aclose()
        await deps.engine.dispose()
        deps.engine = None


def require_ready() -> None:
    """Refuse with the reason rather than failing shapelessly."""
    if deps.engine is None:
        raise HTTPException(
            status_code=503,
            detail=deps.error or "The service is still starting.",
        )


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


def enforce_rate_limit(request: Request, multiplier: int = 1) -> None:
    """Cap requests per client address per minute.

    `multiplier` loosens the cap for endpoints that cost no money. They are not
    free, though: every read still consumes CockroachDB request units, and the
    cluster runs on a free tier with a zero spend limit, so exhausting those
    takes the whole demo down rather than merely stopping answers.

    Per process, so several Lambda containers each enforce their own count.
    That makes this a speed bump rather than a guarantee; the durable thing is
    the spend ledger, which every container shares.
    """
    limit = get_settings().api_rate_limit_per_minute * multiplier
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


# Spend is read before a call and recorded after it, both against the shared
# row rather than any process's memory. See migrations/0004_api_usage.sql for
# why a per-process counter cannot hold a ceiling here.
SPEND_TODAY = text("SELECT coalesce(spend_usd, 0) FROM api_usage WHERE day = current_date()")

RECORD_SPEND = text("""
    INSERT INTO api_usage (day, calls, spend_usd)
    VALUES (current_date(), 1, :amount)
    ON CONFLICT (day) DO UPDATE SET
        calls = api_usage.calls + 1,
        spend_usd = api_usage.spend_usd + :amount,
        updated_at = now()
""")


async def spend_today() -> float:
    async with deps.engine.connect() as conn:
        return float((await conn.execute(SPEND_TODAY)).scalar() or 0.0)


async def check_budget() -> None:
    """Refuse before spending, not after."""
    limit = get_settings().api_budget_usd
    if limit <= 0:
        return
    if await spend_today() >= limit:
        raise HTTPException(
            status_code=503,
            detail=(
                "This demo's model budget for today is spent. Memory is still "
                "readable at /rules, and questions work again tomorrow."
            ),
        )


async def record_spend(amount: float) -> None:
    """Never allowed to fail the request it is accounting for.

    The answer has already been produced and the money already spent by the
    time this runs. Losing the record understates the total, which is a problem
    worth logging and not one worth turning into a 500 for someone who asked a
    perfectly good question.
    """
    if amount <= 0:
        return
    try:
        async with deps.engine.begin() as conn:
            await conn.execute(RECORD_SPEND, {"amount": amount})
    except Exception:
        log.exception("could not record $%.4f of spend; the daily total is now low", amount)


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
    """Answers even when startup failed, because that is when it is needed."""
    settings = get_settings()
    ready = deps.engine is not None
    body: dict = {
        "status": "ok" if ready else "misconfigured",
        "repo": settings.repo_slug,
        "corrections_enabled": settings.api_corrections_enabled,
        "budget_usd": settings.api_budget_usd,
    }
    if not ready:
        # Names the missing setting, never its value.
        body["error"] = deps.error or "not started"
        body["missing"] = [
            name
            for name, present in (
                ("COCKROACH_DSN", bool(settings.cockroach_dsn)),
                ("OPENAI_API_KEY", bool(settings.openai_api_key)),
            )
            if not present
        ]
        return body

    body["spent_today_usd"] = round(await spend_today(), 4)
    return body


@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, request: Request) -> AskResponse:
    require_ready()
    enforce_rate_limit(request)
    await check_budget()

    memory = await recall(deps.engine, deps.provider, repo_id=deps.repo_id, question=body.question)

    before = deps.chat.spent
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
    finally:
        # In `finally` because a call that raised may still have spent money.
        await record_spend(deps.chat.spent - before)

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
    # Near duplicates of the corrected rule that the sweep also retired. Shown
    # in the demo because "one correction retired three rules" is the part that
    # demonstrates memory changing rather than merely being appended to.
    also_retired: list[str]
    reason: str
    changed_memory: bool
    spent_usd: float


@app.post("/correct", response_model=CorrectResponse)
async def correct(body: CorrectRequest, request: Request) -> CorrectResponse:
    require_ready()
    settings = get_settings()
    if not settings.api_corrections_enabled:
        raise HTTPException(status_code=403, detail="Corrections are disabled on this deployment.")
    enforce_rate_limit(request)
    await check_budget()

    before = deps.chat.spent
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
    except UnusableCorrection as exc:
        # 422 rather than 400: the request was well formed, the content was not
        # actionable. Nothing was written, and the message says what is missing.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except BudgetExhausted as exc:
        raise HTTPException(status_code=503, detail="This demo's model budget is spent.") from exc
    finally:
        await record_spend(deps.chat.spent - before)

    return CorrectResponse(
        statement=result.statement,
        outcome=result.outcome,
        rule_id=result.rule_id,
        corrected_rule_id=result.corrected_rule_id,
        corrected_statement=result.corrected_statement,
        also_retired=result.also_retired,
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
async def rules(request: Request, limit: int = 20) -> dict:
    """Read-only, unbilled, and available after the model budget is gone."""
    require_ready()
    # Loosely limited rather than unlimited: it spends no money but it does
    # spend request units on a free-tier cluster.
    enforce_rate_limit(request, multiplier=6)
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


# Mounted last so it cannot shadow an endpoint. A single self-contained page
# with no build step: it deploys with the application, which matters when the
# target is one Lambda rather than a bucket and a CDN.
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
