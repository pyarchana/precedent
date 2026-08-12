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

The counter lives in the database rather than in memory, and it took two
attempts to make it a real ceiling.

The first version kept it on the model client, which is correct under uvicorn
and useless under Lambda: Mangum runs the ASGI lifespan on every invocation, so
the client and its counter were rebuilt per request and the ceiling restarted at
zero each time.

Moving it into the database fixed that and still did not bind, because the
sequence was read the total, call the model, add the cost. Two containers
reading $0.90 against a $1.00 limit both pass and the day ends at $1.30.
Measured, not theorised: ten concurrent requests all passed.

So a request now **reserves** room for the most it could cost, in the same
statement that checks there is any, and settles the reservation against the real
cost afterwards. The same ten concurrent requests now yield one. A limit that
can be exceeded is not a limit, and this file used to claim it was the durable
guarantee while being none of those things.

**A rate limit.** Same reason, faster failure mode. It is per client address and
per process, so it shares the weakness the budget no longer has; it is a speed
bump rather than a guarantee, and the durable budget is what actually bounds the
damage.

Corrections through this API carry a switch that is **off by default**. There is
no way here to verify that whoever put "jbrockmendel" in the maintainer field is
jbrockmendel, and an endpoint that rewrites memory on an unverified name is a
hole rather than a feature. The GitHub App path in `github_app.py` does not need
the switch, because GitHub signs those deliveries and the identity is real.

    uvicorn precedent.api.app:app --reload
"""

from __future__ import annotations

import logging
import os
import time
import uuid
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
from precedent.api.github_app import (
    AlreadyHandled,
    SignatureInvalid,
    acknowledge,
    act_on_pull_request,
    extract_instruction,
    parse_event,
    parse_pull_request,
    speaks_for_project,
    teach,
    verify_signature,
)
from precedent.api.github_auth import Credentials, GitHubApp, GitHubError, NotConfigured
from precedent.config import get_settings
from precedent.db.engine import create_engine
from precedent.db.retry import with_retry
from precedent.embed.provider import OpenAIEmbeddings
from precedent.extract.llm import BudgetExhausted, ChatClient, QuotaExhausted
from precedent.transform.maintainers import load_maintainers

log = logging.getLogger("precedent.api")


@dataclass
class Deps:
    """Built once at startup. Creating an engine per request would open a new
    pool to a cluster in another region on every call."""

    engine: object = None
    provider: object = None
    chat: object = None
    repo_id: str = ""
    # Only set when the GitHub App has credentials. Left None otherwise, and
    # deliberately not a startup failure: listening to a repository needs the
    # webhook secret alone, so a deployment that only learns should not be dead
    # because it cannot also post.
    github: object = None
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

        if settings.github_reviews_enabled:
            try:
                deps.github = GitHubApp(Credentials.from_settings(settings))
            except NotConfigured as exc:
                # Not an error. The receive side works without any of this, and
                # saying so at startup is more useful than discovering it on the
                # first pull request.
                log.info("acting on pull requests is off (%s)", exc)

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


# The most a single request is allowed to cost. It bounds the reservation, so
# it has to be an over-estimate rather than an average: a correction makes up
# to fourteen model calls, and reserving the cost of one would let fourteen
# requests through on the budget for fourteen.
MAX_REQUEST_USD = 0.05

# A reservation this old belongs to a container that died before releasing it.
# Without expiry those leaks would quietly eat the day's allowance.
RESERVATION_TTL_SECONDS = 300

# Outstanding reservations are counted from the table rather than cached in a
# column on api_usage. The cached version needed a sweep to stay honest after a
# container died mid-request, and a counter that needs a background job to stay
# correct is a counter that will eventually be wrong. Summing live rows cannot
# drift: an abandoned reservation simply ages out of the window.
LIVE_RESERVED = """
    coalesce((SELECT sum(amount_usd) FROM api_reservations
              WHERE day = current_date() AND created_at > now() - INTERVAL '{ttl} seconds'), 0)
"""

SPENT_TODAY_SQL = "coalesce((SELECT spend_usd FROM api_usage WHERE day = current_date()), 0)"

# The check and the claim are one statement. Reading the total and then
# spending leaves a window where another container reads the same total, which
# is how a $1.00 ceiling ends the day at $1.30. Under serializable isolation two
# of these conflict and one retries, so the WHERE decides who gets the room.
RESERVE = text(f"""
    INSERT INTO api_reservations (id, day, amount_usd)
    SELECT :id, current_date(), :amount
    WHERE {SPENT_TODAY_SQL} + {LIVE_RESERVED.format(ttl=300)} + :amount <= :limit
    RETURNING id
""")

SETTLE = text("""
    INSERT INTO api_usage (day, calls, spend_usd)
    VALUES (current_date(), 1, :spent)
    ON CONFLICT (day) DO UPDATE SET
        calls = api_usage.calls + 1,
        spend_usd = api_usage.spend_usd + :spent,
        updated_at = now()
""")

RELEASE_HOLD = text("DELETE FROM api_reservations WHERE id = :id")

# Opportunistic, and safe to skip: the window in LIVE_RESERVED already excludes
# these, so this only reclaims disk.
SWEEP_STALE = text("DELETE FROM api_reservations WHERE created_at < now() - INTERVAL '300 seconds'")

SPEND_SUMMARY = text(f"""
    SELECT {SPENT_TODAY_SQL} AS spent, {LIVE_RESERVED.format(ttl=300)} AS reserved
""")


async def spend_today() -> float:
    async with deps.engine.connect() as conn:
        return float((await conn.execute(SPEND_TODAY)).scalar() or 0.0)


async def reserve_budget() -> str | None:
    """Claim room for one request, or refuse.

    Returns a reservation id to settle with, or None when the budget is
    unlimited. Raises 503 when there is no room.

    The claim and the check are the same statement on purpose. Reading the
    total and then spending leaves a window in which another container reads
    the same total, which is how a $1.00 ceiling ends the day at $1.30.
    """
    limit = get_settings().api_budget_usd
    if limit <= 0:
        return None

    reservation_id = str(uuid.uuid4())

    async def op() -> bool:
        async with deps.engine.begin() as conn:
            taken = (
                await conn.execute(
                    RESERVE,
                    {"id": reservation_id, "amount": MAX_REQUEST_USD, "limit": limit},
                )
            ).first()
            return taken is not None

    if not await with_retry(op, description="reserve budget"):
        raise HTTPException(
            status_code=503,
            detail=(
                "This demo's model budget for today is spent. Memory is still "
                "readable at /rules, and questions work again tomorrow."
            ),
        )
    return reservation_id


async def settle_budget(reservation_id: str | None, spent: float) -> None:
    """Replace the reservation with what the request actually cost.

    Never allowed to fail the request it is accounting for: the answer is
    already produced and the money already gone. But it must not be skipped
    either, or the reservation leaks until the sweep reclaims it, so failures
    are logged loudly rather than swallowed quietly.
    """
    if reservation_id is None:
        if spent > 0:
            log.info("unlimited budget; %.4f spent and not recorded", spent)
        return
    try:
        async with deps.engine.begin() as conn:
            await conn.execute(SETTLE, {"spent": max(0.0, spent)})
            await conn.execute(RELEASE_HOLD, {"id": reservation_id})
            await conn.execute(SWEEP_STALE)
    except Exception:
        log.exception(
            "could not settle reservation %s for $%.4f; the sweep will reclaim it in %ds",
            reservation_id,
            spent,
            RESERVATION_TTL_SECONDS,
        )


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
    reservation = await reserve_budget()

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
        # In `finally` because a call that raised may still have spent money,
        # and because the reservation has to be released either way.
        await settle_budget(reservation, deps.chat.spent - before)

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
    reservation = await reserve_budget()

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
        await settle_budget(reservation, deps.chat.spent - before)

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


async def review_pull_request(settings, opened: dict) -> dict:
    """A pull request opened and nobody asked us anything.

    Every response here is a 200, including the ones that do nothing. GitHub
    retries a delivery that failed, and there is no failure it could retry its
    way out of: the app is either configured to post or it is not, and the pull
    request either has conventions attached to it or it does not.
    """
    if deps.github is None:
        return {"status": "ignored", "reason": "the app has no credentials to post with"}

    require_ready()
    reservation = await reserve_budget()
    before = deps.chat.spent
    try:
        decision = await act_on_pull_request(
            deps.engine,
            deps.provider,
            deps.github,
            repo_id=deps.repo_id,
            parsed=opened,
            trigger=settings.github_trigger,
            source_repo=settings.repo_slug,
        )
    except AlreadyHandled:
        return {"status": "ignored", "reason": "already reviewed this pull request"}
    except (BudgetExhausted, QuotaExhausted):
        return {"status": "ignored", "reason": "model budget spent"}
    except GitHubError as exc:
        # Most often a permission the installation was never granted, which no
        # retry fixes. Logged loudly because it is invisible from GitHub's side:
        # the delivery succeeded, the comment simply never appeared.
        log.error("could not act on %s#%s: %s", opened["repo"], opened["pr_number"], exc)
        return {"status": "error", "reason": "github refused the request"}
    finally:
        await settle_budget(reservation, deps.chat.spent - before)

    if not decision.will_speak:
        return {
            "status": "silent",
            "reason": decision.silent_reason,
            "pr_number": decision.pr_number,
            "files_considered": len(decision.paths),
        }

    return {
        "status": "commented",
        "pr_number": decision.pr_number,
        "files_considered": len(decision.paths),
        "rules": [
            {
                "id": item.rule.id,
                "statement": item.rule.statement,
                "scope": item.rule.scope,
                "why": item.reason,
                "confidence": round(item.rule.confidence, 3),
                "distance": round(item.rule.distance, 3),
            }
            for item in decision.selected
        ],
    }


@app.post("/github/webhook")
async def github_webhook(request: Request) -> dict:
    """Learn from a maintainer's comment on a pull request.

    Not rate limited by address: every delivery comes from GitHub's own hosts,
    so a per-client limit would either be pointless or would throttle GitHub
    itself. The signature is the gate, and the daily budget still applies.
    """
    settings = get_settings()
    raw = await request.body()

    try:
        verify_signature(
            settings.github_webhook_secret, raw, request.headers.get("X-Hub-Signature-256")
        )
    except SignatureInvalid as exc:
        # 401 with no detail. Telling an unauthenticated caller *why* their
        # signature failed helps them guess a valid one.
        log.warning("rejected webhook delivery: %s", exc)
        raise HTTPException(status_code=401, detail="invalid signature") from exc

    event = request.headers.get("X-GitHub-Event", "")
    payload = await request.json()

    opened = parse_pull_request(event, payload)
    if opened is not None:
        return await review_pull_request(settings, opened)

    parsed = parse_event(event, payload)
    if parsed is None:
        return {"status": "ignored", "reason": f"nothing to do for {event or 'unknown event'}"}

    instruction = extract_instruction(parsed["body"], settings.github_trigger)
    if not instruction:
        return {"status": "ignored", "reason": "not addressed to the app"}

    if not speaks_for_project(
        parsed["association"], parsed["author"], load_maintainers(settings.repo_slug)
    ):
        # Answered 200, not 403. The sender is a real contributor who wrote a
        # perfectly reasonable comment; there is nothing for them to fix, and a
        # failure would only make GitHub retry a delivery we will keep refusing.
        log.info(
            "ignoring %s (%s), who does not speak for the project",
            parsed["author"],
            parsed["association"],
        )
        return {"status": "ignored", "reason": "author does not have write access"}

    require_ready()
    reservation = await reserve_budget()

    before = deps.chat.spent
    try:
        learned = await teach(
            deps.engine,
            deps.provider,
            deps.chat,
            repo_id=deps.repo_id,
            parsed=parsed,
            instruction=instruction,
        )
    except AlreadyHandled:
        # 200, because GitHub is right to have retried and there is nothing to
        # fix. A failure here would only make it retry again.
        return {"status": "ignored", "reason": "this comment was already learned from"}
    except (BudgetExhausted, QuotaExhausted) as exc:
        raise HTTPException(status_code=503, detail="model budget spent") from exc
    finally:
        await settle_budget(reservation, deps.chat.spent - before)

    if learned is None:
        return {"status": "ignored", "reason": "no convention stated"}

    # Say so on the pull request. Writing to memory and answering 200 to GitHub
    # is invisible to the person who taught it, who is then asked to trust that
    # something happened.
    #
    # Deliberately after the rule is written and deliberately unable to fail the
    # request. The teaching is the durable part; the acknowledgement is a
    # courtesy, and losing it must not cost the rule or make GitHub retry a
    # delivery that already succeeded.
    posted = None
    if deps.github is not None and parsed.get("repo"):
        try:
            posted = await deps.github.comment(
                parsed["repo"], learned.pr_number, acknowledge(learned)
            )
        except GitHubError as exc:
            log.error("learned from %s but could not reply: %s", learned.author, exc)

    return {
        "status": "learned",
        "statement": learned.statement,
        "outcome": learned.outcome,
        "rule_id": learned.rule_id,
        "author": learned.author,
        "pr_number": learned.pr_number,
        "comment_url": posted,
    }


# Mounted last so it cannot shadow an endpoint. A single self-contained page
# with no build step: it deploys with the application, which matters when the
# target is one Lambda rather than a bucket and a CDN.
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
