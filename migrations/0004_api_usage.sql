-- ---------------------------------------------------------------------------
-- 0004: what the public demo has spent
--
-- The API tracks spend in the model client, which works under uvicorn where
-- one process serves every request. It does not work under Lambda, and the way
-- it fails is expensive.
--
-- Mangum runs the ASGI lifespan on every invocation, so the client is rebuilt
-- per request and its counter restarts at zero. Even with that fixed, the
-- counter lives in one execution environment, and Lambda will happily run
-- several at once, each believing it has the whole budget.
--
-- A ceiling that resets is not a ceiling. This project runs on a single
-- unreplenishable top-up, so the number has to live somewhere every container
-- can see, which is the database that is already the memory.
--
-- Keyed by day so the cap is a daily allowance rather than a lifetime one: a
-- demo that stops answering forever the first time someone hammers it is worse
-- for judging than one that recovers tomorrow.
-- ---------------------------------------------------------------------------

CREATE TABLE api_usage (
    day        DATE        NOT NULL,
    calls      INT8        NOT NULL DEFAULT 0,
    spend_usd  FLOAT8      NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT pk_api_usage PRIMARY KEY (day)
);
