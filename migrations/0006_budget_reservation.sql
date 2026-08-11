-- ---------------------------------------------------------------------------
-- 0006: make the daily budget a real ceiling
--
-- 0004 moved spend accounting into the database, which fixed the counter
-- resetting on every Lambda invocation. It did not make the ceiling binding,
-- because the sequence was:
--
--     read the total  ->  call the model  ->  add what it cost
--
-- Two containers reading $0.90 against a $1.00 limit both pass the check, both
-- spend $0.20, and the day ends at $1.30. Nothing is over-spent by much, but a
-- limit that can be exceeded is not a limit, and the comment in the API
-- claimed it was the durable guarantee.
--
-- Reserving first closes it. A request takes out a reservation for the most it
-- could cost, in the same statement that checks there is room, so the check
-- and the claim cannot be separated by another container. The reservation is
-- released and replaced by the real cost once the call returns.
--
-- Reservations are also stamped, because a container that dies mid-request
-- never releases its own, and without expiry those leaks would slowly consume
-- the day's allowance.
-- ---------------------------------------------------------------------------

ALTER TABLE api_usage ADD COLUMN IF NOT EXISTS reserved_usd FLOAT8 NOT NULL DEFAULT 0;

-- One row per in-flight request. Small, short-lived, and swept on read.
CREATE TABLE api_reservations (
    id         UUID        NOT NULL DEFAULT gen_random_uuid(),
    day        DATE        NOT NULL,
    amount_usd FLOAT8      NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT pk_api_reservations PRIMARY KEY (id)
);

CREATE INDEX idx_api_reservations_stale ON api_reservations (created_at);
