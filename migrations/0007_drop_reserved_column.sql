-- ---------------------------------------------------------------------------
-- 0007: remove a column that turned out to be the wrong idea
--
-- 0006 added `api_usage.reserved_usd`, caching the total of outstanding
-- reservations so the admission check could read one number. Keeping that cache
-- honest needed a sweep, because a container that dies mid-request never
-- releases its own reservation, and the sweep needed a CTE that CockroachDB
-- would not accept.
--
-- The failure was useful. A counter that needs a background job to stay correct
-- is a counter that will eventually be wrong, and this one guards money. So the
-- reservations table is now the only source of truth: outstanding spend is a
-- sum over live rows, and an abandoned reservation ages out of the window
-- instead of needing to be found and subtracted.
--
-- Dropped rather than left in place. An unused column that looks like it means
-- something is how the next reader ends up trusting it.
-- ---------------------------------------------------------------------------

ALTER TABLE api_usage DROP COLUMN IF EXISTS reserved_usd;
