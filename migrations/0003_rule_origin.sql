-- ---------------------------------------------------------------------------
-- 0003: where a rule came from
--
-- Every rule so far was inferred. A cluster of review comments went to a model
-- and a convention came back. A correction is a different kind of thing: a
-- maintainer read one specific wrong answer and stated, deliberately, what the
-- project actually does.
--
-- The confidence model cannot tell those apart, and scoring them on the same
-- terms gets the answer backwards. It weights independent voices, distinct
-- pull requests, and how long a convention has persisted. A correction has one
-- author, one occasion, and no history, so it lands near the floor. The agent
-- would then present a maintainer's own words as "weakly evidenced" while
-- treating a pattern guessed from three pull requests in 2016 as settled.
--
-- Recording the origin is what lets the two be scored differently.
-- ---------------------------------------------------------------------------

ALTER TABLE rules ADD COLUMN IF NOT EXISTS origin STRING NOT NULL DEFAULT 'extracted';

ALTER TABLE rules ADD CONSTRAINT ck_rules_origin
    CHECK (origin IN ('extracted', 'correction'));

-- Corrections are what a maintainer will want to review and what a demo shows,
-- so they need to be findable without reading every rule in the repository.
CREATE INDEX idx_rules_corrections
    ON rules (repo_id, updated_at DESC)
    WHERE origin = 'correction';
