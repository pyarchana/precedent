-- ---------------------------------------------------------------------------
-- 0005: conventions a maintainer stated rather than corrected
--
-- Corrections arrive attached to a wrong answer. A maintainer writing
-- "@precedent the whatsnew note goes in the next release file" on a pull
-- request has not corrected anything: there is no prior answer, no cited rule,
-- nothing to retire. They are stating a convention directly.
--
-- Both are authoritative in the same way and both earn the same confidence
-- floor, since a maintainer saying what the project does is not weak evidence
-- whichever way it reaches us. But a correction retires something and a
-- teaching adds something, and recording them under one origin would make the
-- supersession history claim retirements that never happened.
--
-- The CHECK constraint has to be replaced rather than extended; CockroachDB
-- has no ALTER CONSTRAINT for this.
-- ---------------------------------------------------------------------------

ALTER TABLE rules DROP CONSTRAINT ck_rules_origin;

ALTER TABLE rules ADD CONSTRAINT ck_rules_origin
    CHECK (origin IN ('extracted', 'correction', 'taught'));
