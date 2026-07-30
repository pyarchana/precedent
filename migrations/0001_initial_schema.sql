-- Precedent initial schema.
--
-- Multi-tenant on repo_id: every table that holds repository knowledge carries
-- repo_id as the leading primary key column, so one repo's data is contiguous in
-- the keyspace and a tenant scan never touches another tenant's ranges.
--
-- CockroachDB-specific choices are marked [CRDB] with the reason. On plain
-- Postgres most of them would be written differently.

-- ---------------------------------------------------------------------------
-- Enums
-- ---------------------------------------------------------------------------

CREATE TYPE comment_kind AS ENUM (
    'review_thread',        -- inline comment on a diff
    'review_summary',       -- top-level review body (approve / request changes)
    'issue_comment',        -- conversation tab comment on a PR
    'pr_body',              -- the PR description itself
    'maintainer_correction' -- a maintainer correcting one of our answers
);

CREATE TYPE rule_scope AS ENUM (
    'repo',        -- applies everywhere
    'directory',   -- applies under a path prefix
    'file',        -- applies to one file
    'api',         -- about a specific public API
    'testing',
    'docs',
    'style',
    'process'      -- how to contribute, not what to write
);

CREATE TYPE rule_status AS ENUM ('active', 'superseded', 'retired');

-- ---------------------------------------------------------------------------
-- repos
-- ---------------------------------------------------------------------------

CREATE TABLE repos (
    id             UUID        NOT NULL DEFAULT gen_random_uuid(),
    owner          STRING      NOT NULL,
    name           STRING      NOT NULL,
    default_branch STRING,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT pk_repos PRIMARY KEY (id),
    CONSTRAINT uq_repos_slug UNIQUE (owner, name)
);

-- ---------------------------------------------------------------------------
-- review_comments: episodic memory
--
-- One row per thing a human said on a pull request. This is the raw material
-- for rule extraction and the citation target for every answer the agent gives.
-- ---------------------------------------------------------------------------

CREATE TABLE review_comments (
    repo_id            UUID        NOT NULL REFERENCES repos (id),
    id                 UUID        NOT NULL DEFAULT gen_random_uuid(),

    -- GitHub's own node id. The transform upserts on this, which is what makes
    -- reloading from staged JSON idempotent.
    github_node_id     STRING      NOT NULL,

    kind               comment_kind NOT NULL,
    pr_number          INT8        NOT NULL,
    thread_id          STRING,     -- groups inline comments into a conversation
    in_reply_to        STRING,

    author             STRING,     -- null when the GitHub account is deleted
    author_association STRING      NOT NULL,
    is_maintainer      BOOL        NOT NULL DEFAULT false,

    file_path          STRING,
    line               INT8,
    diff_hunk          STRING,

    body               STRING      NOT NULL,
    url                STRING,

    created_at         TIMESTAMPTZ NOT NULL,   -- when it was said on GitHub
    ingested_at        TIMESTAMPTZ NOT NULL DEFAULT now(),

    embedding          VECTOR(1536),
    embedding_model    STRING,
    embedded_at        TIMESTAMPTZ,

    CONSTRAINT pk_review_comments PRIMARY KEY (repo_id, id),
    CONSTRAINT uq_review_comments_node UNIQUE (repo_id, github_node_id),

    -- [CRDB] Column families. The embedding backfill updates only the embedding
    -- columns; without families every such write would rewrite the body and the
    -- diff hunk too, which is the bulk of the row. Postgres has no equivalent
    -- and would rewrite the whole tuple regardless.
    FAMILY f_meta (
        repo_id, id, github_node_id, kind, pr_number, thread_id, in_reply_to,
        author, author_association, is_maintainer, file_path, line, url,
        created_at, ingested_at
    ),
    FAMILY f_text (body, diff_hunk),
    FAMILY f_embedding (embedding, embedding_model, embedded_at)
);

-- The transform's hot path: find work still needing an embedding.
-- Partial index so it stays small once the backfill is done, instead of
-- indexing every row in the corpus forever.
CREATE INDEX idx_rc_unembedded
    ON review_comments (repo_id, id)
    WHERE embedding IS NULL;

-- Pull a whole PR conversation back for citation rendering.
CREATE INDEX idx_rc_by_pr
    ON review_comments (repo_id, pr_number, created_at)
    STORING (author, is_maintainer, file_path, kind);

-- "What has this repo said about this file or directory?"
CREATE INDEX idx_rc_by_path
    ON review_comments (repo_id, file_path)
    WHERE file_path IS NOT NULL;

-- Entity memory: everything one person has said in this repo.
CREATE INDEX idx_rc_by_author
    ON review_comments (repo_id, author, created_at DESC);

-- [CRDB] Hash-sharded. created_at is monotonically increasing, so an ordinary
-- index on it sends every insert to the same range and that range becomes the
-- write bottleneck. Sharding spreads inserts across 8 ranges at the cost of
-- turning an ordered scan into 8 scans that get merged. Postgres has a single
-- writer per index and no range splitting, so this problem does not exist there
-- and a plain btree would be the right answer.
CREATE INDEX idx_rc_recent
    ON review_comments (repo_id, created_at DESC)
    USING HASH WITH (bucket_count = 8);

-- ---------------------------------------------------------------------------
-- rules: semantic memory
--
-- Distilled conventions. Rules are never hard-deleted; a rule that stops being
-- true is superseded and keeps its history, because the history is what lets
-- the agent explain why an answer changed.
-- ---------------------------------------------------------------------------

CREATE TABLE rules (
    repo_id                   UUID        NOT NULL REFERENCES repos (id),
    id                        UUID        NOT NULL DEFAULT gen_random_uuid(),

    statement                 STRING      NOT NULL,
    rationale                 STRING,

    scope                     rule_scope  NOT NULL,
    scope_pattern             STRING,     -- LIKE pattern, e.g. 'pandas/core/arrays/%'

    confidence                FLOAT8      NOT NULL DEFAULT 0,
    evidence_count            INT8        NOT NULL DEFAULT 0,
    maintainer_evidence_count INT8        NOT NULL DEFAULT 0,

    status                    rule_status NOT NULL DEFAULT 'active',
    superseded_by             UUID,
    superseded_at             TIMESTAMPTZ,
    supersession_reason       STRING,

    embedding                 VECTOR(1536),
    embedding_model           STRING,

    first_evidence_at         TIMESTAMPTZ,  -- oldest supporting comment
    last_evidence_at          TIMESTAMPTZ,  -- newest supporting comment
    created_at                TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at                TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT pk_rules PRIMARY KEY (repo_id, id),

    -- Composite self-reference including repo_id. A rule can only ever be
    -- superseded by a rule belonging to the same repository, enforced by the
    -- database rather than by application code.
    CONSTRAINT fk_rules_superseded_by
        FOREIGN KEY (repo_id, superseded_by) REFERENCES rules (repo_id, id),

    CONSTRAINT ck_rules_confidence CHECK (confidence >= 0 AND confidence <= 1),
    CONSTRAINT ck_rules_superseded CHECK (
        (status = 'superseded') = (superseded_by IS NOT NULL)
    ),
    CONSTRAINT ck_rules_scope_pattern CHECK (
        scope NOT IN ('directory', 'file') OR scope_pattern IS NOT NULL
    )
);

-- The answer path reads only live rules, best first.
CREATE INDEX idx_rules_active
    ON rules (repo_id, confidence DESC)
    STORING (statement, scope, scope_pattern, evidence_count)
    WHERE status = 'active';

CREATE INDEX idx_rules_by_scope
    ON rules (repo_id, scope, confidence DESC)
    WHERE status = 'active';

-- Walk a supersession chain backwards to explain why an answer changed.
CREATE INDEX idx_rules_superseded_by
    ON rules (repo_id, superseded_by)
    WHERE superseded_by IS NOT NULL;

-- ---------------------------------------------------------------------------
-- rule_evidence: provenance
--
-- Which comments each rule was learned from. Without this the agent cannot
-- cite, and an uncitable answer is indistinguishable from the base model
-- guessing.
-- ---------------------------------------------------------------------------

CREATE TABLE rule_evidence (
    repo_id       UUID        NOT NULL,
    rule_id       UUID        NOT NULL,
    comment_id    UUID        NOT NULL,

    weight        FLOAT8      NOT NULL DEFAULT 1.0,
    is_maintainer BOOL        NOT NULL DEFAULT false,
    is_correction BOOL        NOT NULL DEFAULT false,
    extracted_at  TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Leading repo_id then rule_id means a rule's evidence is contiguous, so
    -- rendering citations for one rule is a single short range scan.
    CONSTRAINT pk_rule_evidence PRIMARY KEY (repo_id, rule_id, comment_id),

    CONSTRAINT fk_evidence_rule
        FOREIGN KEY (repo_id, rule_id) REFERENCES rules (repo_id, id)
        ON DELETE CASCADE,
    CONSTRAINT fk_evidence_comment
        FOREIGN KEY (repo_id, comment_id) REFERENCES review_comments (repo_id, id)
        ON DELETE CASCADE
);

-- The reverse question: which rules did this comment teach us?
CREATE INDEX idx_evidence_by_comment
    ON rule_evidence (repo_id, comment_id);

-- ---------------------------------------------------------------------------
-- contributors: entity memory
-- ---------------------------------------------------------------------------

CREATE TABLE contributors (
    repo_id       UUID        NOT NULL REFERENCES repos (id),
    login         STRING      NOT NULL,

    is_maintainer BOOL        NOT NULL DEFAULT false,
    pr_count      INT8        NOT NULL DEFAULT 0,
    comment_count INT8        NOT NULL DEFAULT 0,

    -- Directory prefixes this person has actually touched, most recent first.
    areas_touched STRING[]    NOT NULL DEFAULT ARRAY[]::STRING[],

    first_seen_at TIMESTAMPTZ,
    last_seen_at  TIMESTAMPTZ,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT pk_contributors PRIMARY KEY (repo_id, login)
);

-- ---------------------------------------------------------------------------
-- sessions and turns: working memory
-- ---------------------------------------------------------------------------

CREATE TABLE sessions (
    repo_id           UUID        NOT NULL REFERENCES repos (id),
    id                UUID        NOT NULL DEFAULT gen_random_uuid(),
    contributor_login STRING,
    started_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_active_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    state             JSONB       NOT NULL DEFAULT '{}'::JSONB,

    CONSTRAINT pk_sessions PRIMARY KEY (repo_id, id)
);

CREATE INDEX idx_sessions_by_contributor
    ON sessions (repo_id, contributor_login, last_active_at DESC)
    WHERE contributor_login IS NOT NULL;

CREATE TABLE session_turns (
    repo_id     UUID        NOT NULL,
    session_id  UUID        NOT NULL,
    turn_number INT8        NOT NULL,

    question    STRING      NOT NULL,
    answer      STRING,

    -- What the answer was built from. Denormalised deliberately: a correction
    -- arriving weeks later must be able to see exactly which rules were used,
    -- even if those rules have since been superseded.
    cited_rule_ids    UUID[] NOT NULL DEFAULT ARRAY[]::UUID[],
    cited_comment_ids UUID[] NOT NULL DEFAULT ARRAY[]::UUID[],
    answered_from_memory BOOL NOT NULL DEFAULT true,

    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT pk_session_turns PRIMARY KEY (repo_id, session_id, turn_number),
    CONSTRAINT fk_turns_session
        FOREIGN KEY (repo_id, session_id) REFERENCES sessions (repo_id, id)
        ON DELETE CASCADE
);

-- Rules this person has already been told, so the agent does not re-explain.
CREATE TABLE contributor_told (
    repo_id    UUID        NOT NULL,
    login      STRING      NOT NULL,
    rule_id    UUID        NOT NULL,
    session_id UUID,
    told_at    TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT pk_contributor_told PRIMARY KEY (repo_id, login, rule_id),
    CONSTRAINT fk_told_contributor
        FOREIGN KEY (repo_id, login) REFERENCES contributors (repo_id, login)
        ON DELETE CASCADE,
    CONSTRAINT fk_told_rule
        FOREIGN KEY (repo_id, rule_id) REFERENCES rules (repo_id, id)
        ON DELETE CASCADE
);

-- ---------------------------------------------------------------------------
-- corrections
--
-- The correction text itself is stored as a review_comment of kind
-- 'maintainer_correction', so it flows through embedding and rule extraction
-- on exactly the same path as anything a maintainer said on a real PR. This
-- table only records what the correction was aimed at.
-- ---------------------------------------------------------------------------

CREATE TABLE corrections (
    repo_id            UUID        NOT NULL,
    id                 UUID        NOT NULL DEFAULT gen_random_uuid(),

    session_id         UUID        NOT NULL,
    turn_number        INT8        NOT NULL,
    comment_id         UUID        NOT NULL,  -- the stored correction text
    corrected_rule_id  UUID,                  -- null if the answer invented it
    replacement_rule_id UUID,

    maintainer_login   STRING      NOT NULL,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    applied_at         TIMESTAMPTZ,

    CONSTRAINT pk_corrections PRIMARY KEY (repo_id, id),
    CONSTRAINT fk_corrections_turn
        FOREIGN KEY (repo_id, session_id, turn_number)
        REFERENCES session_turns (repo_id, session_id, turn_number)
        ON DELETE CASCADE,
    CONSTRAINT fk_corrections_comment
        FOREIGN KEY (repo_id, comment_id) REFERENCES review_comments (repo_id, id)
);

CREATE INDEX idx_corrections_pending
    ON corrections (repo_id, created_at)
    WHERE applied_at IS NULL;

-- ---------------------------------------------------------------------------
-- ingest_rejects
--
-- Day 4 filters bot authors out of the corpus. Recording what was dropped and
-- why keeps the filter auditable, and makes it cheap to reverse a filtering
-- decision that turns out to be wrong.
-- ---------------------------------------------------------------------------

CREATE TABLE ingest_rejects (
    repo_id        UUID        NOT NULL REFERENCES repos (id),
    github_node_id STRING      NOT NULL,
    pr_number      INT8,
    author         STRING,
    reason         STRING      NOT NULL,
    rejected_at    TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT pk_ingest_rejects PRIMARY KEY (repo_id, github_node_id)
);
