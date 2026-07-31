-- Embedding storage support: a content-addressed cache, and the vector indexes.

-- ---------------------------------------------------------------------------
-- embedding_cache
--
-- Keyed by a hash of the text rather than by comment id, for two reasons.
-- Review corpora are extremely repetitive ("LGTM", "Thanks!", CI templates),
-- so identical bodies are paid for once instead of thousands of times. And a
-- crash between the API responding and the row being written does not cost
-- money on the retry, because the second attempt finds the vector here.
-- ---------------------------------------------------------------------------

CREATE TABLE embedding_cache (
    model        STRING       NOT NULL,
    content_hash STRING       NOT NULL,
    embedding    VECTOR(1536) NOT NULL,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT now(),

    CONSTRAINT pk_embedding_cache PRIMARY KEY (model, content_hash)
);

-- ---------------------------------------------------------------------------
-- Vector indexes
--
-- [CRDB] Built on (repo_id, embedding) rather than (embedding) alone. The
-- leading column partitions the index by tenant, so a search inside one
-- repository never traverses another's vectors. Postgres with pgvector has no
-- equivalent prefix support and would need a partial index per tenant.
--
-- Created while the columns are still empty so the index is maintained
-- incrementally by the backfill, rather than being built in one pass over
-- millions of vectors afterwards.
-- ---------------------------------------------------------------------------

CREATE VECTOR INDEX idx_rc_embedding ON review_comments (repo_id, embedding);

CREATE VECTOR INDEX idx_rules_embedding ON rules (repo_id, embedding);
