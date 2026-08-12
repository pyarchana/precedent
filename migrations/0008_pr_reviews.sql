-- ---------------------------------------------------------------------------
-- 0008: what the agent has already said on a pull request, unprompted
--
-- Every other write path in this system is triggered by someone asking. This
-- one is not: the agent reads a pull request nobody addressed to it and decides
-- whether the project's conventions have anything to say about the files it
-- touches. That changes what a duplicate costs.
--
-- GitHub retries a delivery it believes timed out, and a pull request emits
-- further events every time it is pushed to, reopened or marked ready. Without
-- a record, each of those is a fresh comment on the same pull request, and an
-- agent that repeats itself five times on somebody's first contribution is a
-- worse outcome than one that never spoke.
--
-- So the record is the guard, not a log of it. The primary key is the
-- idempotency key, and the insert that claims it happens before the comment is
-- posted: losing a comment to a crash between the two is recoverable, posting
-- twice is not.
--
-- source_repo is part of the key because the pull request is not in the
-- repository the memory is about. Memory is pandas-dev/pandas; the app is
-- installed on whatever repository invited it, and #14 there has nothing to do
-- with #14 in pandas.
-- ---------------------------------------------------------------------------

CREATE TABLE pr_reviews (
    repo_id      UUID        NOT NULL REFERENCES repos (id),
    source_repo  STRING      NOT NULL,
    pr_number    INT8        NOT NULL,

    -- What was said and on what basis, so a comment can be traced back to the
    -- rules that produced it without re-running the retrieval.
    rule_ids     STRING[]    NOT NULL DEFAULT ARRAY[],
    comment_url  STRING,

    -- Set when the agent looked and decided to stay quiet. A silent decision is
    -- still a decision, and without it there is no way to tell "the agent found
    -- nothing" from "the webhook never arrived".
    silent_reason STRING,

    posted_at    TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT pk_pr_reviews PRIMARY KEY (repo_id, source_repo, pr_number)
);
