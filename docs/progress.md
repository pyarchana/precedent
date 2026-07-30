# Progress

Running state of the build. Updated at the end of each working session so the
next one starts without re-deriving anything.

## Status as of Jul 30 2026, end of day

Plan day 5 by the calendar, day 1 by the repo: the project started four days
late and is now roughly one day behind.

### Done

- **Ingest (plan day 3).** GraphQL walk of pandas-dev/pandas, newest first.
  Raw responses staged as gzipped JSON before parsing; cursor checkpointed
  after every page. Stopped at page 1763, 17,630 of 37,601 PRs.
- **Schema (plan day 2).** `migrations/0001_initial_schema.sql`, eleven tables,
  applied and verified against a local CockroachDB v26.2.4 node in Docker.
- **Transform (plan day 4).** Staged JSON to `review_comments`, idempotent on
  GitHub node ids. 91,297 comments loaded from the 16,100 PRs staged at the
  time, 59,827 of them from maintainers, 3,078 contributors derived.
  Idempotency verified by re-running 200 pages with no change in row count.
- **Async stack decision (plan day 1).** See "Decisions" below.

### Deliberately skipped so far

Plan day 1's cloud provisioning. CockroachDB Cloud, the S3 bucket and the AWS
CLI do not exist yet. The ingest was put first because it is the only
wall-clock-bound task, and it was made to run against local disk so it did not
have to wait for an AWS account.

### Next

1. Embeddings and vector index (plan day 5). Blocked on choosing a provider.
2. Evaluation set (plan day 6). Not blocked, and worth doing before the agent.
3. Sync staged JSON to S3 once a bucket exists. Not on the critical path.
4. Targeted refetch of PRs whose nested connections truncated. About 1.7% of
   PRs; the numbers accumulate in `checkpoint.json` under `truncated_prs`.

## Decisions

- **Driver: `sqlalchemy-cockroachdb`, not stock `postgresql+asyncpg`.** The
  Postgres dialect parses `version()` with a Postgres-shaped regex during
  connection setup and raises AssertionError on CockroachDB's version string,
  so it fails at connect time rather than degrading. `cockroachdb+asyncpg`
  passes all of `scripts/check_async_stack.py`.
- **Vectors are strings over the wire.** `VECTOR` columns come back as text and
  must be rendered as `[1,2,3]` on write. Worth a `TypeDecorator` before the
  embedding pipeline is written rather than after.
- **`CONTRIBUTOR` is not a maintainer.** It only means the person has had a PR
  merged. Only OWNER, MEMBER and COLLABORATOR set `is_maintainer`.
- **Corrections are stored as `review_comments`** with kind
  `maintainer_correction`, so they flow through embedding and rule extraction
  on the same path as anything said on a real PR.
- **Re-running the transform clears an embedding only when the body changed**,
  so a re-run does not invalidate an expensive backfill.

## Open questions

- **Embedding provider.** Anthropic has no embeddings API, so
  `ANTHROPIC_API_KEY` does not cover this. The schema is `VECTOR(1536)`, which
  matches OpenAI `text-embedding-3-small`. Changing the dimension after the
  backfill means a table rewrite, so decide before embedding, not after.

## Local development

The local CockroachDB node runs in Docker with an in-memory store, so its data
is lost when the container stops. Re-apply migrations and re-run the transform
to rebuild; both are idempotent and the transform takes about six minutes.

```bash
docker start crdb-precedent
```
