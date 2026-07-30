# Precedent

An agentic memory layer for open source maintainers, built on CockroachDB.

Large projects answer the same contributor questions forever. The answers exist,
buried in review threads going back a decade, but they are not retrievable, and a
maintainer's correction today does nothing for the person who asks the same thing
next month. Precedent reads a repository's entire review history, distils the durable
conventions out of it, answers contributor questions with citations back to the
specific PRs the claim came from, and lets a maintainer correct an answer once so that
every later answer reflects the correction.

Target repository: [pandas-dev/pandas](https://github.com/pandas-dev/pandas).

> Status: in development. This README grows as the system does.

## Memory model

| Store | Table | What it holds |
| --- | --- | --- |
| Episodic | `review_comments` | Individual review comments, embedded for semantic search |
| Semantic | `rules` | Distilled repo conventions, with confidence and supersession history |
| Provenance | `rule_evidence` | Which comments each rule was learned from |
| Entity | `contributors` | Per-repo contributor state: what they have touched, what they have been told |
| Working | `sessions` | Current conversation state |

Every table is keyed on `repo_id`; the system is multi-tenant from the schema up.

## Running the ingest

The ingest stages raw GitHub GraphQL responses to disk before anything parses them,
so the transform can be replayed against a changed schema without re-hitting the API.
It needs no database and no S3 bucket to start.

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
cp .env.example .env   # optional; falls back to `gh auth token`
```

Smoke test (two pages, then stop):

```bash
python -m precedent.ingest.run --max-pages 2 -v
```

Full run. It resumes from its checkpoint, so killing it is safe:

```bash
python -m precedent.ingest.run --log-file logs/ingest.log
```

Raw pages land in `data/raw/<owner>__<repo>/pr_pages/page_NNNNNN.json.gz`, with
resume state in `checkpoint.json` alongside them. Neither is committed.

## Database

A local single-node cluster is enough for schema work:

```bash
docker run -d --name crdb-precedent -p 26257:26257 -p 8081:8080 cockroachdb/cockroach:latest start-single-node --insecure --store=type=mem,size=2GiB
```

Apply the schema:

```bash
python -m precedent.db.migrate --dsn "postgresql://root@localhost:26257/precedent?sslmode=disable" --create-db
```

Confirm the async stack works against whatever cluster you pointed at:

```bash
python scripts/check_async_stack.py --dsn "postgresql://root@localhost:26257/precedent?sslmode=disable"
```

### Driver notes

Use `sqlalchemy-cockroachdb`, not the stock `postgresql+asyncpg` dialect. SQLAlchemy's
Postgres dialect parses `version()` with a Postgres-shaped regex during connection
setup and raises `AssertionError` on `CockroachDB CCL v26.2.4 ...`, so nothing works at
all, not merely version-gated features. `cockroachdb+asyncpg` is a genuine async
dialect and passes the same checks.

`VECTOR` columns come back from the driver as strings, not sequences, and have to be
parsed on read and rendered as `[1,2,3]` on write. In `text()` queries, cast with
`CAST(:v AS VECTOR(n))` rather than `:v::VECTOR`, because the bind-parameter parser
reads the second colon as the start of another parameter.

## License

MIT. See [LICENSE](LICENSE).
