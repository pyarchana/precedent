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

## Running it

```bash
pip install -e ".[db,api]"
python -m precedent.db.migrate --create-db
uvicorn precedent.api.app:app --port 8000
```

Then open http://localhost:8000. `COCKROACH_DSN` and `OPENAI_API_KEY` come from
`.env`; see `.env.example`.

The page is one self-contained file with no build step, served by the same
process as the API, so the whole demo deploys as a single unit.

Three settings exist because the deployed version is a public URL in front of a
paid model. `API_BUDGET_USD` is a ceiling across the process rather than per
request, since a per-request cap bounds nothing when anyone can send a thousand
requests; when it is reached the API returns 503 rather than falling back to an
unsourced answer. `API_RATE_LIMIT_PER_MINUTE` is the cruder guard.
`API_CORRECTIONS_ENABLED` turns off writes, because a public demo anyone can
rewrite is a demo that will be rewritten.

## Asking, and correcting

Retrieval with citations is a search engine with good manners. What makes this a
memory is that a maintainer can correct an answer once, and the next contributor
to ask gets the corrected one.

```
$ python -m precedent.agent.ask "Where do I put the GitHub issue number in a test?"
You should add the GitHub issue number as a comment at the top of each test in
the format # GH<issue number> [PR #65052].

to correct this answer: python -m precedent.agent.correct 59d1f3b0 1 "..."
```

```
$ python -m precedent.agent.correct 59d1f3b0 1 \
    "Not quite. The number goes next to the specific assertion that covers the
     issue, not at the top of the test, and the format is # GH#12345." --as pyarchana

retired: Add the GitHub issue number as a comment at the top of each test...
reason: The existing rule requires the issue number at the top of each test,
while the new rule specifies it must be next to specific assertions, making it
impossible to follow both simultaneously.
```

```
$ python -m precedent.agent.ask "Where do I put the GitHub issue number in a test?"
Add the GitHub issue number as a comment next to the specific assertion or test
case that covers the issue in the format # GH#<issue number>
[correction by pyarchana, 2026-08-07].
```

Four things about that are deliberate.

**The correction is aimed at the rule the answer actually used**, not at the
rule nearest to the correction's wording. Every answer records the rule ids it
was built from, so a correction arriving weeks later can still see them.
Nearest-neighbour would occasionally retire a different rule than the maintainer
meant, which is worse than doing nothing.

**A model decides whether it is a contradiction.** Embeddings put opposites
close together: "use single quotes" and "use double quotes" sit nearer to each
other than two genuine duplicates do. An earlier version merged on distance
alone, and feeding it a reversal made the reversal *further evidence for* the
thing it reversed. A correction that strengthens the error it corrects is the
one failure this system cannot have.

**A correction that agrees with the cited rule becomes evidence for it**, not a
second copy of it. That case means the rule was right and the answer misused it.

**Nothing is deleted.** The retired rule keeps its evidence, its confidence and
the reason it was replaced, because "we used to say X, then this happened" is
what makes the memory explicable rather than merely current.

Corrections are stored as `review_comments` of kind `maintainer_correction`, so
they are embedded, retrieved and cited on the same path as anything said on a
real pull request. They are never rendered as a PR citation, and every citation
in an answer, PR or correction, is verified against what was actually retrieved
before the answer is shown.

### Scoring a correction

Confidence is built from independent voices, distinct pull requests, persistence
and recency. A correction has one author, one occasion and no history, so on
those terms it scores near the floor, and the agent would present a maintainer's
own words as "weakly evidenced" while treating a pattern inferred from three
pull requests in 2016 as settled.

Rules therefore record their `origin`, and a correction gets a floor of 0.85
rather than a score. The floor sits below what a genuinely well-attested
convention reaches, so a correction outranks the rule it replaced without
outranking the whole corpus.

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

### Bulk loading embeddings

Drop the vector index before a large backfill and rebuild it afterwards. Maintaining
it incrementally across hundreds of thousands of single-row updates costs far more
than building it once at the end. Measured on this corpus, embedding 1,024 comments
into a 316,000 row table:

| | rows/sec |
| --- | --- |
| Vector index present, 3 requests in flight | 9.0 |
| Vector index dropped, 3 requests in flight | 25.2 |
| Vector index dropped, 6 requests in flight | 57.8 |

Between them that is the difference between nine hours and eighty minutes.

```bash
docker exec crdb-precedent ./cockroach sql --insecure --database=precedent --execute "DROP INDEX review_comments@idx_rc_embedding;"
```

```bash
docker exec crdb-precedent ./cockroach sql --insecure --database=precedent --execute "CREATE VECTOR INDEX idx_rc_embedding ON review_comments (repo_id, embedding);"
```

The index definition lives in migration 0002. Dropping it for a backfill is an
operational step, not a schema change, so it is done directly rather than by
adding a migration.

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
