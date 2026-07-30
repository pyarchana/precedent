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

## License

MIT. See [LICENSE](LICENSE).
