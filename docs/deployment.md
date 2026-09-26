# Deployment

Running it locally, then getting it onto Lambda. The short version is in the
[README](../README.md); this is everything the short version leaves out.

## Running locally

```bash
pip install -e ".[db,api]"
python -m precedent.db.migrate --create-db
uvicorn precedent.api.app:app --port 8000
```

Then open http://localhost:8000. `COCKROACH_DSN` and `OPENAI_API_KEY` come from
`.env`; see `.env.example`.

The page is one self-contained file with no build step, served by the same
process as the API, so the whole demo deploys as a single unit.

## Settings

Three settings exist because the deployed version is a public URL in front of a
paid model. `API_BUDGET_USD` is a ceiling across the process rather than per
request, since a per-request cap bounds nothing when anyone can send a thousand
requests; when it is reached the API returns 503 rather than falling back to an
unsourced answer. `API_RATE_LIMIT_PER_MINUTE` is the cruder guard.
`API_CORRECTIONS_ENABLED` turns off writes, because a public demo anyone can
rewrite is a demo that will be rewritten.

## Building for Lambda

```bash
python scripts/build_lambda.py
```

That writes `build/precedent-lambda.zip`, about 10 MB. The build pins wheels to
the Lambda runtime's platform and refuses to build from source, because three
dependencies ship compiled extensions and installing them normally on Windows
produces Windows binaries that import fine locally and fail in Lambda with "no
module named _asyncpg".

`tiktoken` is excluded deliberately. It is a compiled extension that fetches its
encoding table over the network the first time it is imported, which is a poor
thing to do on a cold start, and `embed/provider.py` already falls back to a
character bound without it. Requests are capped at 2,000 characters, well under
either limit, so nothing about the deployed behaviour changes.

## Configuring the function

In the Lambda console: create a function on **Python 3.12, x86_64**, upload the
zip, and set

| setting | value |
| --- | --- |
| Handler | `precedent.api.lambda_handler.handler` |
| Timeout | 30 seconds |
| Memory | 1024 MB |
| Environment | `COCKROACH_DSN`, `OPENAI_API_KEY` |

The default 3 second timeout will fail: a cold start is about 3 seconds before
any work happens. More memory also buys proportionally more CPU, which mostly
pays for itself in faster imports.

Then add a **Function URL** with auth type `NONE`.

Verify the handler before uploading anything:

```bash
python scripts/invoke_lambda_local.py
```

That calls `handler(event, context)` with the event shape a Function URL
actually sends, so routing and startup are exercised without reading CloudWatch
to discover a typo.

## What Lambda changed

Mangum runs the ASGI lifespan on **every invocation**, not once per execution
environment, and two things followed from that.

Rebuilding the engine per request opened a fresh connection pool to another
region every time: measured at 3.20s to serve a static page, against 0.12s once
the clients are built lazily and kept. The lifespan no longer tears anything
down when `AWS_LAMBDA_FUNCTION_NAME` is set.

More seriously, it reset the spend counter, so the budget ceiling restarted at
zero on every request and capped nothing. Even fixed, Lambda runs several
execution environments at once and each would believe it owned the whole budget.
The counter therefore lives in the database, keyed by day, so the cap is a daily
allowance every container shares. `/rules` stays readable once it is reached,
because refusing to answer is fine and refusing to show what memory holds is
not.

## Related

- [Failure modes](failure-modes.md)
- [Ingest and database](ingest.md)
