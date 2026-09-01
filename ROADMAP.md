# Roadmap

What is actually next, why, and what is blocked on money rather than on ideas.

This runs on a CockroachDB Basic cluster capped at the free allowance and a
prepaid OpenAI credit with about $3 left. That constraint decides the ordering
more than preference does: anything requiring a re-ingest or a large evaluation
is real work that cannot be done yet, and saying so is more useful than listing
it as though it were imminent.

Issues and pull requests welcome on any of these.

## Next

Small, decided, and costs nothing to run.

**Recover the fields that parsed when one field does not.** `extract/schemas.py`
coerces `null` to `""` for `str` fields, but `bool` and `Literal` fields still
fail the whole object, so `{"answered": null, "answer": "..."}` discards a
perfectly good answer. `ValidationError` names the failing locations, so the fix
is to drop those and revalidate rather than to write a validator per field.
Raised by Vinh Nguyen on the dev.to writeup.

**Count parse failures separately from refusals.** A schema failure and a
genuine "no convention here" both surface as `200` with
`{"status": "ignored", "reason": "no convention stated"}`. They are
indistinguishable in logs and in metrics, which is why the null-rationale bug
took an afternoon to find rather than a minute. Raised by jkming, same thread.

**Validate every `scope_pattern` against the real file list at ingest.** The
selectivity check exists but runs at query time. Running it once at ingest turns
"68 of 79 patterns match zero files" from an invisible condition into a loud
one, with no model call involved. Also jkming.

**Add `mypy --strict` to CI.** Async database code, Pydantic models and webhook
handlers are exactly where static analysis pays, and the project currently runs
`ruff` alone.

## After that

Needs design, not just typing.

**Rebuild `idx_rc_embedding`.** The vector index on `review_comments` was
dropped for the embedding backfill, which is a real speedup (9 rows/sec against
57.8), and the rebuild never completed on a free-tier cluster. Comment search
therefore full-scans 86,000 rows, which is both slow and the most expensive
operation against the request unit allowance. Rebuilding it costs a large slice
of a month's allowance up front, so it needs to be done deliberately rather than
opportunistically, and the ingest should rebuild it automatically afterwards
rather than leaving it to be forgotten again.

**Say which line violates a convention, not just which conventions apply.** The
agent names what the project has settled and links where; it does not read the
diff. Doing so is the difference between a bot that recites and one that
reviews. It is deliberately not first on this list: being wrong about somebody's
diff costs more than being unhelpful about it, so this needs an evaluation set
of its own before any of it ships.

**Fix the over-cautious refusals.** Three of the evaluation's answerable
questions were refused when the evidence supported an answer. That is a
retrieval and prompting problem on questions already written down, so it is
measurable without spending anything new.

**Decay confidence on rules nobody has reaffirmed.** Every rule carries
`last_evidence_at`. A convention last mentioned in 2014 and never since is
probably still true, but it should not outrank one a maintainer stated last
month. Nothing currently uses that column.

**Extract incrementally rather than in batch.** Extraction runs over the whole
history at once. It should run as review happens, so a convention settled this
morning is citable this afternoon.

## Blocked on budget

Not forgotten, just not fundable right now.

**A serious evaluation.** 29 questions is a smoke test. Several hundred, across
several repositories, is evidence. The judging model calls alone put this beyond
the current credit.

**Validation on repositories other than pandas.** The schema is multi-tenant
from the first migration and has never been tested with a second tenant. A small
repository with 200 review comments would probably break assumptions that 86,000
comments hid.

**Integration tests against a real cluster in CI.** The suite never touches
CockroachDB, which is documented in the workflow and is a known blind spot. A
containerised cluster in CI is the fix and it is not free to run on every push.

**A provider abstraction.** Hard-wired to OpenAI. Local models through Ollama,
plus Anthropic and Google, would matter to anyone who cannot send their review
history to a third party. Straightforward work, but it needs testing against
each provider to be worth claiming.

## Not planned

**Replacing linters.** Deterministic rules belong in `ruff`, `mypy` and CI. This
is for the conventions that only exist in people's heads and in review threads.

**Answering from the base model when memory is empty.** Refusing is the feature.
An agent that falls back to general knowledge proves nothing about its memory
and cannot be checked.
