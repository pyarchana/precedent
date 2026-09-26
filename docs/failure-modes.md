# What happens when things go wrong

Every row below is a failure that actually occurred, or a hole found while
looking for one. None of them are hypothetical.

| Failure | What handles it |
| --- | --- |
| A forged webhook claiming to be a maintainer | HMAC SHA-256 over the raw body, compared with `compare_digest` so a wrong guess cannot be timed. No signature, no write. |
| Anyone claiming to be a maintainer on the public API | Corrections are **off by default** there. The web form cannot verify a typed name, so it does not get to rewrite memory. Only signed GitHub deliveries can. |
| A retried webhook posting a duplicate comment | GitHub's timeout is 10s and the review path exceeds it on a cold start, so retries are routine. The claim row is taken **before** the work, so the retry finds the pull request already claimed. |
| Scraped review comments carrying prompt injection | Retrieved text is untrusted input. It sits inside an `<evidence>` delimiter that is stripped from the text itself, so it cannot close its own container. |
| A model returning `"false"`, `null`, or nonsense | Every response validates into an **inert** default: an answer becomes a refusal, a drafted rule becomes unusable, a contradiction verdict retires nothing. Garbled output is not evidence for writing to memory. |
| An answer citing something it did not read | Citations are verified against the retrieved evidence before release. An answer whose citations do not check out is discarded rather than shown. |
| The public demo draining an unreplenishable API credit | A daily ceiling held in the database, claimed as a reservation in the same statement that checks for room. Measured: ten concurrent requests at $0.90 against a $1.00 limit previously all passed, and now exactly one does. |
| One caller hammering the endpoint | Per-address rate limiting, six times stricter on the correction path. |
| A misconfigured deployment | `/health` answers even when startup failed, and names the missing environment variable rather than returning an opaque 502. |
| A dependency missing from the Lambda package | CI unzips the built artefact and imports the handler from it. Added after a deployment 502'd on `No module named 'yaml'` that every other check had passed. |
| A maintainer who has since left the project | `author_association` reflects permissions held now, so it reports 74,077 of `jreback`'s comments as CONTRIBUTOR. Maintainer status is derived from review behaviour instead. |

## Not covered

Observability is structured logs and `/health`, with no metrics or alerting.
Backups are whatever CockroachDB Cloud does by default. Both are where I would
start if this ran for anyone but me.

## Related

- [Architecture](architecture.md), including the correction semantics several of these protect
- [Deployment](deployment.md)
