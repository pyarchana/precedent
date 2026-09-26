<div align="center">

<h1>
  <img src="assets/precedent.gif" height="30" alt="" valign="middle">
  Precedent
</h1>

### Agentic memory for open source maintainers, so a convention only has to be explained once

[![Live demo](https://img.shields.io/badge/Live_demo-Try_it-9184d9?style=for-the-badge)](https://oczmd7cauwjhlt6zvjozdqz5i40rmdxj.lambda-url.ap-south-1.on.aws)

[![CockroachDB](https://img.shields.io/badge/CockroachDB-Vector_Index-6933FF?logo=cockroachlabs&logoColor=white)](https://www.cockroachlabs.com/)
[![AWS](https://img.shields.io/badge/AWS-Lambda_+_S3-FF9900?logo=amazonaws&logoColor=white)](https://aws.amazon.com/lambda/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-249_passing-2ea44f)](tests/)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

**86,317** review comments &nbsp;·&nbsp; **21,762** pull requests &nbsp;·&nbsp; **2011 to 2026** &nbsp;·&nbsp; **298** learned conventions

</div>

---

Large projects answer the same contributor questions forever. The answers exist,
buried in review threads going back a decade, but they are not retrievable, and a
maintainer's correction today does nothing for the person who asks the same thing
next month.

Precedent reads a repository's entire review history, distils the durable
conventions out of it, and says them back where the work happens: on a pull
request, unprompted, cited to the specific pull requests the claim came from. A
maintainer can correct it in place, and every later answer reflects that.

Target repository: [pandas-dev/pandas](https://github.com/pandas-dev/pandas).

## Correct it once, and it stays corrected

![A terminal session: asking where the GitHub issue number goes in a test returns an answer citing pull request 65052. A maintainer corrects it. The same question then returns the corrected answer, citing the correction instead.](assets/demo.gif)

Ask, correct, ask again. The rule the first answer used is retired, the
correction replaces it, and the second answer cites the maintainer rather than
the pull request it used to. Nothing was re-ingested and nothing was retrained.

## It comments on pull requests without being asked

[**See it happen on a real pull request.**](https://github.com/pyarchana/precedent/pull/5)

Nobody addressed the agent. A pull request opened touching
`pandas/core/groupby/groupby.py` and `doc/source/whatsnew/v3.0.0.rst`, and it
read the changed paths, found the conventions anchored to those files, and
posted them:

> **From pandas-dev/pandas review history**
>
> Nobody asked me. I read the files this pull request changes and found three
> conventions the project has settled before, each linked to where it was settled.
>
> - **Always include a whatsnew entry in the appropriate file for any bug fixes or new features.**
>   Established in [pandas-dev/pandas#64119](https://github.com/pandas-dev/pandas/pull/64119) and [#61985](https://github.com/pandas-dev/pandas/pull/61985).
>   Raised because it changes `doc/source/whatsnew/v3.0.0.rst`.

One of the three it raised there is not a rule it inferred from the corpus at
all. It is a convention a maintainer taught it earlier, through a pull request
comment, and it now sits alongside conventions distilled from a decade of review
threads.

Identity comes from GitHub, not from a login. Webhook deliveries are signed with
HMAC SHA-256, so `sender.login` is trustworthy without this application ever
handling a password, running an OAuth flow, or holding a session, and
`author_association` is GitHub's own answer to whether someone may speak for the
project.

**What it does not do is check whether your diff complies.** It names what the
project has already decided and links where. Claiming a diff violates a
convention means being right about the diff, and being wrong there costs more
than being unhelpful.

## Does the memory actually help

The obvious rebuttal is that `gpt-4o-mini` has read a lot of pandas and might
answer these questions on its own. So the same 29 evaluation questions went to
both, and the answer is uncomfortable in one column and decisive in the other.

| | correct | refused when it should | citations that resolve |
| --- | --- | --- | --- |
| Precedent | 8/24 (33%) | **4/5 (80%)** | **43/43 (100%)** |
| `gpt-4o-mini` alone | 9/24 (38%) | 0/5 (0%) | 0/24 (0%) |

**Memory did not make it more accurate.** The baseline scored one question
higher, which on 24 questions is noise rather than a result. Three of
Precedent's misses were refusals on questions it could have answered, so it is
also over-cautious.

What it changed is whether the answer can be trusted. Five questions are
deliberately unanswerable from review history. The baseline answered all five,
confidently, inventing project policy on release schedules, governance and
credentials. And of the 24 pull requests it cited, **none** resolve to a real
discussion in the corpus: it is generating plausible five-digit numbers.
Precedent cited 43 and every one resolves, because citations are verified
against retrieved evidence before an answer is released and a failure suppresses
the whole answer.

Memory does not make the model smarter. It makes it accountable, which is the
difference that matters when a contributor cannot tell a confident right answer
from a confident invented one.

Reproduce with `python scripts/run_baseline.py`, about a cent and a half.
Full output in [`eval/baseline.json`](eval/baseline.json).

## Memory model

![Precedent architecture: an offline ingest pipeline stages raw GitHub pages in S3, transforms and embeds them into CockroachDB, and clusters them into rules. An AWS Lambda serves pull request comments, questions and maintainer corrections from the same database.](assets/architecture.svg)

Four kinds of memory, one database. Not four services pretending to be one
system.

| Store | Table | What it holds |
| --- | --- | --- |
| Episodic | `review_comments` | Individual review comments, embedded for semantic search |
| Semantic | `rules` | Distilled repo conventions, with confidence and supersession history |
| Provenance | `rule_evidence` | Which comments each rule was learned from |
| Entity | `contributors` | Per-repo contributor state: volume, tenure, and areas touched |
| Working | `sessions`, `session_turns` | The conversation, and which rules each answer used |
| Corrections | `corrections` | What a maintainer corrected, and what it changed |

Every table is keyed on `repo_id`, and embeddings sit in the same tables as the
data they describe, so retrieval and the operational data can never disagree
about what the project said. [How corrections retire a rule without deleting
it](docs/architecture.md) is the part worth reading next.

## Built with

| CockroachDB | How it is used |
| --- | --- |
| **Distributed vector indexing** | `idx_rules_embedding` serves the agent's hot path. `EXPLAIN` confirms `vector search: rules@idx_rules_embedding`. Embeddings live beside the rows they describe, so there is no second store to keep consistent. |
| **ccloud CLI** | Provisioned and manages the serverless cluster in `ap-south-1`, next to the Lambda that queries it. |

| AWS | How it is used |
| --- | --- |
| **Lambda** | Serves the whole application, API, page and GitHub webhook, behind a Function URL. |
| **S3** | Holds the raw ingested review history, 3,801 gzipped pages, staged before transform. |

| GitHub | How it is used |
| --- | --- |
| **App webhooks** | Signed deliveries make `sender.login` trustworthy with no login of any kind. `author_association` decides who may teach the memory. |
| **Installation tokens** | RS256 JWT exchanged for a short-lived token, so the grant is revocable by uninstalling rather than by rotating a key. |

## Running it

```bash
pip install -e ".[db,api]"
python -m precedent.db.migrate --create-db
uvicorn precedent.api.app:app --port 8000
```

Then open http://localhost:8000. `COCKROACH_DSN` and `OPENAI_API_KEY` come from
`.env`; see `.env.example`. Deploying it, and the settings that exist because the
demo is a public URL in front of a paid model, are in
[docs/deployment.md](docs/deployment.md).

## Documentation

| | |
| --- | --- |
| [Architecture](docs/architecture.md) | Memory model, correction semantics, and why a model rather than a distance threshold decides what contradicts what |
| [Failure modes](docs/failure-modes.md) | Every failure that actually happened, and what handles it |
| [Deployment](docs/deployment.md) | Local, Lambda, and the two things Mangum's lifespan behaviour broke |
| [Ingest and database](docs/ingest.md) | GraphQL staging, bulk embedding, CockroachDB driver notes |

## License

MIT. See [LICENSE](LICENSE).
