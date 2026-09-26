# Architecture

Ingest runs once per repository. Everything below the dashed line runs per
request, in one Lambda, against one database.

![Precedent architecture: an offline ingest pipeline stages raw GitHub pages in S3, transforms and embeds them into CockroachDB, and clusters them into rules. An AWS Lambda serves pull request comments, questions and maintainer corrections from the same database.](../assets/architecture.svg)

The six stores and the tables behind them are listed in the
[README](../README.md#memory-model). Every table is keyed on `repo_id`, so the
system is multi-tenant from the schema up. It has never been run with a second
tenant, which is a [known gap](../ROADMAP.md), not a claim.

## What a correction does

Retrieval with citations is a search engine with good manners. What makes this a
memory is that a maintainer can correct an answer once, and the next contributor
to ask gets the corrected one.

![What a correction does: review comments become a rule, the rule becomes a cited answer, and a maintainer correction is judged by a model. If it contradicts, the old rule is retired with its evidence kept and a replacement is written with a confidence floor of 0.85. If it agrees, it becomes further evidence for the same rule.](../assets/memory-lifecycle.svg)

<details>
<summary>The same three steps at the command line</summary>

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

</details>

## Four decisions the diagram cannot show

**The correction targets the rule the answer used**, not the rule nearest to the
correction's wording. Every answer records the rule ids it was built from, so a
correction arriving weeks later can still find them. Nearest neighbour would
occasionally retire a different rule than the maintainer meant, which is worse
than doing nothing.

**A model judges contradiction, not a distance threshold.** Embeddings put
opposites close together: "use single quotes" and "use double quotes" sit nearer
to each other than two genuine duplicates do. An earlier version merged on
distance alone, and feeding it a reversal made the reversal *further evidence
for* the thing it reversed. A correction that strengthens the error it corrects
is the one failure this system cannot have.

**A correction gets a confidence floor, not a score.** Confidence is built from
independent voices, distinct pull requests, persistence and recency, and a
correction has one author, one occasion and no history, so on those terms it
scores near the floor. A maintainer's own words would be presented as "weakly
evidenced" while a pattern inferred from three pull requests in 2016 read as
settled. Rules therefore record their `origin`, and a correction is floored at
0.85: above the rule it replaced, below a genuinely well attested convention.

**Nothing is deleted.** The retired rule keeps its evidence, its confidence and
the reason it was replaced, because "we used to say X, then this happened" is
what makes the memory explicable rather than merely current.

## Related

- [Failure modes](failure-modes.md), and what handles each one
- [Deployment](deployment.md), including what Lambda changed about this design
- [Ingest and database](ingest.md)
