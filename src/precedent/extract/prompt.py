"""The rule extraction prompt and its output schema.

Written against real clusters rather than imagined ones. Reading a sample made
the failure mode obvious: most clusters are comments that share vocabulary
without sharing a convention. Six maintainers discussing the `_offset`
attribute are not expressing a rule, they are debugging. The prompt therefore
spends most of its effort on refusal, because a pile of plausible-sounding
non-rules is worse than a short list of real ones.

Two clusters that *were* real, and what they should produce:

  * Four separate pull requests where one maintainer said the SQL layer must
    stay database agnostic and defer to SQLAlchemy. That is a convention with
    a scope, and it is repo-specific in a way no base model would know.

  * Three different maintainers across four PRs asking contributors to keep
    the diff minimal, revert unrelated changes, and add a test that fails
    before the fix. A process convention, and stronger for being independent.
"""

from __future__ import annotations

# Matches the rule_scope enum in migration 0001.
SCOPES = ("repo", "directory", "file", "api", "testing", "docs", "style", "process")

SYSTEM = """\
You extract durable engineering conventions from code review history.

You are given several review comments from one repository that a semantic \
search grouped together. Decide whether they express a convention the project \
actually holds, and if so, state it.

A convention is guidance that would apply again to a future contributor who \
had never seen these pull requests. It is a rule the project follows.

Most clusters are NOT conventions. Refuse when:
  - the comments merely share vocabulary while discussing different things
  - the guidance is specific to one pull request's code, bug, or design
  - the comments are debugging, questions, or thinking aloud
  - the comments disagree with each other, or the matter is unresolved
  - the "rule" would be true of any Python project and says nothing about \
this repository specifically

Refusing is the correct answer more often than not. A wrong rule is worse \
than a missing one, because it will be cited to a contributor as though the \
project had said it.

Reply with JSON only, no prose, in exactly this shape:

{
  "is_convention": true or false,
  "reason": "one sentence, required when is_convention is false",
  "statement": "the rule in one sentence, addressed to a contributor",
  "rationale": "why the project holds it, one sentence, only if the comments say",
  "scope": one of SCOPE_LIST,
  "scope_pattern": "SQL LIKE pattern such as pandas/tests/%" or null,
  "supporting_prs": [pull request numbers that genuinely support the rule]
}

Rules for the fields:
  - "statement" must be actionable and specific to this repository. Not \
"write good tests" but "put the GitHub issue number as a comment on the first \
line of a new test".
  - "scope_pattern" is required when scope is "directory" or "file", and null \
otherwise.
  - "supporting_prs" must only list pull requests whose comments actually \
support the rule. Drop any that do not. If fewer than two remain, the cluster \
is one conversation, so set is_convention to false.
""".replace("SCOPE_LIST", ", ".join(f'"{s}"' for s in SCOPES))

USER = """\
Repository: {repo}

Review comments:

{cluster}
"""


def build_messages(repo: str, cluster_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER.format(repo=repo, cluster=cluster_text)},
    ]
