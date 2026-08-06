"""Deciding whether a new rule contradicts one already in memory.

Merging handles rules that say the same thing. This handles the harder case:
rules about the same subject that cannot both be followed. Copy-on-Write is
the motivating example. Advice written before it became the default is not
false, it was true and then the project changed underneath it, and a memory
that cannot represent that will confidently cite guidance the project has
since abandoned.

Only rules in a narrow band are compared. Below the merge threshold they are
the same rule; far above it they are about different things and asking a model
to compare them is money spent to be told "unrelated". The band is where the
interesting cases live: same subject, different instruction.

Recency decides direction, not truth. If two rules conflict and one rests on
more recent evidence, the older one is marked superseded rather than deleted,
and keeps its evidence so the change remains explicable.
"""

from __future__ import annotations

SYSTEM = """\
You compare two engineering conventions from the same repository and decide \
how they relate.

Reply with JSON only:

{
  "relation": "same" | "contradicts" | "compatible",
  "reason": "one sentence"
}

  - "same": they express the same convention. Answer "same" whenever a \
contributor following one is automatically following the other, even if the \
wording, the level of detail, or the emphasis differs. "Include the issue \
number in the pull request description" and "include the issue number in the \
first line of your pull request description" are the same convention stated \
with different precision, not two conventions. Prefer "same" over \
"compatible" when the two are about the same specific action.
  - "contradicts": a contributor cannot follow both. One instructs something \
the other forbids, or they prescribe different answers to the same question. \
This includes a rule that has been overtaken by a change in the project, \
where the older instruction is no longer what the project does.
  - "compatible": they are about genuinely different actions, even if they \
share a subject. "Put the issue number in the pull request description" and \
"put the issue number in a comment on the test" are compatible: both can be \
followed and they ask for different things in different places.

Be strict about "contradicts". Two rules about the same file, or the same \
API, or the same part of the process, are usually compatible: being about the \
same subject is not the same as being in conflict. Only say "contradicts" \
when following one genuinely means violating the other, because the \
consequence is that a rule gets retired.
"""

USER = """\
Existing rule, most recent supporting evidence {old_date}:
{old}

New rule, most recent supporting evidence {new_date}:
{new}
"""


def build_messages(
    *,
    old_statement: str,
    new_statement: str,
    old_date: str,
    new_date: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": USER.format(
                old=old_statement,
                new=new_statement,
                old_date=old_date,
                new_date=new_date,
            ),
        },
    ]
