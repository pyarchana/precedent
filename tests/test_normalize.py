"""Tests for the rules that decide what enters episodic memory.

These two classifications are load-bearing. `is_maintainer` feeds confidence
weighting, so getting it wrong quietly inflates the authority of ordinary
contributors. Bot filtering keeps CI chatter out of the corpus, and CI chatter
is repetitive enough that it would dominate any clustering built on top.
"""

from __future__ import annotations

from precedent.transform.normalize import (
    RejectReason,
    is_bot,
    is_maintainer,
    is_pr_template,
    normalize_page,
)


class TestIsMaintainer:
    def test_project_roles_are_maintainers(self):
        assert is_maintainer("OWNER")
        assert is_maintainer("MEMBER")
        assert is_maintainer("COLLABORATOR")

    def test_contributor_is_not_a_maintainer(self):
        # "Has had a PR merged" is not "speaks for the project".
        assert not is_maintainer("CONTRIBUTOR")
        assert not is_maintainer("FIRST_TIME_CONTRIBUTOR")
        assert not is_maintainer("NONE")

    def test_missing_association_is_not_a_maintainer(self):
        assert not is_maintainer(None)
        assert not is_maintainer("")

    def test_case_insensitive(self):
        assert is_maintainer("member")


class TestIsBot:
    def test_graphql_bot_typename(self):
        assert is_bot({"__typename": "Bot", "login": "dependabot"})

    def test_bot_suffix(self):
        assert is_bot({"__typename": "User", "login": "pre-commit-ci[bot]"})
        assert is_bot({"__typename": "User", "login": "GitHub-Actions[Bot]"})

    def test_known_bot_without_suffix(self):
        assert is_bot({"__typename": "User", "login": "codecov-io"})
        assert is_bot({"__typename": "User", "login": "MeeseeksDev"})

    def test_humans_are_not_bots(self):
        assert not is_bot({"__typename": "User", "login": "jbrockmendel"})

    def test_deleted_account_is_not_a_bot(self):
        # A null author means the GitHub account is gone, not that it was a bot.
        assert not is_bot(None)


def _page(*prs):
    return {"repository": {"pullRequests": {"nodes": list(prs)}}}


def _pr(number=1, **overrides):
    pr = {
        "number": number,
        "bodyText": "Fixes a thing.",
        "createdAt": "2024-01-01T00:00:00Z",
        "authorAssociation": "CONTRIBUTOR",
        "author": {"__typename": "User", "login": "someone"},
        "reviews": {"nodes": []},
        "reviewThreads": {"nodes": []},
        "comments": {"nodes": []},
    }
    pr.update(overrides)
    return pr


class TestNormalizePage:
    def test_pr_body_becomes_a_record(self):
        kept, dropped = normalize_page(_page(_pr(number=42)))
        assert len(kept) == 1
        assert not dropped
        record = kept[0]
        assert record.kind == "pr_body"
        assert record.pr_number == 42
        assert record.body == "Fixes a thing."
        assert not record.is_maintainer

    def test_pr_body_id_is_deterministic(self):
        # The query does not select the PR's node id, so this one is synthesised
        # and the upsert depends on it being stable across runs.
        first, _ = normalize_page(_page(_pr(number=42)))
        second, _ = normalize_page(_page(_pr(number=42)))
        assert first[0].github_node_id == second[0].github_node_id == "pr:42:body"

    def test_empty_body_is_rejected_not_dropped_silently(self):
        kept, dropped = normalize_page(_page(_pr(bodyText="   ")))
        assert not kept
        assert [r.reason for r in dropped] == [RejectReason.EMPTY_BODY]

    def test_bot_author_is_rejected(self):
        bot = {"__typename": "Bot", "login": "codecov[bot]"}
        kept, dropped = normalize_page(_page(_pr(author=bot)))
        assert not kept
        assert dropped[0].reason == RejectReason.BOT_AUTHOR
        assert dropped[0].author == "codecov[bot]"

    def test_review_thread_keeps_its_file_context(self):
        pr = _pr(
            bodyText="",
            reviewThreads={
                "nodes": [
                    {
                        "id": "THREAD1",
                        "path": "pandas/core/frame.py",
                        "line": 120,
                        "comments": {
                            "nodes": [
                                {
                                    "id": "COMMENT1",
                                    "bodyText": "Use a fixture here.",
                                    "createdAt": "2024-02-01T00:00:00Z",
                                    "authorAssociation": "MEMBER",
                                    "author": {"__typename": "User", "login": "maint"},
                                    "diffHunk": "@@ -1 +1 @@",
                                    "path": "pandas/core/frame.py",
                                    "originalLine": 120,
                                    "replyTo": None,
                                }
                            ]
                        },
                    }
                ]
            },
        )
        kept, _ = normalize_page(_page(pr))
        comment = next(r for r in kept if r.kind == "review_thread")
        assert comment.file_path == "pandas/core/frame.py"
        assert comment.line == 120
        assert comment.thread_id == "THREAD1"
        assert comment.diff_hunk == "@@ -1 +1 @@"
        assert comment.is_maintainer

    def test_null_pull_request_nodes_are_skipped(self):
        # GitHub returns nulls for nodes it could not resolve.
        kept, dropped = normalize_page(_page(None, _pr(number=7)))
        assert len(kept) == 1
        assert not dropped

    def test_all_four_kinds_are_extracted(self):
        pr = _pr(
            reviews={
                "nodes": [
                    {
                        "id": "REVIEW1",
                        "state": "APPROVED",
                        "bodyText": "Looks good.",
                        "submittedAt": "2024-03-01T00:00:00Z",
                        "authorAssociation": "MEMBER",
                        "author": {"__typename": "User", "login": "maint"},
                    }
                ]
            },
            reviewThreads={
                "nodes": [
                    {
                        "id": "T1",
                        "path": "a.py",
                        "line": 1,
                        "comments": {
                            "nodes": [
                                {
                                    "id": "C1",
                                    "bodyText": "Inline note.",
                                    "createdAt": "2024-03-01T00:00:00Z",
                                    "authorAssociation": "MEMBER",
                                    "author": {"__typename": "User", "login": "maint"},
                                    "diffHunk": "",
                                    "path": "a.py",
                                    "originalLine": 1,
                                    "replyTo": None,
                                }
                            ]
                        },
                    }
                ]
            },
            comments={
                "nodes": [
                    {
                        "id": "IC1",
                        "bodyText": "Conversation note.",
                        "createdAt": "2024-03-02T00:00:00Z",
                        "authorAssociation": "NONE",
                        "author": {"__typename": "User", "login": "passerby"},
                    }
                ]
            },
        )
        kept, _ = normalize_page(_page(pr))
        assert {r.kind for r in kept} == {
            "pr_body",
            "review_summary",
            "review_thread",
            "issue_comment",
        }

    def test_node_ids_are_unique_within_a_page(self):
        # The loader upserts on this id; a collision inside one batch would make
        # CockroachDB reject the statement outright.
        kept, _ = normalize_page(_page(_pr(number=1), _pr(number=2)))
        ids = [r.github_node_id for r in kept]
        assert len(ids) == len(set(ids))


class TestPullRequestTemplate:
    """The template is a form, not review guidance.

    Auditing the extracted rules found six of the top fifty built on this
    text, producing statements whose supporting evidence was a checklist. It
    is 2,585 comments, one per pull request, about 4% of the corpus.
    """

    def test_filled_in_template_is_rejected(self):
        body = (
            "closes #12345 (Replace xxxx with the Github issue number)\n"
            "Tests added and passed if fixing a bug or adding a new feature\n"
            "All code checks passed.\n"
            "Added type annotations to new arguments/methods/functions.\n"
            "Added an entry in the latest doc/source/whatsnew/v2.0.0.rst file."
        )
        assert is_pr_template(body)

    def test_a_comment_quoting_one_line_is_kept(self):
        # A reviewer reminding someone about one checklist item is real
        # guidance, so a single marker must not be enough to reject.
        body = (
            "Please make sure all code checks passed before I take another "
            "look, the linter is failing on this file."
        )
        assert not is_pr_template(body)

    def test_ordinary_review_comment_is_kept(self):
        body = "Can you use the match parameter of pytest.raises to check the message?"
        assert not is_pr_template(body)

    def test_rejected_with_its_own_reason(self):
        pr = _pr(
            bodyText=(
                "closes #999 (Replace xxxx with the Github issue number). All code checks passed."
            )
        )
        kept, dropped = normalize_page(_page(pr))
        assert not kept
        assert dropped[0].reason == RejectReason.PR_TEMPLATE
