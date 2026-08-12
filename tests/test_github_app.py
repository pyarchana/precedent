"""Tests for learning from pull request comments.

The signature check is the whole access control story here. There is no login,
no session and no OAuth: the only reason `sender.login` can be trusted is that
GitHub signed the payload. So it gets tested like the security boundary it is,
including the ways a forgery might slip past.

The rest is about not learning the wrong things. A webhook fires on every
comment in the repository, most of which are not conventions, several of which
are the agent's own words quoted back at it.
"""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest

from precedent.api.github_app import (
    SignatureInvalid,
    TaughtRule,
    acknowledge,
    extract_instruction,
    parse_event,
    parse_pull_request,
    speaks_for_project,
    verify_signature,
)

SECRET = "s3cret"


def signed(body: bytes, secret: str = SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


class TestSignature:
    def test_a_genuine_delivery_passes(self):
        body = json.dumps({"action": "created"}).encode()
        verify_signature(SECRET, body, signed(body))

    def test_a_forged_delivery_is_rejected(self):
        body = b'{"action":"created"}'
        with pytest.raises(SignatureInvalid):
            verify_signature(SECRET, body, signed(body, secret="wrong"))

    def test_an_altered_body_is_rejected(self):
        # The point of signing. Someone replaying a real delivery with the
        # author swapped must not be able to teach memory as that person.
        original = b'{"sender":{"login":"contributor"}}'
        header = signed(original)
        tampered = b'{"sender":{"login":"jbrockmendel"}}'
        with pytest.raises(SignatureInvalid):
            verify_signature(SECRET, tampered, header)

    def test_a_missing_header_is_rejected(self):
        with pytest.raises(SignatureInvalid):
            verify_signature(SECRET, b"{}", None)

    def test_an_unsigned_scheme_is_rejected(self):
        body = b"{}"
        digest = hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
        with pytest.raises(SignatureInvalid):
            verify_signature(SECRET, body, digest)  # no "sha256=" prefix

    def test_an_unconfigured_secret_refuses_everything(self):
        # Failing open here would mean a deployment that forgot the secret
        # accepts anything the internet sends it.
        body = b"{}"
        with pytest.raises(SignatureInvalid):
            verify_signature("", body, signed(body))


class TestWhoSpeaksForTheProject:
    known = frozenset({"jreback"})

    def test_a_current_member_speaks(self):
        assert speaks_for_project("MEMBER", "someone", frozenset())

    def test_an_owner_and_collaborator_speak(self):
        assert speaks_for_project("OWNER", "a", frozenset())
        assert speaks_for_project("COLLABORATOR", "b", frozenset())

    def test_a_contributor_does_not(self):
        assert not speaks_for_project("CONTRIBUTOR", "someone", frozenset())

    def test_a_former_maintainer_still_speaks(self):
        # authorAssociation reports current permissions, so someone who has
        # stepped back reads as CONTRIBUTOR. The derived list covers that.
        assert speaks_for_project("CONTRIBUTOR", "jreback", self.known)

    def test_the_known_list_is_case_insensitive(self):
        assert speaks_for_project("NONE", "JReback", self.known)


def _comment_event(body: str, association="MEMBER", login="maint", is_pr=True) -> dict:
    return {
        "action": "created",
        "issue": {"number": 42, **({"pull_request": {"url": "..."}} if is_pr else {})},
        "comment": {
            "node_id": "IC_abc",
            "body": body,
            "user": {"login": login},
            "author_association": association,
            "html_url": "https://github.com/o/r/pull/42#issuecomment-1",
        },
    }


class TestParsing:
    def test_a_pull_request_comment_is_read(self):
        parsed = parse_event("issue_comment", _comment_event("@precedent do X"))
        assert parsed["pr_number"] == 42
        assert parsed["author"] == "maint"
        assert parsed["association"] == "MEMBER"

    def test_a_plain_issue_is_ignored(self):
        # The corpus is review history. Issue threads are a different thing.
        assert parse_event("issue_comment", _comment_event("@precedent x", is_pr=False)) is None

    def test_an_edit_is_ignored(self):
        event = _comment_event("@precedent x")
        event["action"] = "edited"
        assert parse_event("issue_comment", event) is None

    def test_unrelated_events_are_ignored(self):
        assert parse_event("push", {"action": "created"}) is None
        assert parse_event("star", {"action": "created"}) is None

    def test_an_inline_review_comment_keeps_its_file(self):
        payload = {
            "action": "created",
            "pull_request": {"number": 7},
            "comment": {
                "node_id": "PRRC_1",
                "body": "@precedent use a fixture",
                "user": {"login": "maint"},
                "author_association": "MEMBER",
                "path": "pandas/tests/test_x.py",
            },
        }
        parsed = parse_event("pull_request_review_comment", payload)
        assert parsed["file_path"] == "pandas/tests/test_x.py"
        assert parsed["pr_number"] == 7


class TestExtractingTheInstruction:
    def test_text_after_the_trigger_is_taken(self):
        assert (
            extract_instruction("@precedent the note goes in the next file", "@precedent")
            == "the note goes in the next file"
        )

    def test_a_comment_without_the_trigger_is_not_for_us(self):
        assert extract_instruction("looks good to me", "@precedent") is None

    def test_the_trigger_is_case_insensitive(self):
        assert extract_instruction("@Precedent do X", "@precedent") == "do X"

    def test_a_trailing_colon_is_not_part_of_the_instruction(self):
        assert extract_instruction("@precedent: do X", "@precedent") == "do X"

    def test_quoted_text_is_dropped(self):
        # GitHub's reply button quotes what it is replying to. Without this the
        # agent reads its own earlier answer back as a maintainer's words.
        body = "> @precedent something the bot said\n\n@precedent actually do Y"
        assert extract_instruction(body, "@precedent") == "actually do Y"

    def test_a_bare_mention_teaches_nothing(self):
        assert extract_instruction("@precedent", "@precedent") is None
        assert extract_instruction("@precedent   ", "@precedent") is None

    def test_a_multi_line_instruction_is_flattened(self):
        body = "@precedent put the note\nin the next release file"
        assert extract_instruction(body, "@precedent") == "put the note in the next release file"


class TestActingUnprompted:
    """Which pull request events are worth reading at all.

    This is the one path with no human intent behind it, so the filtering has
    to happen before any work is done. Every event admitted here becomes a
    comment on somebody's pull request.
    """

    @staticmethod
    def payload(action="opened", *, draft=False, number=42, repo="pyarchana/demo"):
        return {
            "action": action,
            "repository": {"full_name": repo},
            "pull_request": {
                "number": number,
                "title": "Fix groupby apply",
                "draft": draft,
                "user": {"login": "contributor"},
            },
        }

    def test_a_new_pull_request_is_acted_on(self):
        parsed = parse_pull_request("pull_request", self.payload())
        assert parsed["pr_number"] == 42
        assert parsed["repo"] == "pyarchana/demo"
        assert parsed["title"] == "Fix groupby apply"

    def test_a_push_to_an_existing_pull_request_is_not(self):
        # `synchronize` fires on every push. Four force-pushes would collect
        # four identical comments, and nothing the agent has to say changes
        # between them.
        assert parse_pull_request("pull_request", self.payload("synchronize")) is None

    def test_reopening_does_not_repeat_the_comment(self):
        assert parse_pull_request("pull_request", self.payload("reopened")) is None

    def test_a_draft_is_left_alone(self):
        assert parse_pull_request("pull_request", self.payload(draft=True)) is None

    def test_a_draft_is_read_once_it_is_marked_ready(self):
        parsed = parse_pull_request("pull_request", self.payload("ready_for_review", draft=True))
        assert parsed["pr_number"] == 42

    def test_a_comment_event_is_not_a_pull_request_event(self):
        assert parse_pull_request("issue_comment", self.payload()) is None

    def test_a_payload_without_a_repository_is_unusable(self):
        # The pull request is not in the repository the memory is about, so
        # there is nothing to fall back on: without the name there is nowhere
        # to read the diff from or post to.
        payload = self.payload()
        del payload["repository"]
        assert parse_pull_request("pull_request", payload) is None

    def test_a_payload_without_a_number_is_unusable(self):
        payload = self.payload()
        del payload["pull_request"]["number"]
        assert parse_pull_request("pull_request", payload) is None


class TestAcknowledging:
    """Writing to memory and answering 200 to GitHub is invisible.

    A maintainer typed a correction, got silence, and had no way to tell
    whether it worked. That is the same problem as an unverifiable citation
    pointed the other way: they are asked to trust an outcome they cannot see.
    """

    @staticmethod
    def taught(outcome="inserted", superseded=None):
        return TaughtRule(
            statement="Place whatsnew entries in the file for the next release.",
            rule_id="r1",
            outcome=outcome,
            comment_id="c1",
            author="pyarchana",
            pr_number=2,
            superseded_statement=superseded,
        )

    def test_it_quotes_the_rule_as_stored(self):
        # Not a thank you. A maintainer who reads the wording back and
        # disagrees can correct the correction, which needs the wording shown.
        body = acknowledge(self.taught())
        assert "> Place whatsnew entries in the file for the next release." in body

    def test_it_credits_the_person_who_said_it(self):
        assert "pyarchana" in acknowledge(self.taught())

    def test_a_new_rule_says_nothing_was_retired(self):
        assert "Nothing was retired" in acknowledge(self.taught())

    def test_supersession_names_what_it_replaced(self):
        body = acknowledge(
            self.taught("superseded", "Put the whatsnew note in the current release file.")
        )
        assert "retired" in body
        assert "current release file" in body

    def test_a_merge_does_not_claim_to_have_retired_anything(self):
        body = acknowledge(self.taught("merged"))
        assert "already held" in body.lower()
        assert "retired" not in body

    def test_supersession_without_the_old_text_does_not_invent_it(self):
        # The lookup can come back empty if the rule was deleted between the
        # write and the read. Better to say less than to describe a rule that
        # cannot be quoted.
        body = acknowledge(self.taught("superseded", None))
        assert "Nothing was retired" in body


class TestTheReplyAddress:
    def test_the_repository_is_carried_so_a_reply_has_somewhere_to_go(self):
        # The pull request is not in the repository the memory is about, so
        # this cannot be taken from settings.
        payload = {
            "action": "created",
            "repository": {"full_name": "pyarchana/precedent"},
            "issue": {
                "number": 2,
                "pull_request": {"url": "https://api.github.com/repos/pyarchana/precedent/pulls/2"},
            },
            "comment": {
                "node_id": "IC_1",
                "body": "@precedent do X",
                "user": {"login": "pyarchana"},
                "author_association": "OWNER",
            },
        }
        assert parse_event("issue_comment", payload)["repo"] == "pyarchana/precedent"
