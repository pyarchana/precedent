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
    extract_instruction,
    parse_event,
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
