"""Tests for authenticating as the GitHub App.

This is the first code in the project that writes somewhere other than its own
database, so the failure modes are different in kind. A wrong answer is
embarrassing; a wrong write appears under the project's name on a stranger's
pull request and cannot be taken back.

No network. Every request is served by an in-process transport, so the tests
assert on the exact URL and body that would have gone to GitHub.
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from precedent.api.github_auth import (
    JWT_LIFETIME_SECONDS,
    MAX_CHANGED_FILES,
    Credentials,
    GitHubApp,
    GitHubError,
    NotConfigured,
    app_jwt,
)


@pytest.fixture(scope="module")
def keypair() -> tuple[str, str]:
    """A throwaway RSA key. Generated rather than checked in, because a private
    key in a repository is a private key on the internet even when it is a toy."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    public = (
        key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode()
    )
    return private, public


@dataclass
class FakeSettings:
    github_app_id: str = "12345"
    github_private_key: str = ""
    github_private_key_b64: str = ""
    github_installation_id: str = "67890"


class TestReadingCredentials:
    def test_a_raw_pem_is_accepted(self, keypair):
        private, _ = keypair
        credentials = Credentials.from_settings(FakeSettings(github_private_key=private))
        assert credentials.private_key == private.strip()

    def test_a_base64_pem_is_decoded(self, keypair):
        # The form actually used in deployment: a PEM is multi-line and a
        # multi-line environment variable does not survive the Lambda console.
        private, _ = keypair
        encoded = base64.b64encode(private.encode()).decode()
        credentials = Credentials.from_settings(FakeSettings(github_private_key_b64=encoded))
        assert credentials.private_key == private

    def test_a_raw_key_wins_over_an_encoded_one(self, keypair):
        private, _ = keypair
        settings = FakeSettings(
            github_private_key=private,
            github_private_key_b64=base64.b64encode(b"something else").decode(),
        )
        assert Credentials.from_settings(settings).private_key == private.strip()

    def test_mangled_base64_says_so_rather_than_failing_later(self):
        with pytest.raises(NotConfigured, match="base64"):
            Credentials.from_settings(FakeSettings(github_private_key_b64="not base64 at all"))

    def test_nothing_configured_names_every_missing_variable(self):
        with pytest.raises(NotConfigured) as exc:
            Credentials.from_settings(FakeSettings(github_app_id="", github_installation_id=""))
        message = str(exc.value)
        assert "GITHUB_APP_ID" in message
        assert "GITHUB_INSTALLATION_ID" in message
        assert "GITHUB_PRIVATE_KEY" in message

    def test_a_missing_key_alone_is_reported(self, keypair):
        with pytest.raises(NotConfigured, match="PRIVATE_KEY"):
            Credentials.from_settings(FakeSettings())


class TestTheAssertion:
    def test_github_can_verify_it(self, keypair):
        private, public = keypair
        token = app_jwt(Credentials("12345", private, "67890"))
        claims = jwt.decode(token, public, algorithms=["RS256"])
        assert claims["iss"] == "12345"

    def test_it_is_backdated_against_clock_skew(self, keypair):
        # A JWT issued one second in GitHub's future is rejected outright, and
        # the resulting 401 looks like a bad key rather than a clock.
        private, public = keypair
        now = time.time()
        claims = jwt.decode(
            app_jwt(Credentials("1", private, "2"), now=now),
            public,
            algorithms=["RS256"],
        )
        assert claims["iat"] < now

    def test_it_expires_inside_the_ten_minutes_github_allows(self, keypair):
        # Measured from iat, not from now, which is the trap: backdating spends
        # part of the ten minutes. The first version of this used 540 plus 60
        # and landed on exactly 600, which is the boundary rather than inside
        # it.
        private, public = keypair
        now = time.time()
        claims = jwt.decode(
            app_jwt(Credentials("1", private, "2"), now=now),
            public,
            algorithms=["RS256"],
        )
        assert claims["exp"] - claims["iat"] < 600
        assert claims["exp"] - int(now) == JWT_LIFETIME_SECONDS


@dataclass
class Recorder:
    """An in-process GitHub. Records what was asked of it."""

    requests: list[httpx.Request] = field(default_factory=list)
    token_calls: int = 0
    token_lifetime: timedelta = timedelta(hours=1)
    files_pages: list[list[str]] = field(default_factory=list)
    token_status: int = 201

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path

        if path.endswith("/access_tokens"):
            self.token_calls += 1
            if self.token_status != 201:
                return httpx.Response(self.token_status, text="no")
            expires = datetime.now(UTC) + self.token_lifetime
            return httpx.Response(
                201,
                json={
                    "token": f"ghs_token_{self.token_calls}",
                    "expires_at": expires.isoformat().replace("+00:00", "Z"),
                },
            )

        if path.endswith("/files"):
            page = int(request.url.params.get("page", 1))
            names = self.files_pages[page - 1] if page <= len(self.files_pages) else []
            return httpx.Response(200, json=[{"filename": name} for name in names])

        if path.endswith("/comments"):
            return httpx.Response(
                201, json={"html_url": "https://github.com/o/r/pull/1#issuecomment-1"}
            )

        return httpx.Response(404, text="unexpected")


def build(recorder: Recorder, keypair) -> GitHubApp:
    private, _ = keypair
    return GitHubApp(
        Credentials("12345", private, "67890"),
        client=httpx.AsyncClient(
            base_url="https://api.github.com", transport=httpx.MockTransport(recorder.handler)
        ),
    )


class TestInstallationTokens:
    async def test_a_token_is_exchanged_for_the_installation(self, keypair):
        recorder = Recorder()
        app = build(recorder, keypair)
        assert await app.token() == "ghs_token_1"
        assert recorder.requests[0].url.path == "/app/installations/67890/access_tokens"

    async def test_the_exchange_is_signed_with_the_app_assertion(self, keypair):
        _, public = keypair
        recorder = Recorder()
        app = build(recorder, keypair)
        await app.token()
        header = recorder.requests[0].headers["Authorization"]
        claims = jwt.decode(header.removeprefix("Bearer "), public, algorithms=["RS256"])
        assert claims["iss"] == "12345"

    async def test_a_live_token_is_not_re_minted(self, keypair):
        recorder = Recorder()
        app = build(recorder, keypair)
        await app.token()
        await app.token()
        await app.token()
        assert recorder.token_calls == 1

    async def test_a_token_about_to_lapse_is_replaced_before_it_does(self, keypair):
        # Renewed on a margin rather than at expiry, so a token cannot die
        # between the check and the request that uses it.
        recorder = Recorder(token_lifetime=timedelta(seconds=60))
        app = build(recorder, keypair)
        assert await app.token() == "ghs_token_1"
        assert await app.token() == "ghs_token_2"

    async def test_a_refusal_is_not_mistaken_for_a_token(self, keypair):
        recorder = Recorder(token_status=403)
        app = build(recorder, keypair)
        with pytest.raises(GitHubError, match="403"):
            await app.token()


class TestReadingAPullRequest:
    async def test_the_changed_paths_come_back(self, keypair):
        recorder = Recorder(files_pages=[["pandas/core/frame.py", "doc/source/x.rst"]])
        app = build(recorder, keypair)
        assert await app.changed_files("o/r", 7) == ["pandas/core/frame.py", "doc/source/x.rst"]

    async def test_a_pull_request_larger_than_one_page_is_followed(self, keypair):
        recorder = Recorder(files_pages=[[f"file_{i}.py" for i in range(100)], ["last.py"]])
        app = build(recorder, keypair)
        paths = await app.changed_files("o/r", 7)
        assert len(paths) == 101
        assert paths[-1] == "last.py"

    async def test_a_short_page_ends_the_walk(self, keypair):
        recorder = Recorder(files_pages=[["a.py"], ["never_read.py"]])
        app = build(recorder, keypair)
        assert await app.changed_files("o/r", 7) == ["a.py"]

    async def test_an_enormous_pull_request_is_cut_off(self, keypair):
        recorder = Recorder(files_pages=[[f"f{i}.py" for i in range(100)] for _ in range(10)])
        app = build(recorder, keypair)
        assert len(await app.changed_files("o/r", 7)) == MAX_CHANGED_FILES


class TestCommenting:
    async def test_it_posts_to_the_conversation_not_the_diff(self, keypair):
        # On GitHub a pull request conversation comment is an issue comment.
        # The pulls endpoint creates review comments, which must be anchored to
        # a line of a diff and would fail without one.
        recorder = Recorder()
        app = build(recorder, keypair)
        await app.comment("o/r", 42, "hello")
        posted = recorder.requests[-1]
        assert posted.url.path == "/repos/o/r/issues/42/comments"
        assert json.loads(posted.content)["body"] == "hello"

    async def test_it_returns_where_the_comment_landed(self, keypair):
        app = build(Recorder(), keypair)
        assert "issuecomment" in await app.comment("o/r", 42, "hello")

    async def test_the_installation_token_is_what_authorises_it(self, keypair):
        # Not the app assertion. The assertion identifies the application and
        # can write nothing; only the installation token carries the permissions
        # the repository owner granted.
        recorder = Recorder()
        app = build(recorder, keypair)
        await app.comment("o/r", 42, "hello")
        assert recorder.requests[-1].headers["Authorization"] == "Bearer ghs_token_1"
