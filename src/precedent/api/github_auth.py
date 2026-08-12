"""Authenticate as the GitHub App, so the agent can act rather than only listen.

Receiving a webhook needs nothing but the shared secret. Writing back needs the
app to prove who it is, and GitHub's scheme for that has two steps.

The app signs a short JWT with its own private key. That JWT identifies the
application but can do almost nothing on its own: it exists to be exchanged for
an *installation* token, which is what carries the permissions a repository
owner actually granted when they installed the app. The exchange matters because
it is what makes the grant revocable. An owner who uninstalls stops the next
token from being issued, without anyone having to rotate a key.

Installation tokens last an hour, so they are cached. A fresh token per pull
request would be a needless round trip on a path already waiting on a model.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import time
from dataclasses import dataclass
from datetime import datetime

import httpx

log = logging.getLogger(__name__)

API = "https://api.github.com"
ACCEPT = "application/vnd.github+json"

# GitHub rejects a JWT that lives longer than ten minutes, and it measures that
# from `iat`, not from now. The backdating therefore spends part of the
# allowance: 540 plus 60 is exactly 600, sitting on the boundary, which is the
# kind of thing that works until GitHub rounds the other way. These two must sum
# to comfortably under 600, and a test asserts it.
#
# The backdating itself is not optional. A JWT issued one second in GitHub's
# future is refused outright, and the 401 that comes back looks like a bad key
# rather than a clock a second fast.
JWT_LIFETIME_SECONDS = 480
JWT_BACKDATE_SECONDS = 60

# Refresh this far before expiry rather than at it, so a token cannot lapse
# between the check and the call that uses it.
TOKEN_MARGIN_SECONDS = 300

# A pull request touching more files than this is not one a convention check has
# anything useful to say about, and paging through it would cost several round
# trips to say nothing.
MAX_CHANGED_FILES = 300


class NotConfigured(Exception):
    """The app cannot authenticate, because it was never given credentials."""


class GitHubError(Exception):
    """GitHub refused a request."""


@dataclass(frozen=True, slots=True)
class Credentials:
    app_id: str
    private_key: str
    installation_id: str

    @classmethod
    def from_settings(cls, settings) -> Credentials:
        """Read credentials from configuration, or raise if they are incomplete.

        The key is accepted base64 encoded as well as raw. A PEM is multi-line,
        and a multi-line value survives neither the Lambda console nor a `.env`
        file reliably, so the encoded form is the one actually used in
        deployment.
        """
        key = settings.github_private_key.strip()
        if not key and settings.github_private_key_b64.strip():
            try:
                key = base64.b64decode(settings.github_private_key_b64, validate=True).decode()
            except (binascii.Error, UnicodeDecodeError) as exc:
                raise NotConfigured(f"GITHUB_PRIVATE_KEY_B64 is not valid base64: {exc}") from exc

        missing = [
            name
            for name, value in (
                ("GITHUB_APP_ID", settings.github_app_id),
                ("GITHUB_PRIVATE_KEY or GITHUB_PRIVATE_KEY_B64", key),
                ("GITHUB_INSTALLATION_ID", settings.github_installation_id),
            )
            if not str(value).strip()
        ]
        if missing:
            raise NotConfigured(f"not set: {', '.join(missing)}")

        return cls(
            app_id=str(settings.github_app_id).strip(),
            private_key=key,
            installation_id=str(settings.github_installation_id).strip(),
        )


def app_jwt(credentials: Credentials, *, now: float | None = None) -> str:
    """Sign the assertion that identifies this application.

    Imported here rather than at module scope so that a deployment which never
    posts anything does not need the signing libraries present at all.
    """
    import jwt

    issued = int(now if now is not None else time.time())
    return jwt.encode(
        {
            "iat": issued - JWT_BACKDATE_SECONDS,
            "exp": issued + JWT_LIFETIME_SECONDS,
            "iss": credentials.app_id,
        },
        credentials.private_key,
        algorithm="RS256",
    )


class GitHubApp:
    """A client that holds an installation token and renews it when it ages out."""

    def __init__(self, credentials: Credentials, *, client: httpx.AsyncClient | None = None):
        self._credentials = credentials
        self._client = client or httpx.AsyncClient(base_url=API, timeout=httpx.Timeout(20.0))
        self._token: str | None = None
        self._expires_at: float = 0.0
        # Two pull requests opened together would otherwise both see an expired
        # token and both mint a replacement.
        self._lock = asyncio.Lock()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def token(self) -> str:
        async with self._lock:
            if self._token and time.time() < self._expires_at - TOKEN_MARGIN_SECONDS:
                return self._token

            response = await self._client.post(
                f"/app/installations/{self._credentials.installation_id}/access_tokens",
                headers={
                    "Authorization": f"Bearer {app_jwt(self._credentials)}",
                    "Accept": ACCEPT,
                },
            )
            if response.status_code != 201:
                raise GitHubError(
                    f"could not get an installation token ({response.status_code}): "
                    f"{response.text[:200]}"
                )

            payload = response.json()
            self._token = payload["token"]
            # 3.11's fromisoformat parses the trailing Z, which earlier
            # versions did not. The project's floor is 3.11, so no fixup.
            self._expires_at = datetime.fromisoformat(payload["expires_at"]).timestamp()
            log.info(
                "installation token good for %d minutes",
                max(0, int((self._expires_at - time.time()) // 60)),
            )
            return self._token

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        response = await self._client.request(
            method,
            path,
            headers={"Authorization": f"Bearer {await self.token()}", "Accept": ACCEPT},
            **kwargs,
        )
        if response.status_code >= 400:
            raise GitHubError(
                f"{method} {path} returned {response.status_code}: {response.text[:200]}"
            )
        return response

    async def changed_files(self, repo: str, pr_number: int) -> list[str]:
        """The paths a pull request touches, which is what decides what applies."""
        paths: list[str] = []
        page = 1
        while len(paths) < MAX_CHANGED_FILES:
            response = await self._request(
                "GET",
                f"/repos/{repo}/pulls/{pr_number}/files",
                params={"per_page": 100, "page": page},
            )
            batch = response.json()
            if not batch:
                break
            paths.extend(entry["filename"] for entry in batch)
            if len(batch) < 100:
                break
            page += 1

        if len(paths) >= MAX_CHANGED_FILES:
            log.info("#%s touches at least %d files; truncating", pr_number, MAX_CHANGED_FILES)
        return paths[:MAX_CHANGED_FILES]

    async def comment(self, repo: str, pr_number: int, body: str) -> str:
        """Post to the pull request conversation.

        The issues endpoint, not the pulls one: on GitHub a conversation comment
        is an issue comment even when the conversation is a pull request. The
        pulls endpoint creates review comments, which have to be anchored to a
        line of a diff.
        """
        response = await self._request(
            "POST",
            f"/repos/{repo}/issues/{pr_number}/comments",
            json={"body": body},
        )
        url = response.json().get("html_url", "")
        log.info("commented on %s#%s: %s", repo, pr_number, url)
        return url
