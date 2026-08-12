"""Runtime configuration, loaded from environment / .env."""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(REPO_ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    github_token: str = ""
    target_repo_owner: str = "pandas-dev"
    target_repo_name: str = "pandas"

    raw_data_dir: Path = Field(default=Path("data/raw"))

    aws_region: str = "ap-south-1"
    s3_raw_bucket: str = ""
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""

    cockroach_dsn: str = ""

    openai_api_key: str = ""

    # The demo is a public URL in front of a paid model, funded by one $5 top-up
    # with no second one coming. These are the guards on that, and they are
    # deliberately settings rather than constants so the ceiling can be lowered
    # without a deploy.
    #
    # api_budget_usd is spend across the whole process, not per request: a
    # per-request cap bounds nothing when anyone can send a thousand requests.
    api_budget_usd: float = 1.00
    api_rate_limit_per_minute: int = 10
    # Corrections write to memory, and the deployed API has no way to verify
    # that whoever claims to be a maintainer is one. Off unless deliberately
    # switched on, because a public endpoint that rewrites memory on an
    # unverified name is a hole, not a feature. The GitHub App path does not
    # need this: GitHub signs those deliveries.
    api_corrections_enabled: bool = False
    api_cors_origins: str = "*"

    # The GitHub App. The secret is what makes a delivery trustworthy: it is
    # how the webhook proves it came from GitHub, which is what lets a comment
    # be attributed to a real person without any login at all.
    github_webhook_secret: str = ""
    github_trigger: str = "@precedent"
    # Acting on a pull request, rather than only being taught by one, needs the
    # app to authenticate as itself. RS256 over a private key, exchanged for a
    # short-lived installation token. The key is a PEM, which is multi-line, and
    # a multi-line environment variable survives neither the Lambda console nor
    # a .env file reliably, so it is accepted base64 encoded as well.
    github_app_id: str = ""
    github_private_key: str = ""
    github_private_key_b64: str = ""
    github_installation_id: str = ""
    # Off unless all four of the above are present. A half-configured app that
    # tries to post and fails on every pull request is worse than one that
    # quietly does not post.
    github_reviews_enabled: bool = True
    # 1536 dimensions, matching the VECTOR columns in migration 0001. Changing
    # the model to one with a different width means rewriting those columns.
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    @property
    def raw_dir(self) -> Path:
        """Absolute path to the raw staging directory."""
        p = self.raw_data_dir
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def repo_slug(self) -> str:
        return f"{self.target_repo_owner}/{self.target_repo_name}"

    def resolve_github_token(self) -> str:
        """Explicit token wins; otherwise borrow the one `gh` is already using.

        Falling back to `gh auth token` means the ingest runs without a separate
        PAT, as long as the gh CLI is logged in with `repo` scope.
        """
        if self.github_token:
            return self.github_token
        try:
            out = subprocess.run(
                ["gh", "auth", "token"],
                capture_output=True,
                text=True,
                timeout=15,
                check=True,
                shell=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(
                "No GITHUB_TOKEN set and `gh auth token` failed. "
                "Set GITHUB_TOKEN in .env or run `gh auth login`."
            ) from exc
        token = out.stdout.strip()
        if not token:
            raise RuntimeError("`gh auth token` returned nothing. Run `gh auth login`.")
        return token


@lru_cache
def get_settings() -> Settings:
    return Settings()
