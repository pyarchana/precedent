"""Async engine construction for CockroachDB.

CockroachDB speaks the Postgres wire protocol, but the stock `postgresql+asyncpg`
dialect still cannot drive it: on connect it calls `_get_server_version_info`,
which parses `version()` with a Postgres-shaped regex and raises AssertionError
on "CockroachDB CCL v26.2.4 ...". The failure is at connection time, so nothing
works at all, not just version-dependent features.

`sqlalchemy-cockroachdb` 2.0.4 provides `cockroachdb+asyncpg`, which fixes the
version parsing and adjusts reflection and DDL compilation. It is a real async
dialect, not a sync shim.

Two DSN details bite when moving from a local node to CockroachDB Cloud:

  * asyncpg does not accept libpq's `sslmode`. SQLAlchemy translates the common
    values, but `verify-full` needs a real SSL context pointed at the cluster CA.
  * Cloud Serverless routes by cluster name in the `options` startup parameter
    (`--cluster=<name>`). That has to be passed through connect_args, not left
    in the URL query string.
"""

from __future__ import annotations

import ssl
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

# Query parameters libpq understands and asyncpg does not.
_LIBPQ_ONLY = {"sslmode", "sslrootcert", "options", "application_name"}


@dataclass
class ParsedDsn:
    url: str
    connect_args: dict[str, Any]


def _split(dsn: str) -> tuple[Any, dict[str, Any], dict[str, str]]:
    """Common work: clean the string, pull out the libpq-only parameters.

    Surrounding whitespace and quotes are stripped because a DSN pasted into
    a .env file frequently arrives wrapped in them, and the resulting parse
    failure ("scheme is expected to be postgresql") points nowhere useful.
    """
    cleaned = dsn.strip().strip('"').strip("'")
    if not cleaned.startswith(("postgresql://", "postgres://")):
        raise ValueError(
            "COCKROACH_DSN must start with postgresql:// . "
            f"It currently starts with {cleaned[:12]!r}."
        )

    parsed = urlparse(cleaned)
    params = {k: v[0] for k, v in parse_qs(parsed.query).items()}

    connect_args: dict[str, Any] = {}

    sslmode = params.get("sslmode", "").lower()
    if sslmode in ("disable", ""):
        connect_args["ssl"] = False
    elif sslmode in ("require", "prefer", "allow"):
        # Encrypted but unverified. Fine for a local node, not for production.
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        connect_args["ssl"] = ctx
    else:
        ctx = ssl.create_default_context(cafile=params.get("sslrootcert") or None)
        connect_args["ssl"] = ctx

    if options := params.get("options"):
        # e.g. "--cluster=quiet-goose-1234"
        connect_args["server_settings"] = {"options": options}

    keep = {k: v for k, v in params.items() if k not in _LIBPQ_ONLY}
    return parsed, connect_args, keep


def parse_dsn(dsn: str) -> ParsedDsn:
    """Split a libpq-style DSN into a SQLAlchemy URL plus connect_args."""
    parsed, connect_args, keep = _split(dsn)
    url = urlunparse(parsed._replace(scheme="cockroachdb+asyncpg", query=urlencode(keep)))
    return ParsedDsn(url=url, connect_args=connect_args)


def asyncpg_dsn(dsn: str) -> ParsedDsn:
    """The same, for code that drives asyncpg directly rather than through SQLAlchemy.

    Migrations need this: asyncpg rejects libpq's `sslmode` outright, so the
    parameter has to become an SSL context passed as a keyword argument.
    """
    parsed, connect_args, keep = _split(dsn)
    url = urlunparse(parsed._replace(scheme="postgresql", query=urlencode(keep)))
    return ParsedDsn(url=url, connect_args=connect_args)


def create_engine(dsn: str, **kwargs: Any) -> AsyncEngine:
    """An engine sized for a long-lived process, not for Lambda.

    Lambda gets its own construction on Day 17; a pool that outlives a frozen
    execution context is the classic way to break serverless plus Postgres.
    """
    parsed = parse_dsn(dsn)
    options: dict[str, Any] = {
        "pool_size": 5,
        "max_overflow": 5,
        "pool_pre_ping": True,
        "pool_recycle": 300,
        "connect_args": parsed.connect_args,
    }
    options.update(kwargs)
    return create_async_engine(parsed.url, **options)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker:
    return async_sessionmaker(engine, expire_on_commit=False)
