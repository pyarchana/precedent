"""Apply numbered SQL migrations to a CockroachDB cluster.

Deliberately not Alembic. Migrations here are plain SQL because the schema uses
CockroachDB-specific syntax (column families, hash-sharded indexes, vector
indexes) that Alembic's Postgres dialect cannot express without escape hatches
on every statement, at which point Alembic is only buying a version table.

Runs on raw asyncpg rather than SQLAlchemy: DDL needs the simple query protocol
to send several statements at once, and there is no ORM value in a migration.

    python -m precedent.db.migrate --create-db
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import asyncpg

from precedent.config import REPO_ROOT, get_settings

log = logging.getLogger("precedent.migrate")

MIGRATIONS_DIR = REPO_ROOT / "migrations"

BOOTSTRAP = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    STRING      NOT NULL PRIMARY KEY,
    checksum   STRING      NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def _checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def discover(directory: Path) -> list[tuple[str, Path]]:
    """Migration files in lexical order, which is why they are zero-padded."""
    return [(p.stem, p) for p in sorted(directory.glob("*.sql"))]


async def _create_database_if_missing(dsn: str) -> None:
    """Connect to the cluster's default database to CREATE DATABASE."""
    parsed = urlparse(dsn)
    target = parsed.path.lstrip("/")
    if not target:
        raise ValueError(f"DSN has no database name: {dsn}")

    admin_dsn = urlunparse(parsed._replace(path="/defaultdb"))
    conn = await asyncpg.connect(admin_dsn)
    try:
        # Identifier cannot be parameterised; the value comes from our own DSN,
        # not from user input, and is quoted to survive odd names.
        await conn.execute(f'CREATE DATABASE IF NOT EXISTS "{target}"')
        log.info("ensured database %s exists", target)
    finally:
        await conn.close()


async def apply_all(dsn: str, *, directory: Path = MIGRATIONS_DIR, dry_run: bool = False) -> int:
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(BOOTSTRAP)
        applied = {
            r["version"]: r["checksum"]
            for r in await conn.fetch("SELECT version, checksum FROM schema_migrations")
        }

        pending = 0
        for version, path in discover(directory):
            sql = path.read_text(encoding="utf-8")
            checksum = _checksum(sql)

            if version in applied:
                if applied[version] != checksum:
                    raise RuntimeError(
                        f"{version} was already applied but its file has changed since "
                        f"({applied[version]} -> {checksum}). Write a new migration "
                        "instead of editing an applied one."
                    )
                log.debug("%s already applied", version)
                continue

            pending += 1
            if dry_run:
                log.info("would apply %s", version)
                continue

            log.info("applying %s", version)
            await conn.execute(sql)
            await conn.execute(
                "INSERT INTO schema_migrations (version, checksum) VALUES ($1, $2)",
                version,
                checksum,
            )

        if pending == 0:
            log.info("schema is up to date")
        return pending
    finally:
        await conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply SQL migrations.")
    parser.add_argument("--dsn", default=None, help="Defaults to COCKROACH_DSN.")
    parser.add_argument("--create-db", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s | %(message)s",
        stream=sys.stdout,
    )

    dsn = args.dsn or get_settings().cockroach_dsn
    if not dsn:
        parser.error("no DSN: pass --dsn or set COCKROACH_DSN in .env")

    async def run() -> int:
        if args.create_db:
            await _create_database_if_missing(dsn)
        return await apply_all(dsn, dry_run=args.dry_run)

    asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
