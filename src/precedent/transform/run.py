"""Load staged GitHub JSON into CockroachDB.

A separate, replayable step: the ingest owns the API, this owns the database,
and neither has to re-run because the other changed.

    python -m precedent.transform.run --dsn "postgresql://root@localhost:26257/precedent?sslmode=disable"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from collections import Counter
from pathlib import Path

from precedent.config import get_settings
from precedent.db.engine import create_engine
from precedent.ingest.staging import RawStore
from precedent.transform.load import ensure_repo, refresh_contributors, write_batch
from precedent.transform.normalize import normalize_page

log = logging.getLogger("precedent.transform")


async def transform(
    *,
    dsn: str,
    raw_dir: Path,
    owner: str,
    name: str,
    batch_size: int,
    max_pages: int | None,
) -> int:
    store = RawStore(raw_dir, f"{owner}/{name}")
    engine = create_engine(dsn)

    kinds: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    pages = comments_written = rejects_written = 0
    started = time.monotonic()

    pending_comments = []
    pending_rejects = []

    try:
        repo_id = await ensure_repo(engine, owner, name)
        log.info("repo %s/%s is %s", owner, name, repo_id)

        for _path, envelope in store.iter_pages():
            if max_pages is not None and pages >= max_pages:
                break

            kept, dropped = normalize_page(envelope["data"])
            pending_comments.extend(kept)
            pending_rejects.extend(dropped)
            kinds.update(c.kind for c in kept)
            reasons.update(r.reason for r in dropped)
            pages += 1

            if len(pending_comments) + len(pending_rejects) >= batch_size:
                await write_batch(engine, repo_id, pending_comments, pending_rejects)
                comments_written += len(pending_comments)
                rejects_written += len(pending_rejects)
                pending_comments, pending_rejects = [], []

                elapsed = time.monotonic() - started
                log.info(
                    "%d pages | %d comments | %d rejected | %.0f pages/s",
                    pages,
                    comments_written,
                    rejects_written,
                    pages / elapsed if elapsed else 0,
                )

        if pending_comments or pending_rejects:
            await write_batch(engine, repo_id, pending_comments, pending_rejects)
            comments_written += len(pending_comments)
            rejects_written += len(pending_rejects)

        contributors = await refresh_contributors(engine, repo_id)
    finally:
        await engine.dispose()

    log.info("done in %.1fs", time.monotonic() - started)
    log.info("pages read:        %d", pages)
    log.info("comments loaded:   %d", comments_written)
    log.info("  by kind:         %s", dict(kinds.most_common()))
    log.info("rejected:          %d", rejects_written)
    log.info("  by reason:       %s", dict(reasons.most_common()))
    log.info("contributors:      %d", contributors)
    return 0


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Load staged GitHub JSON into CockroachDB.")
    parser.add_argument("--dsn", default=settings.cockroach_dsn or None)
    parser.add_argument("--raw-dir", type=Path, default=settings.raw_dir)
    parser.add_argument("--owner", default=settings.target_repo_owner)
    parser.add_argument("--name", default=settings.target_repo_name)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    if not args.dsn:
        parser.error("no DSN: pass --dsn or set COCKROACH_DSN in .env")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s | %(message)s",
        stream=sys.stdout,
    )

    return asyncio.run(
        transform(
            dsn=args.dsn,
            raw_dir=args.raw_dir,
            owner=args.owner,
            name=args.name,
            batch_size=args.batch_size,
            max_pages=args.max_pages,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
