"""Ingest driver: page through a repository's PRs and stage the raw responses.

Designed to be killed and restarted at any point. Every page is written to
disk before the checkpoint advances, so a crash costs at most one page and
never produces a gap.

    python -m precedent.ingest.run --log-file logs/ingest.log
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

from precedent.config import get_settings
from precedent.ingest.client import GitHubGraphQL, QueryTooExpensive
from precedent.ingest.query import (
    DEFAULT_PAGE_SIZES,
    MIN_PAGE_SIZES,
    PR_CURSOR_QUERY,
    PR_PAGE_QUERY,
)
from precedent.ingest.staging import Checkpoint, RawStore

log = logging.getLogger("precedent.ingest")

# Consecutive clean pages before we try growing the page size back.
RECOVERY_STREAK = 20


def _setup_logging(log_file: Path | None, verbose: bool) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    # -v is about our own progress, not the HTTP stack's.
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _shrink(sizes: dict[str, int]) -> dict[str, int] | None:
    """Halve every page size. Returns None if already at the floor."""
    if all(sizes[k] <= MIN_PAGE_SIZES[k] for k in sizes):
        return None
    return {k: max(MIN_PAGE_SIZES[k], v // 2) for k, v in sizes.items()}


def _grow(sizes: dict[str, int]) -> dict[str, int]:
    return {k: min(DEFAULT_PAGE_SIZES[k], max(v * 2, v + 1)) for k, v in sizes.items()}


def _truncated_prs(page: dict[str, Any]) -> list[int]:
    """PRs whose nested connections were cut off and need a targeted refetch."""
    out = []
    for pr in page["repository"]["pullRequests"]["nodes"]:
        if pr is None:
            continue
        truncated = any(
            (pr.get(conn) or {}).get("pageInfo", {}).get("hasNextPage")
            for conn in ("files", "reviews", "reviewThreads", "comments")
        )
        if not truncated:
            for thread in (pr.get("reviewThreads") or {}).get("nodes") or []:
                if thread and (thread.get("comments") or {}).get("pageInfo", {}).get("hasNextPage"):
                    truncated = True
                    break
        if truncated:
            out.append(pr["number"])
    return out


def run_ingest(
    *,
    owner: str,
    name: str,
    raw_dir: Path,
    max_pages: int | None,
    restart: bool,
) -> int:
    settings = get_settings()
    repo_slug = f"{owner}/{name}"

    store = RawStore(raw_dir, repo_slug)
    ckpt_path = raw_dir / repo_slug.replace("/", "__") / "checkpoint.json"

    if restart and ckpt_path.exists():
        log.warning("--restart given; discarding checkpoint at %s", ckpt_path)
        ckpt_path.unlink()

    ckpt = Checkpoint.load(ckpt_path, repo_slug)
    if ckpt.completed:
        log.info("checkpoint says this repo is fully ingested (%d PRs)", ckpt.prs_seen)
        return 0

    if ckpt.page_number:
        log.info(
            "resuming from page %d (%d PRs staged, oldest PR #%s)",
            ckpt.page_number,
            ckpt.prs_seen,
            ckpt.oldest_pr_number,
        )
    else:
        log.info("starting fresh ingest of %s", repo_slug)

    sizes = dict(DEFAULT_PAGE_SIZES)
    clean_streak = 0
    pages_this_run = 0
    started = time.monotonic()

    with GitHubGraphQL(settings.resolve_github_token()) as gh:
        while True:
            if max_pages is not None and pages_this_run >= max_pages:
                log.info("reached --max-pages=%d; stopping cleanly", max_pages)
                break

            page_number = ckpt.page_number + 1
            variables = {"owner": owner, "name": name, "cursor": ckpt.cursor, **sizes}

            try:
                data = gh.execute(PR_PAGE_QUERY, variables)
            except QueryTooExpensive as exc:
                smaller = _shrink(sizes)
                if smaller is not None:
                    log.warning(
                        "page %d too expensive (%s); shrinking to %s", page_number, exc, smaller
                    )
                    sizes = smaller
                    clean_streak = 0
                    continue
                log.error(
                    "page %d times out even at minimum page size; stepping over it",
                    page_number,
                )
                skipped = _step_over(gh, owner, name, ckpt)
                ckpt.truncated_prs.extend(skipped)
                ckpt.save(ckpt_path)
                sizes = dict(DEFAULT_PAGE_SIZES)
                clean_streak = 0
                continue

            conn = data["repository"]["pullRequests"]
            nodes = [n for n in conn["nodes"] if n is not None]
            page_info = conn["pageInfo"]

            store.write_page(page_number, data, cursor=ckpt.cursor, variables=variables)

            numbers = [n["number"] for n in nodes]
            truncated = _truncated_prs(data)

            ckpt.page_number = page_number
            ckpt.cursor = page_info["endCursor"]
            ckpt.prs_seen += len(nodes)
            if numbers:
                ckpt.newest_pr_number = ckpt.newest_pr_number or max(numbers)
                ckpt.oldest_pr_number = min(numbers)
            ckpt.truncated_prs.extend(truncated)
            ckpt.completed = not page_info["hasNextPage"]
            ckpt.save(ckpt_path)

            pages_this_run += 1
            clean_streak += 1

            rl = gh.last_rate_limit
            elapsed = time.monotonic() - started
            log.info(
                "page %d | +%d PRs (total %d) | oldest #%s | truncated %d | "
                "budget %s | %.1f pages/min",
                page_number,
                len(nodes),
                ckpt.prs_seen,
                ckpt.oldest_pr_number,
                len(truncated),
                f"{rl.remaining}/{rl.limit}" if rl else "?",
                pages_this_run / (elapsed / 60) if elapsed > 0 else 0.0,
            )

            if ckpt.completed:
                log.info(
                    "reached the end: %d PRs staged across %d pages",
                    ckpt.prs_seen,
                    ckpt.page_number,
                )
                break

            if clean_streak >= RECOVERY_STREAK and sizes != DEFAULT_PAGE_SIZES:
                sizes = _grow(sizes)
                clean_streak = 0
                log.info("page sizes recovered to %s", sizes)

    if ckpt.truncated_prs:
        log.warning(
            "%d PR(s) have truncated nested data and need a targeted refetch; "
            "they are recorded in %s",
            len(ckpt.truncated_prs),
            ckpt_path,
        )
    return 0


def _step_over(gh: GitHubGraphQL, owner: str, name: str, ckpt: Checkpoint) -> list[int]:
    """Advance the cursor past one PR that cannot be fetched in full."""
    data = gh.execute(
        PR_CURSOR_QUERY,
        {"owner": owner, "name": name, "cursor": ckpt.cursor, "prs": 1},
    )
    conn = data["repository"]["pullRequests"]
    nodes = [n for n in conn["nodes"] if n is not None]
    ckpt.cursor = conn["pageInfo"]["endCursor"]
    ckpt.completed = not conn["pageInfo"]["hasNextPage"]
    skipped = [n["number"] for n in nodes]
    log.warning("stepped over PR(s) %s; recorded for a targeted refetch", skipped)
    return skipped


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Stage raw GitHub PR data for a repo.")
    parser.add_argument("--owner", default=settings.target_repo_owner)
    parser.add_argument("--name", default=settings.target_repo_name)
    parser.add_argument("--raw-dir", type=Path, default=settings.raw_dir)
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Stop after N pages this run. Useful for a smoke test.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Discard the checkpoint and start from the newest PR again.",
    )
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    _setup_logging(args.log_file, args.verbose)

    try:
        return run_ingest(
            owner=args.owner,
            name=args.name,
            raw_dir=args.raw_dir,
            max_pages=args.max_pages,
            restart=args.restart,
        )
    except KeyboardInterrupt:
        log.info("interrupted; checkpoint is safe, rerun to resume")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
