"""See what the agent would say on a pull request, before it says it anywhere.

The unprompted comment has two numbers behind it, `MAX_DISTANCE` and
`MIN_CONFIDENCE`, and both started as guesses. A guess is fine for a filter
whose failure is a slightly worse answer to a question somebody asked. It is not
fine here, where the failure is a comment nobody wanted on a stranger's first
contribution.

So this measures them. `--sample` reconstructs real pandas pull requests out of
memory itself: the file paths reviewers left comments on are the paths those
pull requests changed, and the pull request description is its own summary. That
gives hundreds of genuine cases to run the decision against without a single
GitHub call, and the number worth reading is not accuracy but **how often it
speaks at all**. A setting that comments on nine pull requests in ten is
describing the weather.

    python scripts/try_review.py --paths pandas/core/groupby/groupby.py
    python scripts/try_review.py --pr 58432
    python scripts/try_review.py --sample 40
    python scripts/try_review.py --sample 40 --sensitivity
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import statistics
import sys

import httpx
from sqlalchemy import text

from precedent.agent import review as review_module
from precedent.agent.review import classify, describe, review, select
from precedent.config import get_settings
from precedent.db.engine import create_engine
from precedent.embed.provider import OpenAIEmbeddings

log = logging.getLogger("precedent.try_review")

# Pull requests reconstructed from the comments left on them. Two distinct files
# is the floor for a change with any shape to it; a one-file pull request tells
# the path filter almost nothing.
SAMPLE = text("""
    SELECT pr_number, array_agg(DISTINCT file_path) AS paths
    FROM review_comments
    WHERE repo_id = :repo_id AND file_path IS NOT NULL
    GROUP BY pr_number
    HAVING count(DISTINCT file_path) >= 2
    ORDER BY random()
    LIMIT :n
""")

TITLES = text("""
    SELECT pr_number, body
    FROM review_comments
    WHERE repo_id = :repo_id AND kind = 'pr_body' AND pr_number = ANY(:prs)
""")

SWEEP = (0.30, 0.45, 0.60, 0.75, 0.90)


def changed_files_from_github(pr_number: int, repo: str) -> tuple[str, list[str]]:
    """Fetch a real pull request's title and paths, for checking one case by hand."""
    token = get_settings().resolve_github_token()
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    with httpx.Client(base_url="https://api.github.com", timeout=30.0, headers=headers) as client:
        pull = client.get(f"/repos/{repo}/pulls/{pr_number}")
        pull.raise_for_status()
        files = client.get(f"/repos/{repo}/pulls/{pr_number}/files", params={"per_page": 100})
        files.raise_for_status()
    return pull.json()["title"], [entry["filename"] for entry in files.json()]


async def one(engine, provider, repo_id: str, title: str | None, paths: list[str]) -> None:
    touched = classify(paths)

    print(f"\n{len(paths)} files changed")
    for path in paths[:12]:
        print(f"  {path}")
    if len(paths) > 12:
        print(f"  ... and {len(paths) - 12} more")
    print(f"\ntests={touched.tests}  docs={touched.docs}  code={touched.code}")
    print(f"query: {describe(title, paths, touched)}")

    decision = await review(
        engine, provider, repo_id=repo_id, pr_number=0, title=title, paths=paths
    )

    print(f"\n{len(decision.considered)} anchored rules, nearest 15 to the change:")
    chosen = {item.rule.id for item in decision.selected}
    for rule in decision.considered[:15]:
        mark = "->" if rule.id in chosen else "  "
        verdict = review_module.applies_to(rule, paths, touched) or "does not apply to these paths"
        print(
            f" {mark} d={rule.distance:.3f} c={rule.confidence:.2f} "
            f"{rule.scope:<10} {rule.statement[:60]:<60} | {verdict}"
        )

    print("\n" + "-" * 78)
    if decision.will_speak:
        print(decision.body)
    else:
        print(f"stays quiet: {decision.silent_reason}")
    print("-" * 78)


async def sample(engine, provider, repo_id: str, count: int, sensitivity: bool) -> None:
    async with engine.connect() as conn:
        rows = (await conn.execute(SAMPLE, {"repo_id": repo_id, "n": count})).mappings().all()
        numbers = [row["pr_number"] for row in rows]
        titles = {
            row["pr_number"]: (row["body"] or "").strip().splitlines()[0][:120]
            for row in (await conn.execute(TITLES, {"repo_id": repo_id, "prs": numbers}))
            .mappings()
            .all()
        }

    print(f"{len(rows)} pull requests reconstructed from memory\n")

    # Retrieval is the expensive half and does not depend on the thresholds, so
    # it runs once and every threshold is applied to the same recall.
    cases = []
    for row in rows:
        paths = list(row["paths"])
        title = titles.get(row["pr_number"])
        touched = classify(paths)
        decision = await review(
            engine,
            provider,
            repo_id=repo_id,
            pr_number=row["pr_number"],
            title=title,
            paths=paths,
        )
        cases.append((row["pr_number"], paths, touched, decision))
        state = (
            ", ".join(f"{i.rule.scope}/{i.rule.distance:.2f}" for i in decision.selected)
            if decision.will_speak
            else "quiet"
        )
        print(f"  #{row['pr_number']:<7} {len(paths):>3} files  {state}")

    spoke = [case for case in cases if case[3].will_speak]
    print(f"\nspeaks on {len(spoke)} of {len(cases)} ({100 * len(spoke) / max(len(cases), 1):.0f}%)")
    if spoke:
        distances = [item.rule.distance for _, _, _, d in spoke for item in d.selected]
        print(
            f"selected rule distances: min {min(distances):.3f} "
            f"median {statistics.median(distances):.3f} max {max(distances):.3f}"
        )

    if not sensitivity:
        return

    scopes: dict[str, int] = {}
    for _, _, _, decision in spoke:
        for item in decision.selected:
            scopes[item.rule.scope] = scopes.get(item.rule.scope, 0) + 1
    if scopes:
        print("cited by scope: " + ", ".join(f"{k} {v}" for k, v in sorted(scopes.items())))

    if not sensitivity:
        return

    print("\nHow often it speaks, by confidence floor. Distance is not swept")
    print("because it no longer gates anything; see the review module docstring.\n")
    print(f"  {'min confidence':<16} {'speaks':>8} {'rules cited':>13}")
    original = review_module.MIN_CONFIDENCE
    try:
        for threshold in SWEEP:
            review_module.MIN_CONFIDENCE = threshold
            talked = 0
            cited = 0
            for _, paths, touched, decision in cases:
                selected = select(decision.considered, paths, touched)
                talked += bool(selected)
                cited += len(selected)
            share = 100 * talked / max(len(cases), 1)
            print(f"  {threshold:<16.2f} {f'{share:.0f}%':>8} {cited:>13}")
    finally:
        review_module.MIN_CONFIDENCE = original

    print(
        "\nCandidates were fetched at the default floor, so a higher value here"
        "\nonly narrows what was already retrieved, never widens it."
    )


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=settings.cockroach_dsn)
    parser.add_argument("--repo", default=settings.repo_slug)
    parser.add_argument("--pr", type=int, help="Read a real pull request from GitHub.")
    parser.add_argument("--paths", nargs="+", help="Changed paths, instead of a pull request.")
    parser.add_argument("--title", default=None)
    parser.add_argument("--sample", type=int, help="Run against N pull requests from memory.")
    parser.add_argument("--sensitivity", action="store_true", help="Sweep the distance threshold.")
    args = parser.parse_args()

    if not args.dsn:
        parser.error("no DSN: pass --dsn or set COCKROACH_DSN in .env")
    if not (args.pr or args.paths or args.sample):
        parser.error("pass one of --pr, --paths or --sample")

    logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s | %(message)s")

    async def run() -> int:
        engine = create_engine(args.dsn)
        provider = OpenAIEmbeddings(
            settings.openai_api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dim,
        )
        owner, _, name = args.repo.partition("/")
        try:
            async with engine.connect() as conn:
                repo_id = str(
                    (
                        await conn.execute(
                            text("SELECT id FROM repos WHERE owner = :o AND name = :n"),
                            {"o": owner, "n": name},
                        )
                    ).scalar_one()
                )

            if args.sample:
                await sample(engine, provider, repo_id, args.sample, args.sensitivity)
            else:
                title = args.title
                paths = args.paths
                if args.pr:
                    title, paths = changed_files_from_github(args.pr, args.repo)
                    print(f"#{args.pr}: {title}")
                await one(engine, provider, repo_id, title, paths)
            return 0
        finally:
            await provider.aclose()
            await engine.dispose()

    return asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())
