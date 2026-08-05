"""Rule extraction, first pass.

Deliberately writes to files rather than into the `rules` table. The plan for
this stage is to read the output and find out how wrong it is, and rules that
are already in the database are harder to throw away than a file. Persistence,
deduplication and supersession come next, once the prompt earns it.

Costs are bounded three ways: clusters are filtered before any call, results
are cached by cluster content so re-running after a prompt change only pays
for what changed, and the client refuses to spend past a ceiling.

    python -m precedent.extract.run --limit 40 --max-spend 0.05
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import sys
from collections import Counter
from datetime import UTC, datetime

from sqlalchemy import text

from precedent.config import REPO_ROOT, get_settings
from precedent.db.engine import create_engine
from precedent.extract.cluster import build_cluster, fetch_seeds
from precedent.extract.confidence import score
from precedent.extract.llm import BudgetExhausted, ChatClient, QuotaExhausted
from precedent.extract.prompt import SYSTEM, build_messages

log = logging.getLogger("precedent.extract")

CACHE_DIR = REPO_ROOT / "data" / "extract_cache"
OUTPUT_DIR = REPO_ROOT / "data" / "extracted"


# The prompt is part of the cache key. Without it, editing the prompt and
# re-running would silently serve answers produced by the previous one, which
# would make every prompt iteration unmeasurable.
PROMPT_FINGERPRINT = hashlib.sha256(SYSTEM.encode("utf-8")).hexdigest()[:12]


def cache_key(model: str, cluster_text: str) -> str:
    digest = hashlib.sha256(f"{model}\n{PROMPT_FINGERPRINT}\n{cluster_text}".encode()).hexdigest()
    return digest[:32]


def load_cached(model: str, cluster_text: str) -> dict | None:
    path = CACHE_DIR / f"{cache_key(model, cluster_text)}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def save_cached(model: str, cluster_text: str, result: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{cache_key(model, cluster_text)}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def render_markdown(records: list[dict]) -> str:
    """A readable dump, because the point of this pass is human review."""
    # Strongest first: the audit has limited attention and should spend it
    # where the system is most confident, because that is what the agent will
    # actually cite.
    kept = sorted(
        (r for r in records if r["result"].get("is_convention")),
        key=lambda r: r.get("confidence", 0),
        reverse=True,
    )
    refused = [r for r in records if not r["result"].get("is_convention")]

    lines = [
        "# Extracted rules, first pass",
        "",
        f"Generated {datetime.now(UTC).isoformat(timespec='seconds')}.",
        "",
        f"{len(kept)} conventions from {len(records)} clusters ({len(refused)} refused).",
        "",
        "Read these sceptically. The question for each is not whether it sounds",
        "reasonable but whether a pandas maintainer would agree it is a rule the",
        "project actually holds.",
        "",
        "## Conventions",
        "",
    ]

    for i, record in enumerate(kept, 1):
        r = record["result"]
        lines.append(f"### {i}. {r.get('statement', '(no statement)')}")
        lines.append("")
        if r.get("rationale"):
            lines.append(f"*{r['rationale']}*")
            lines.append("")
        scope = r.get("scope", "?")
        pattern = f" `{r['scope_pattern']}`" if r.get("scope_pattern") else ""
        prs = ", ".join(f"#{p}" for p in r.get("supporting_prs") or [])
        parts = record.get("confidence_parts", {})
        lines.append(f"- confidence: **{record.get('confidence', 0):.2f}**")
        lines.append(f"- scope: **{scope}**{pattern}")
        lines.append(f"- supporting PRs: {prs or 'none listed'}")
        lines.append(
            f"- evidence: {record['distinct_prs']} PRs, "
            f"{record['distinct_authors']} authors, "
            f"independence {parts.get('independence', 0):.2f}, "
            f"persistence {parts.get('persistence', 0):.2f}, "
            f"recency {parts.get('recency', 0):.2f}"
        )
        lines.append("")

    lines += ["## Refused", ""]
    for record in refused:
        reason = record["result"].get("reason", "(no reason given)")
        lines.append(f"- {reason} ({record['distinct_prs']} PRs)")

    return "\n".join(lines) + "\n"


async def extract(
    *,
    dsn: str,
    repo_slug: str,
    limit: int,
    max_spend: float,
    model: str,
    k: int,
    min_distinct_prs: int,
    offset: int,
    concurrency: int,
) -> int:
    settings = get_settings()
    engine = create_engine(dsn)
    client = ChatClient(settings.openai_api_key, model=model, max_spend=max_spend)

    owner, _, name = repo_slug.partition("/")
    records: list[dict] = []
    reasons: Counter[str] = Counter()
    claimed: set[str] = set()

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

        # Over-fetch seeds: many will be skipped for falling inside a cluster
        # already built, or for not clearing the distinct-PR filter.
        seeds = await fetch_seeds(engine, repo_id, limit=limit * 4, offset=offset)
        log.info("%d candidate seeds, ceiling $%.2f", len(seeds), max_spend)

        stop = False
        for i in range(0, len(seeds), concurrency):
            if stop or len(records) >= limit:
                break

            window = [s for s in seeds[i : i + concurrency] if s not in claimed]
            if not window:
                continue

            clusters = await asyncio.gather(
                *(build_cluster(engine, repo_id, s, k=k) for s in window)
            )

            for cluster in clusters:
                if len(records) >= limit:
                    stop = True
                    break
                if cluster.seed_id in claimed:
                    continue
                rejection = cluster.rejection_reason(min_distinct_prs=min_distinct_prs)
                if rejection is not None:
                    reasons[f"skipped: {rejection}"] += 1
                    continue

                cluster_text = cluster.render()
                result = load_cached(model, cluster_text)
                if result is not None:
                    client.usage.cached_calls += 1
                else:
                    try:
                        result = await client.complete_json(build_messages(repo_slug, cluster_text))
                    except BudgetExhausted as exc:
                        log.warning("stopping: %s", exc)
                        stop = True
                        break
                    save_cached(model, cluster_text, result)

                claimed |= cluster.comment_ids

                dates = [c.created_at for c in cluster.comments if c.created_at]
                breakdown = score(
                    distinct_authors=cluster.distinct_authors,
                    distinct_prs=cluster.distinct_prs,
                    first_evidence_at=min(dates) if dates else None,
                    last_evidence_at=max(dates) if dates else None,
                )
                records.append(
                    {
                        "seed_id": cluster.seed_id,
                        "distinct_prs": cluster.distinct_prs,
                        "distinct_authors": cluster.distinct_authors,
                        "mean_distance": cluster.mean_distance,
                        "first_evidence_at": min(dates).isoformat() if dates else None,
                        "last_evidence_at": max(dates).isoformat() if dates else None,
                        "confidence": breakdown.confidence,
                        "confidence_parts": {
                            "independence": breakdown.independence,
                            "repetition": breakdown.repetition,
                            "persistence": breakdown.persistence,
                            "recency": breakdown.recency,
                        },
                        "comment_ids": sorted(cluster.comment_ids),
                        "result": result,
                    }
                )
                if not result.get("is_convention"):
                    reasons[result.get("reason", "no reason given")[:60]] += 1

                # A run over thousands of clusters takes hours. Without this
                # there is no way to tell progress from a hang except by
                # counting files in the cache directory, which is what I
                # ended up doing.
                if len(records) % 100 == 0:
                    kept_so_far = sum(1 for r in records if r["result"].get("is_convention"))
                    log.info(
                        "%d/%d clusters | %d conventions | $%.4f of $%.2f",
                        len(records),
                        limit,
                        kept_so_far,
                        client.spent,
                        max_spend,
                    )

    except QuotaExhausted as exc:
        log.error("%s", exc)
    finally:
        await client.aclose()
        await engine.dispose()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    jsonl = OUTPUT_DIR / f"rules-{stamp}.jsonl"
    with jsonl.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    markdown = OUTPUT_DIR / f"rules-{stamp}.md"
    markdown.write_text(render_markdown(records), encoding="utf-8")

    kept = sum(1 for r in records if r["result"].get("is_convention"))
    log.info("clusters extracted: %d", len(records))
    log.info("  conventions found: %d", kept)
    log.info("  refused:           %d", len(records) - kept)
    log.info("skipped before any call: %s", dict(reasons.most_common(5)))
    log.info(
        "spend: $%.4f over %d calls (%d served from cache)",
        client.spent,
        client.usage.calls,
        client.usage.cached_calls,
    )
    log.info("wrote %s", markdown)
    return 0


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Extract rules from comment clusters.")
    parser.add_argument("--dsn", default=settings.cockroach_dsn or None)
    parser.add_argument("--repo", default=settings.repo_slug)
    parser.add_argument("--limit", type=int, default=40, help="Clusters to extract.")
    parser.add_argument("--max-spend", type=float, default=0.10, help="Hard ceiling in USD.")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--k", type=int, default=8, help="Comments per cluster.")
    parser.add_argument("--min-distinct-prs", type=int, default=3)
    parser.add_argument("--offset", type=int, default=0, help="Skip this many seeds.")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    if not args.dsn:
        parser.error("no DSN: pass --dsn or set COCKROACH_DSN in .env")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s | %(message)s",
        stream=sys.stdout,
    )
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return asyncio.run(
        extract(
            dsn=args.dsn,
            repo_slug=args.repo,
            limit=args.limit,
            max_spend=args.max_spend,
            model=args.model,
            k=args.k,
            min_distinct_prs=args.min_distinct_prs,
            offset=args.offset,
            concurrency=args.concurrency,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
