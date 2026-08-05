"""Export rules for human audit, and apply the verdicts back.

Plan day 12 is a human reading the top rules and deciding whether a pandas
maintainer would actually agree with them. That step is not optional: the
extraction prompt reduces bad rules but does not eliminate them, and a
plausible-sounding rule the project does not hold is worse than no rule,
because the agent will cite it with a straight face.

This exports each rule together with the comments it was learned from, so the
judgement can be made from the evidence rather than from whether the sentence
sounds reasonable. Quotes are included inline precisely so the reviewer is
not tempted to skip opening them.

    python scripts/audit_rules.py --export --top 50
    # edit eval/rule_audit.yaml, setting verdict on each rule
    python scripts/audit_rules.py --apply

Applying retires rules marked "disagree". Retired is a status, not a deletion:
the rule keeps its evidence and can be reinstated by editing the file and
applying again.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import yaml
from sqlalchemy import text

from precedent.config import REPO_ROOT, get_settings
from precedent.db.engine import create_engine

log = logging.getLogger("precedent.audit")

DEFAULT_PATH = REPO_ROOT / "eval" / "rule_audit.yaml"

TOP_RULES = text("""
    SELECT id, statement, rationale, scope::STRING AS scope, scope_pattern,
           confidence, evidence_count, status::STRING AS status,
           first_evidence_at, last_evidence_at
    FROM rules
    WHERE repo_id = :repo_id AND status = 'active'
    ORDER BY confidence DESC, evidence_count DESC
    LIMIT :top
""")

EVIDENCE_FOR = text("""
    SELECT rc.pr_number, rc.author, rc.file_path, rc.body, rc.url
    FROM rule_evidence re
    JOIN review_comments rc
      ON rc.repo_id = re.repo_id AND rc.id = re.comment_id
    WHERE re.repo_id = :repo_id AND re.rule_id = :rule_id
    ORDER BY rc.created_at
    LIMIT :per_rule
""")

RETIRE = text("""
    UPDATE rules SET status = 'retired', updated_at = now()
    WHERE repo_id = :repo_id AND id = :rule_id
""")

REINSTATE = text("""
    UPDATE rules SET status = 'active', updated_at = now()
    WHERE repo_id = :repo_id AND id = :rule_id AND superseded_by IS NULL
""")

HEADER = """\
# Rule audit
#
# For each rule below, decide whether a pandas maintainer would agree that the
# project actually holds it. Judge from the evidence quotes, not from whether
# the sentence sounds sensible.
#
# Set verdict to one of:
#   agree      the project holds this. Leave it active.
#   disagree   it does not. Applying will retire it.
#   unsure     leave it active but flag it. Worth a second opinion.
#
# The note field is free text and is kept in this file rather than the
# database, so the reasoning survives in version control.
#
# Then: python scripts/audit_rules.py --apply
"""


async def export(engine, repo_id: str, path: Path, top: int, per_rule: int) -> int:
    async with engine.connect() as conn:
        rules = (await conn.execute(TOP_RULES, {"repo_id": repo_id, "top": top})).mappings().all()

        existing = {}
        if path.exists():
            previous = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            existing = {r["id"]: r for r in previous.get("rules", [])}

        out = []
        for rule in rules:
            rule_id = str(rule["id"])
            evidence = (
                (
                    await conn.execute(
                        EVIDENCE_FOR,
                        {"repo_id": repo_id, "rule_id": rule_id, "per_rule": per_rule},
                    )
                )
                .mappings()
                .all()
            )
            prior = existing.get(rule_id, {})
            out.append(
                {
                    "id": rule_id,
                    "statement": rule["statement"],
                    "scope": rule["scope"],
                    "scope_pattern": rule["scope_pattern"],
                    "confidence": round(float(rule["confidence"]), 3),
                    "evidence_count": rule["evidence_count"],
                    # Verdicts already recorded survive a re-export, so the
                    # audit can be done in several sittings.
                    "verdict": prior.get("verdict", "unreviewed"),
                    "note": prior.get("note", ""),
                    "evidence": [
                        {
                            "pr": e["pr_number"],
                            "author": e["author"],
                            "file": e["file_path"],
                            "quote": " ".join((e["body"] or "").split())[:320],
                        }
                        for e in evidence
                    ],
                }
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump({"rules": out}, sort_keys=False, allow_unicode=True, width=100)
    path.write_text(HEADER + body, encoding="utf-8")

    reviewed = sum(1 for r in out if r["verdict"] != "unreviewed")
    log.info("wrote %d rules to %s (%d already reviewed)", len(out), path, reviewed)
    return 0


async def apply(engine, repo_id: str, path: Path) -> int:
    if not path.exists():
        log.error("%s does not exist. Run with --export first.", path)
        return 1

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rules = data.get("rules", [])

    counts = {"agree": 0, "disagree": 0, "unsure": 0, "unreviewed": 0}
    async with engine.begin() as conn:
        for rule in rules:
            verdict = (rule.get("verdict") or "unreviewed").strip().lower()
            counts[verdict] = counts.get(verdict, 0) + 1
            if verdict == "disagree":
                await conn.execute(RETIRE, {"repo_id": repo_id, "rule_id": rule["id"]})
            elif verdict in ("agree", "unsure"):
                # Reinstates anything retired by an earlier verdict that has
                # since been changed. Never touches a superseded rule.
                await conn.execute(REINSTATE, {"repo_id": repo_id, "rule_id": rule["id"]})

    log.info("verdicts: %s", counts)
    if counts["unreviewed"]:
        log.warning("%d rules are still unreviewed and remain active", counts["unreviewed"])
    return 0


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Export rules for audit, or apply verdicts.")
    parser.add_argument("--dsn", default=settings.cockroach_dsn or None)
    parser.add_argument("--repo", default=settings.repo_slug)
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--per-rule", type=int, default=4, help="Evidence quotes per rule.")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    if args.export == args.apply:
        parser.error("choose exactly one of --export or --apply")
    if not args.dsn:
        parser.error("no DSN: pass --dsn or set COCKROACH_DSN in .env")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s | %(message)s",
        stream=sys.stdout,
    )

    async def run() -> int:
        engine = create_engine(args.dsn)
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
            if args.export:
                return await export(engine, repo_id, args.path, args.top, args.per_rule)
            return await apply(engine, repo_id, args.path)
        finally:
            await engine.dispose()

    return asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())
