"""Run a SQL statement against a cluster and print the result.

The local node can be reached with `docker exec ... cockroach sql`, but the
Cloud cluster cannot, and installing a second CLI to run one statement is not
worth it. This uses the connection handling the application already has, so it
works against either.

    python scripts/sql.py "SELECT count(*) FROM review_comments"
    python scripts/sql.py --local "SHOW INDEXES FROM review_comments"
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from sqlalchemy import text

from precedent.config import get_settings
from precedent.db.engine import create_engine

LOCAL_DSN = "postgresql://root@localhost:26257/precedent?sslmode=disable"


async def run(dsn: str, statement: str) -> int:
    engine = create_engine(dsn, pool_size=1, max_overflow=0)
    # EXPLAIN output is drawn with box characters, which the Windows console
    # encodes as cp1252 by default and then dies on. Reconfiguring beats
    # stripping them: the tree structure is the readable part of a query plan.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        async with engine.begin() as conn:
            result = await conn.execute(text(statement))
            if result.returns_rows:
                rows = result.mappings().all()
                if not rows:
                    print("(no rows)")
                    return 0
                headers = list(rows[0].keys())
                widths = [max(len(h), max(len(str(r[h])) for r in rows)) for h in headers]
                print("  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True)))
                print("  ".join("-" * w for w in widths))
                for row in rows:
                    print(
                        "  ".join(
                            str(row[h]).ljust(w) for h, w in zip(headers, widths, strict=True)
                        )
                    )
                print(f"\n({len(rows)} rows)")
            else:
                print(f"OK, {result.rowcount} rows affected")
    finally:
        await engine.dispose()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one SQL statement.")
    parser.add_argument("statement")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Target the local Docker node instead of COCKROACH_DSN.",
    )
    parser.add_argument("--dsn", default=None)
    args = parser.parse_args()

    dsn = args.dsn or (LOCAL_DSN if args.local else get_settings().cockroach_dsn)
    if not dsn:
        parser.error("no DSN: use --local, pass --dsn, or set COCKROACH_DSN")
    return asyncio.run(run(dsn, args.statement))


if __name__ == "__main__":
    sys.exit(main())
