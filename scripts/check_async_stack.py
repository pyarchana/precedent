"""Day 1 go/no-go: does the async stack actually work against CockroachDB?

Answers, with evidence rather than assertion:
  1. Does SQLAlchemy async + asyncpg connect, write and read over a session?
  2. Does the ORM layer work, or only Core?
  3. Can we round-trip a VECTOR column, which the whole memory layer depends on?
  4. Does a vector index build?

    python scripts/check_async_stack.py --dsn "postgresql://root@localhost:26257/precedent?sslmode=disable"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid

import sqlalchemy
from sqlalchemy import Column, MetaData, String, Table, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from precedent.db.engine import create_engine

PASS = "  PASS  "
FAIL = "  FAIL  "

results: list[tuple[bool, str, str]] = []


def record(ok: bool, name: str, detail: str = "") -> None:
    results.append((ok, name, detail))
    print(f"[{PASS if ok else FAIL}] {name}" + (f"\n           {detail}" if detail else ""))


async def check_core(engine) -> None:
    async with engine.begin() as conn:
        version = (await conn.execute(text("SELECT version()"))).scalar_one()
    record(True, "async connect + query", version.split(" (")[0])


async def check_orm_session(engine) -> None:
    """Create a table, insert, read back, over an AsyncSession."""
    meta = MetaData()
    probe = Table(
        "async_stack_probe",
        meta,
        Column("id", String, primary_key=True),
        Column("note", String),
    )

    async with engine.begin() as conn:
        await conn.run_sync(meta.drop_all)
        await conn.run_sync(meta.create_all)

    key = str(uuid.uuid4())
    async with AsyncSession(engine) as session:
        await session.execute(probe.insert().values(id=key, note="written over an async session"))
        await session.commit()

    async with AsyncSession(engine) as session:
        got = (await session.execute(select(probe.c.note).where(probe.c.id == key))).scalar_one()

    record(got == "written over an async session", "async session insert + select", got)

    async with engine.begin() as conn:
        await conn.run_sync(meta.drop_all)


async def check_transaction_rollback(engine) -> None:
    """A failed transaction must leave nothing behind."""
    async with engine.begin() as conn:
        await conn.execute(text("CREATE TABLE IF NOT EXISTS async_txn_probe (id INT PRIMARY KEY)"))

    try:
        async with AsyncSession(engine) as session:
            await session.execute(text("INSERT INTO async_txn_probe (id) VALUES (1)"))
            await session.execute(text("INSERT INTO async_txn_probe (id) VALUES (1)"))
            await session.commit()
    except IntegrityError:
        pass  # Expected: this is the failure the check is built around.

    async with engine.begin() as conn:
        n = (await conn.execute(text("SELECT count(*) FROM async_txn_probe"))).scalar_one()
        await conn.execute(text("DROP TABLE async_txn_probe"))
    record(n == 0, "transaction rollback on constraint violation", f"rows left behind: {n}")


async def check_vector_roundtrip(engine) -> None:
    """The memory layer is worthless if embeddings do not survive the driver."""
    dim = 8
    vec = [round(i * 0.125, 3) for i in range(dim)]
    literal = "[" + ",".join(str(v) for v in vec) + "]"

    async with engine.begin() as conn:
        await conn.execute(text("DROP TABLE IF EXISTS async_vec_probe"))
        await conn.execute(
            text(f"CREATE TABLE async_vec_probe (id INT PRIMARY KEY, v VECTOR({dim}))")
        )
        # CAST rather than `::`, because SQLAlchemy's text() bind parser reads
        # the second colon in `:v::VECTOR` as the start of another parameter.
        await conn.execute(
            text(f"INSERT INTO async_vec_probe (id, v) VALUES (1, CAST(:v AS VECTOR({dim})))"),
            {"v": literal},
        )
        got = (await conn.execute(text("SELECT v FROM async_vec_probe WHERE id = 1"))).scalar_one()

    parsed = [float(x) for x in str(got).strip("[]").split(",")]
    ok = len(parsed) == dim and all(abs(a - b) < 1e-6 for a, b in zip(parsed, vec, strict=True))
    record(ok, "VECTOR round-trip", f"{type(got).__name__} -> {str(got)[:60]}")

    # Distance operators are what semantic search will actually issue.
    async with engine.begin() as conn:
        d = (
            await conn.execute(
                text(f"SELECT v <-> CAST(:q AS VECTOR({dim})) FROM async_vec_probe WHERE id = 1"),
                {"q": literal},
            )
        ).scalar_one()
    record(abs(float(d)) < 1e-6, "vector distance operator <->", f"self-distance = {d}")


async def check_vector_index(engine) -> None:
    async with engine.begin() as conn:
        try:
            await conn.execute(text("CREATE VECTOR INDEX ON async_vec_probe (v)"))
            record(True, "CREATE VECTOR INDEX", "supported")
        except Exception as exc:  # noqa: BLE001 - reporting, not handling
            record(False, "CREATE VECTOR INDEX", f"{type(exc).__name__}: {str(exc)[:160]}")
        await conn.execute(text("DROP TABLE IF EXISTS async_vec_probe"))


async def main_async(dsn: str) -> int:
    print(f"sqlalchemy {sqlalchemy.__version__}")
    engine = create_engine(dsn, pool_size=2, max_overflow=0)
    try:
        await check_core(engine)
        await check_orm_session(engine)
        await check_transaction_rollback(engine)
        await check_vector_roundtrip(engine)
        await check_vector_index(engine)
    finally:
        await engine.dispose()

    failed = [name for ok, name, _ in results if not ok]
    print()
    if failed:
        print(f"{len(failed)} check(s) failed: {', '.join(failed)}")
        return 1
    print(f"all {len(results)} checks passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", required=True)
    args = parser.parse_args()
    return asyncio.run(main_async(args.dsn))


if __name__ == "__main__":
    sys.exit(main())
