"""Ask the agent a question from the command line.

python -m precedent.agent.ask "Do I need a whatsnew note for a bug fix?"
python -m precedent.agent.ask --show-material "Where does the GH number go?"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from sqlalchemy import text

from precedent.agent.answer import answer_question, render_material
from precedent.agent.retrieve import recall
from precedent.config import get_settings
from precedent.db.engine import create_engine
from precedent.embed.provider import OpenAIEmbeddings
from precedent.extract.llm import ChatClient

log = logging.getLogger("precedent.ask")


async def main_async(args: argparse.Namespace) -> int:
    settings = get_settings()
    engine = create_engine(args.dsn)
    provider = OpenAIEmbeddings(
        settings.openai_api_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dim,
    )
    chat = ChatClient(settings.openai_api_key, model=args.model, max_spend=args.max_spend)

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

        memory = await recall(
            engine,
            provider,
            repo_id=repo_id,
            question=args.question,
            rule_k=args.rules,
            comment_k=args.comments,
        )

        if args.show_material:
            print("=" * 70)
            print(render_material(memory))
            print("=" * 70)
            print()

        result = await answer_question(chat, memory)
    finally:
        await provider.aclose()
        await chat.aclose()
        await engine.dispose()

    print(result.text)
    print()
    print(
        f"[{'answered' if result.answered else 'declined'} | "
        f"confidence {result.confidence} | "
        f"{len(memory.rules)} rules, {len(memory.comments)} comments recalled | "
        f"${chat.spent:.4f}]"
    )
    if result.cited_prs:
        print(f"cited: {', '.join('#' + str(p) for p in result.cited_prs)}")
    if result.invented_prs:
        print(f"FABRICATED CITATIONS: {result.invented_prs}")
        return 2
    if not result.answered and result.missing:
        print(f"missing: {result.missing}")
    return 0


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Ask the memory a question.")
    parser.add_argument("question")
    parser.add_argument("--dsn", default=settings.cockroach_dsn or None)
    parser.add_argument("--repo", default=settings.repo_slug)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max-spend", type=float, default=0.02)
    parser.add_argument("--rules", type=int, default=5)
    parser.add_argument("--comments", type=int, default=6)
    parser.add_argument("--show-material", action="store_true", help="Print what was recalled.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if not args.dsn:
        parser.error("no DSN: pass --dsn or set COCKROACH_DSN in .env")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)-7s | %(message)s",
        stream=sys.stderr,
    )
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
