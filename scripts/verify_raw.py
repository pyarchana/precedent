"""Check that the staged corpus is complete and internally consistent.

The transform trusts these files, so it is worth being able to prove they are
sound rather than assuming it. Pull requests are paged by CREATED_AT DESC and
GitHub assigns numbers at creation time, so PR numbers must decrease strictly
across the whole run. Any repeat or increase means two pages overlap.

    python scripts/verify_raw.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from precedent.config import get_settings
from precedent.ingest.staging import RawStore


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=settings.raw_dir)
    parser.add_argument("--repo", default=settings.repo_slug)
    args = parser.parse_args()

    store = RawStore(args.raw_dir, args.repo)

    seen: dict[int, int] = {}  # pr number -> page it first appeared on
    duplicates: list[tuple[int, int, int]] = []
    out_of_order: list[tuple[int, int, int]] = []
    page_numbers: list[int] = []
    empty_pages: list[int] = []
    previous: int | None = None
    total = 0

    for path, env in store.iter_pages():
        # "page_000123.json.gz" -> 123, for envelopes written before the
        # page_number field existed.
        page_no = env.get("page_number") or int(path.name.split("_")[1].split(".")[0])
        page_numbers.append(page_no)

        nodes = [n for n in env["data"]["repository"]["pullRequests"]["nodes"] if n]
        if not nodes:
            empty_pages.append(page_no)
            continue

        for pr in nodes:
            number = pr["number"]
            total += 1
            if number in seen:
                duplicates.append((number, seen[number], page_no))
            else:
                seen[number] = page_no
            if previous is not None and number >= previous:
                out_of_order.append((page_no, previous, number))
            previous = number

    expected = list(range(1, len(page_numbers) + 1))
    missing = sorted(set(expected) - set(page_numbers))

    print(f"pages on disk:      {len(page_numbers)}")
    print(f"PR records:         {total}")
    print(f"distinct PRs:       {len(seen)}")
    if seen:
        print(f"range:              #{max(seen)} down to #{min(seen)}")
    print(f"missing page files: {len(missing)}{'' if not missing else ' ' + str(missing[:10])}")
    print(f"empty pages:        {len(empty_pages)}")
    print(f"duplicate PRs:      {len(duplicates)}")
    print(f"ordering breaks:    {len(out_of_order)}")

    for number, first, again in duplicates[:5]:
        print(f"  PR #{number} appears on pages {first} and {again}")
    for page_no, prev, number in out_of_order[:5]:
        print(f"  page {page_no}: #{number} follows #{prev}, expected a decrease")

    bad = bool(missing or duplicates or out_of_order)
    print()
    print("corpus is INCONSISTENT" if bad else "corpus is consistent")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
