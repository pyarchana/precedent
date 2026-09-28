"""Re-analyse eval/baseline.json. No model calls, so rerunning costs nothing.

Computes what the raw counts do not: McNemar's exact test on the paired
accuracy comparison, Wilson intervals, refusal precision as well as recall, and
a per-tag breakdown. `--json` for the machine-readable form.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from collections import defaultdict

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BASELINE = REPO_ROOT / "eval" / "baseline.json"
QUESTIONS = REPO_ROOT / "eval" / "questions.yaml"

SYSTEMS = ("precedent", "baseline")


def wilson(hits: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval, which behaves at 0 and at n where normal does not."""
    if total == 0:
        return (0.0, 0.0)
    p = hits / total
    d = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / d
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for paired binary outcomes.

    `b` and `c` are the disagreements in each direction. Under the null they are
    a fair coin, so this is a two-sided sign test on b successes in b+c trials.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def load() -> tuple[dict, dict]:
    data = json.loads(BASELINE.read_text(encoding="utf-8"))
    by_system: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in data["outcomes"]:
        by_system[row["system"]][row["id"]] = row

    tags: dict[str, list[str]] = {}
    try:
        import yaml

        loaded = yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))
        items = loaded["questions"] if isinstance(loaded, dict) else loaded
        tags = {q["id"]: list(q.get("tags") or []) for q in items}
    except Exception as exc:  # noqa: BLE001 - tags are a nicety, not the point
        print(f"(no tags: {exc})\n")

    return data, {"by_system": dict(by_system), "tags": tags}


def correct(row: dict, lenient: bool) -> bool:
    return row["verdict"] == "yes" or (lenient and row["verdict"] == "partial")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    data, extra = load()
    by_system, tags = extra["by_system"], extra["tags"]
    ids = sorted(by_system["precedent"])
    answerable = [i for i in ids if by_system["precedent"][i]["answerable"]]
    unanswerable = [i for i in ids if not by_system["precedent"][i]["answerable"]]

    report: dict = {"n_questions": len(ids), "n_answerable": len(answerable)}
    out: list[str] = []

    def say(line: str = "") -> None:
        out.append(line)

    say(
        f"Evaluation recorded {data['generated_at'][:10]}, {len(ids)} questions, "
        f"${data['spend_usd']:.4f} spent."
    )
    say(f"{len(answerable)} answerable, {len(unanswerable)} deliberately not.")
    say()

    # --- accuracy, strict and lenient -------------------------------------
    say("ACCURACY on the answerable questions")
    say(f"  {'':22} {'strict':>16}  {'with partials':>16}")
    for system in SYSTEMS:
        cells = []
        for lenient in (False, True):
            hits = sum(correct(by_system[system][i], lenient) for i in answerable)
            lo, hi = wilson(hits, len(answerable))
            cells.append(f"{hits:>2}/{len(answerable)} [{lo:.0%}-{hi:.0%}]")
            report.setdefault(system, {})[f"correct_{'lenient' if lenient else 'strict'}"] = hits
        say(f"  {system:22} {cells[0]:>16}  {cells[1]:>16}")
    say()

    # --- is the difference real -------------------------------------------
    say("IS THE DIFFERENCE REAL (McNemar, paired, exact)")
    for lenient in (False, True):
        b = sum(
            correct(by_system["precedent"][i], lenient)
            and not correct(by_system["baseline"][i], lenient)
            for i in answerable
        )
        c = sum(
            correct(by_system["baseline"][i], lenient)
            and not correct(by_system["precedent"][i], lenient)
            for i in answerable
        )
        p = mcnemar_exact(b, c)
        label = "with partials" if lenient else "strict"
        say(
            f"  {label:14} precedent only {b}, baseline only {c}, "
            f"agree on {len(answerable) - b - c}  ->  p = {p:.3f}"
        )
        report[f"mcnemar_p_{'lenient' if lenient else 'strict'}"] = round(p, 4)
    say("  A difference this small on this many questions is not a result either way.")
    say()

    # --- citations ---------------------------------------------------------
    say("CITATIONS that resolve to a real discussion in the corpus")
    for system in SYSTEMS:
        cited = sum(len(by_system[system][i].get("cited") or []) for i in ids)
        good = sum(len(by_system[system][i].get("resolving") or []) for i in ids)
        lo, hi = wilson(good, cited)
        say(f"  {system:22} {good:>3}/{cited:<3} [{lo:.0%}-{hi:.0%}]")
        report.setdefault(system, {})["citations"] = {"total": cited, "resolving": good}
    say("  This is the column the evaluation actually settled.")
    say()

    # --- refusal, both directions -----------------------------------------
    say("REFUSAL")
    for system in SYSTEMS:
        tp = sum(not by_system[system][i]["answered"] for i in unanswerable)
        fp = sum(not by_system[system][i]["answered"] for i in answerable)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / len(unanswerable) if unanswerable else 0.0
        say(
            f"  {system:22} recall {tp}/{len(unanswerable)} ({recall:.0%}), "
            f"precision {tp}/{tp + fp} ({precision:.0%})"
        )
        report.setdefault(system, {})["refusal"] = {
            "correct": tp,
            "over_refused": fp,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
        }
    say("  Precision is the honest half: it counts answerable questions refused too.")
    say()

    # --- by tag ------------------------------------------------------------
    if tags:
        say("BY TAG, precedent, strict (n is small everywhere, read as counts)")
        buckets: dict[str, list[str]] = defaultdict(list)
        for i in answerable:
            for tag in tags.get(i) or ["untagged"]:
                buckets[tag].append(i)
        for tag, members in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
            hits = sum(correct(by_system["precedent"][i], False) for i in members)
            say(f"  {tag:16} {hits}/{len(members)}")

    text = "\n".join(out)
    print(json.dumps(report, indent=2) if args.json else text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
