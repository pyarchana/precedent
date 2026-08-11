"""Exercise a deployed Precedent instance and report what actually happened.

Written to run against the real URL rather than a local server, because the
things most likely to be wrong after a deploy are environmental: a missing
variable, a cold start over the timeout, a region that makes every query slow.
None of those show up in the unit tests.

The interesting cases are the negative ones. Any retrieval system can look good
answering a question its corpus covers. What distinguishes this one is refusing
a question it does not cover, refusing a correction that states no fact, and
refusing to invent a citation, so those are checked as carefully as the happy
path.

    python scripts/smoke_deployment.py https://xxxx.lambda-url.ap-south-1.on.aws
    python scripts/smoke_deployment.py <url> --write        # also corrects, and does not undo it
    python scripts/smoke_deployment.py <url> --rate-limit   # costs a minute
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

PASS, FAIL, WARN = "pass", "FAIL", "warn"


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, outcome: str, name: str, detail: str = "") -> None:
        self.rows.append((outcome, name, detail))
        mark = {PASS: "  ok  ", FAIL: " FAIL ", WARN: " warn "}[outcome]
        print(f"[{mark}] {name}")
        if detail:
            print(f"          {detail}")

    @property
    def failed(self) -> int:
        return sum(1 for outcome, _, _ in self.rows if outcome == FAIL)


def call(url: str, path: str, payload: dict | None = None, timeout: float = 60):
    """Returns (status, parsed_body_or_text, seconds)."""
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
            status = response.status
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        status = exc.code
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        # A connection that never landed. Reported as status 0 so a network
        # failure reads differently from an HTTP error the server chose to send.
        return 0, str(exc), time.monotonic() - started

    elapsed = time.monotonic() - started
    try:
        return status, json.loads(raw), elapsed
    except json.JSONDecodeError:
        return status, raw, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test a deployed instance.")
    parser.add_argument("url")
    parser.add_argument("--write", action="store_true", help="Also test correcting, then revert.")
    parser.add_argument("--rate-limit", action="store_true", help="Verify 429. Takes a minute.")
    args = parser.parse_args()

    url = args.url
    r = Report()

    # --- reachability and configuration ------------------------------------
    status, body, secs = call(url, "/health")
    if status == 200 and isinstance(body, dict) and body.get("status") == "ok":
        r.add(PASS, f"health, {secs:.2f}s", f"spent today ${body.get('spent_today_usd')}")
    elif status == 200 and isinstance(body, dict):
        r.add(FAIL, "health reports misconfigured", str(body.get("missing") or body.get("error")))
        return 1
    else:
        r.add(FAIL, f"health returned {status}", str(body)[:200])
        return 1

    # --- the page itself ----------------------------------------------------
    status, body, secs = call(url, "/")
    ok = status == 200 and isinstance(body, str) and "Precedent" in body
    r.add(PASS if ok else FAIL, f"serves the page, {secs:.2f}s", f"{len(str(body))} bytes")

    # --- memory is readable without spending anything -----------------------
    status, body, secs = call(url, "/rules?limit=5")
    if status == 200 and isinstance(body, dict) and body.get("active"):
        active, superseded = len(body["active"]), len(body.get("superseded") or [])
        r.add(PASS, f"rules readable, {secs:.2f}s", f"{active} active, {superseded} superseded")
    else:
        r.add(FAIL, f"rules returned {status}", str(body)[:200])

    # --- a question the review history covers -------------------------------
    status, body, secs = call(
        url, "/ask", {"question": "Where do I put the GitHub issue number in a test?"}
    )
    if status == 200 and isinstance(body, dict):
        cited = body.get("cited_prs") or []
        detail = f"answered={body['answered']} trustworthy={body['trustworthy']} cited={cited}"
        r.add(
            PASS if body["answered"] and body["trustworthy"] else FAIL,
            f"answers, {secs:.2f}s",
            detail,
        )
        r.add(
            PASS if body["trustworthy"] else FAIL,
            "citations verified against retrieved material",
            "no fabricated citations" if body["trustworthy"] else "FABRICATED",
        )
        session, turn = body.get("session_id"), body.get("turn")
    else:
        r.add(FAIL, f"ask returned {status}", str(body)[:200])
        session = turn = None

    # --- a question it does not cover. Refusing is the correct answer -------
    status, body, secs = call(
        url, "/ask", {"question": "What is the best way to deploy a Kubernetes operator in Rust?"}
    )
    if status == 200 and isinstance(body, dict):
        refused = not body["answered"]
        r.add(
            PASS if refused else FAIL,
            f"refuses what memory does not cover, {secs:.2f}s",
            body["answer"][:110],
        )
    else:
        r.add(FAIL, f"out-of-scope ask returned {status}", str(body)[:200])

    # --- input validation ---------------------------------------------------
    status, _, _ = call(url, "/ask", {"question": "x"})
    r.add(PASS if status == 422 else FAIL, "rejects a too-short question", f"HTTP {status}")

    # --- a correction that states no fact must be refused -------------------
    if session and turn:
        status, body, _ = call(
            url,
            "/correct",
            {
                "session_id": session,
                "turn": turn,
                "correction": "no thats wrong",
                "maintainer": "smoketest",
            },
        )
        detail = body.get("detail", "")[:110] if isinstance(body, dict) else str(body)[:110]
        if status == 403:
            # api_corrections_enabled is off, which is the right default for a
            # public URL that cannot verify who is claiming to be a maintainer.
            r.add(PASS, "corrections are disabled on this deployment", detail)
            corrections_open = False
        else:
            r.add(PASS if status == 422 else FAIL, "refuses a contentless correction", detail)
            corrections_open = True
    else:
        corrections_open = False

    # --- a real correction, then put memory back ----------------------------
    if args.write and corrections_open and session and turn:
        status, body, secs = call(
            url,
            "/correct",
            {
                "session_id": session,
                "turn": turn,
                "correction": (
                    "Not quite. The number goes on the line above the assertion it "
                    "explains, so long assertions stay readable."
                ),
                "maintainer": "smoketest",
            },
            timeout=120,
        )
        if status == 200 and isinstance(body, dict):
            retired = [body.get("corrected_statement")] + (body.get("also_retired") or [])
            retired = [x for x in retired if x]
            r.add(
                PASS,
                f"accepts a real correction, {secs:.2f}s",
                f"outcome={body['outcome']}, retired {len(retired)}",
            )
            status, after, _ = call(
                url, "/ask", {"question": "Where do I put the GitHub issue number in a test?"}
            )
            reflects = isinstance(after, dict) and "line above" in after.get("answer", "").lower()
            r.add(
                PASS if reflects else WARN,
                "the next answer reflects the correction",
                after.get("answer", "")[:110] if isinstance(after, dict) else "",
            )
            print("\n  NOTE: this wrote to memory and there is no undo. Inspect /rules.")
        else:
            r.add(FAIL, f"correction returned {status}", str(body)[:200])

    # --- the rate limit -----------------------------------------------------
    if args.rate_limit:
        codes = [call(url, "/rules?limit=1")[0] for _ in range(13)]
        r.add(
            PASS if 429 in codes else FAIL,
            "rate limit engages",
            f"codes: {codes.count(200)} x 200, {codes.count(429)} x 429",
        )

    print()
    print(f"{len(r.rows)} checks, {r.failed} failed")
    return 1 if r.failed else 0


if __name__ == "__main__":
    sys.exit(main())
