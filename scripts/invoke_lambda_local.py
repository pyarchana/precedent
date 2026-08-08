"""Invoke the Lambda handler in-process with a Function URL event.

Deploying to find out whether the handler works means reading CloudWatch to
learn that an import failed, which is a slow way to discover a typo. This calls
`handler(event, context)` directly with the event shape a Function URL actually
sends, so routing, the lifespan startup and JSON encoding are all exercised
before anything is uploaded.

It does not prove the package is correct, because it runs against the local
interpreter and the local Windows wheels. What it proves is that the wiring is
right, which is the part most likely to be wrong.

    python scripts/invoke_lambda_local.py
    python scripts/invoke_lambda_local.py --ask "Do I need a whatsnew note?"
"""

from __future__ import annotations

import argparse
import json
import sys
import time


def event(method: str, path: str, body: str | None = None) -> dict:
    """The subset of the Function URL payload format Mangum reads."""
    return {
        "version": "2.0",
        "routeKey": "$default",
        "rawPath": path,
        "rawQueryString": "",
        "headers": {
            "content-type": "application/json",
            "host": "example.lambda-url.ap-south-1.on.aws",
        },
        "requestContext": {
            "http": {
                "method": method,
                "path": path,
                "protocol": "HTTP/1.1",
                "sourceIp": "127.0.0.1",
            },
            "stage": "$default",
        },
        "body": body,
        "isBase64Encoded": False,
    }


class Context:
    function_name = "precedent"
    memory_limit_in_mb = 1024
    invoked_function_arn = "arn:aws:lambda:ap-south-1:000000000000:function:precedent"
    aws_request_id = "local"

    def get_remaining_time_in_millis(self) -> int:
        return 30_000


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Lambda handler locally.")
    parser.add_argument("--ask", default=None, help="Also exercise /ask, which costs a model call.")
    args = parser.parse_args()

    from precedent.api.lambda_handler import handler

    checks: list[tuple[str, dict]] = [
        ("GET /health", event("GET", "/health")),
        ("GET /rules?limit=3", event("GET", "/rules", None)),
        ("GET / (the page)", event("GET", "/")),
    ]
    if args.ask:
        checks.append(("POST /ask", event("POST", "/ask", json.dumps({"question": args.ask}))))

    failures = 0
    for label, payload in checks:
        started = time.monotonic()
        try:
            response = handler(payload, Context())
        except Exception as exc:  # noqa: BLE001 - the point is to report anything
            print(f"FAIL  {label}: {type(exc).__name__}: {exc}")
            failures += 1
            continue

        elapsed = time.monotonic() - started
        status = response.get("statusCode")
        body = response.get("body") or ""
        ok = 200 <= int(status) < 300
        failures += 0 if ok else 1
        print(f"{'ok  ' if ok else 'FAIL'}  {label}  {status}  {elapsed:.2f}s  {len(body)} bytes")
        if not ok:
            print(f"      {body[:300]}")
        elif label.startswith("POST /ask"):
            print(f"      {json.loads(body)['answer'][:140]}")

    print("\ncold start is the first line above; later requests reuse the pool")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
