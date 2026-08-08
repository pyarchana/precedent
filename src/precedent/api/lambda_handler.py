"""AWS Lambda entry point for the same ASGI application the local server runs.

Deliberately the same `app`. A separate Lambda-shaped copy of the endpoints
would be a second implementation to keep in agreement with the first, and the
whole point of the API module is that there is one place where a question turns
into an answer.

## Lifespan matters more here than locally

The engine, the embedding client and the chat client are built once in the
application's lifespan. Under uvicorn that happens at startup and nobody thinks
about it. Under Lambda it happens on cold start and then persists for the life
of the execution environment, which is what makes the second request fast: a
connection pool to a cluster in another region is expensive to build and cheap
to keep.

That is also why `lifespan="on"` is explicit rather than left at "auto". If the
lifespan protocol is skipped, `deps.engine` stays None and every request fails
on an attribute error rather than on anything that explains itself.

## What the cold start costs

The first request after an idle period pays for the import, the pool, and a
round trip to Mumbai. Several seconds. Provisioned concurrency removes it and
costs money this project does not have, so the cold start is accepted and the
page is written to look busy rather than broken while it happens.

`tiktoken` is deliberately absent from the deployment package. It is a compiled
extension that fetches its encoding table over the network the first time it is
imported, which is a poor thing to do on a cold start, and `embed/provider.py`
already falls back to a character bound without it. Questions are capped at
2,000 characters by the request model, far below either limit, so the fallback
never actually truncates anything.
"""

from __future__ import annotations

import logging

from mangum import Mangum

from precedent.api.app import app

logging.getLogger().setLevel(logging.INFO)

handler = Mangum(app, lifespan="on")
