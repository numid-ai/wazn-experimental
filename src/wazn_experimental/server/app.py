"""The HTTP API.

    POST /predict   {"request": {...}, "options": {...}} -> a response
    GET  /info      the model, its token budgets and the server's limits
    GET  /health    liveness, GPU memory, counters

Requests run one at a time: the model lives on one device, so there is
nothing to gain from two forwards at once. This is a test server: it binds to
localhost by default and has no authentication.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from fastapi import FastAPI, HTTPException

from .._version import __version__
from ..errors import RequestError
from ..request import PredictOptions, Request
from ..runtime.engine import free_memory
from .schemas import PredictBody

if TYPE_CHECKING:
    from ..runtime.api import Wazn

log = logging.getLogger("wazn_experimental.server")


@dataclass
class Limits:
    """Refuse requests that would take the server down rather than answer.
    `max_labels` counts labels across all instructions; past it, a
    tournament (`group_size`) is the way to ask about a large label set, so
    it is not applied to tournament requests."""

    max_labels: int = 512
    max_instructions: int = 64


def create_app(wazn: "Wazn", limits: Limits | None = None) -> FastAPI:
    limits = limits or Limits()
    lock = threading.Lock()
    started = time.time()
    counters = {"requests": 0, "errors": 0}

    app = FastAPI(title="Wazn (experimental)", version=__version__,
                  docs_url=None, redoc_url=None, openapi_url=None)

    def check(request: Request, options: PredictOptions) -> None:
        if len(request.instructions) > limits.max_instructions:
            raise RequestError(
                f"{len(request.instructions)} instructions; this server takes at most "
                f"{limits.max_instructions} per request"
            )
        if options.group_size is None and request.num_labels > limits.max_labels:
            raise RequestError(
                f"{request.num_labels} labels in one request; this server scores at most "
                f"{limits.max_labels} at once. Split the request, or answer by tournament "
                "(options.group_size)."
            )

    def run(request: Request, options: PredictOptions) -> dict[str, Any]:
        check(request, options)
        with lock:
            counters["requests"] += 1
            try:
                return wazn.predict_with(request, options).to_dict()
            except torch.OutOfMemoryError:
                counters["errors"] += 1
                free_memory()
                raise HTTPException(
                    503,
                    "out of memory on this request; send fewer labels, shorter "
                    "context, or answer by tournament (options.group_size)",
                ) from None
            except RequestError:
                raise
            except ValueError as e:
                # the runtime's own refusals (e.g. an instruction too long for
                # the prefix budget) are the request's fault
                counters["errors"] += 1
                raise RequestError(str(e)) from None
            except Exception as e:
                counters["errors"] += 1
                free_memory()
                log.exception("prediction failed")
                raise HTTPException(500, f"{type(e).__name__}: {e}") from None

    @app.post("/predict")
    def predict(body: PredictBody) -> dict[str, Any]:
        o = body.options
        try:
            request = Request.from_dict(body.request.model_dump(exclude_none=True))
            return run(request, PredictOptions(o.group_size, o.top_k, o.seed, o.usage_detail))
        except RequestError as e:
            raise HTTPException(422, str(e)) from None

    @app.get("/info")
    def info() -> dict[str, Any]:
        return {**wazn.info(), "limits": vars(limits)}

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": wazn.name,
            "uptime_seconds": round(time.time() - started),
            "busy": lock.locked(),
            **counters,
            "oom_retries": wazn.engine.oom_retries,
            **wazn.memory(),
        }

    return app


def run(wazn: "Wazn", host: str = "127.0.0.1", port: int = 8000, **limits) -> None:
    import uvicorn

    app = create_app(wazn, Limits(**limits))
    log.warning(
        "wazn-experimental %s serving %s on http://%s:%d. "
        "Experimental: for testing, not for production.",
        __version__, wazn.name, host, port,
    )
    if host not in {"127.0.0.1", "localhost", "::1"}:
        log.warning("listening on %s with no authentication: anyone who can reach it can use it", host)
    uvicorn.run(app, host=host, port=port, log_level="info")
