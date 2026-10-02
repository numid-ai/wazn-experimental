"""Talking to a running Wazn server.

    from wazn_experimental import Client

    with Client("http://127.0.0.1:8000") as client:
        client.wait_until_ready()
        response = client.predict(request)
        print(response.answer.choice)

Requests are validated here before they are sent, so a malformed request
raises `RequestError` without a round trip. Needs only `httpx`.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

import httpx

from .errors import RequestError, ServerError
from .request import PredictOptions, Request
from .response import Response

DEFAULT_URL = "http://127.0.0.1:8000"


class Client:
    """A synchronous client for `wazn-experimental serve`.

    `timeout` is per call, in seconds. A large request on a CPU can take a
    while, so the default is generous.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_URL,
        timeout: float = 600.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport)

    # ----------------------------------------------------------------- calls

    def predict(
        self,
        request: Request | Mapping[str, Any],
        *,
        group_size: int | None = None,
        top_k: int = 1,
        seed: int | None = None,
        usage_detail: bool = False,
    ) -> Response:
        """Answer every instruction in one request.

        `group_size`/`top_k`/`seed` answer by tournament instead; see
        `PredictOptions`.
        """
        request = Request.coerce(request)
        options = PredictOptions(group_size, top_k, seed, usage_detail)
        payload = self._post(
            "/predict", {"request": request.to_dict(), "options": options.to_dict()}
        )
        return Response.from_dict(payload)

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def info(self) -> dict[str, Any]:
        """The model behind the server: name, backbone, token budgets, limits."""
        return self._get("/info")

    def wait_until_ready(self, timeout: float = 600.0, interval: float = 1.0) -> dict[str, Any]:
        """Poll `/health` until the server answers; returns its health."""
        deadline = time.monotonic() + timeout
        while True:
            try:
                return self.health()
            except ServerError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(interval)

    # -------------------------------------------------------------- plumbing

    def _get(self, path: str) -> dict[str, Any]:
        return self._handle(lambda: self._http.get(path))

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        return self._handle(lambda: self._http.post(path, json=body))

    def _handle(self, call) -> dict[str, Any]:
        try:
            resp = call()
        except httpx.ConnectError:
            raise ServerError(
                f"cannot reach a Wazn server at {self.base_url}; start one with "
                "`wazn-experimental serve --model <model>`"
            ) from None
        except httpx.TimeoutException:
            raise ServerError(f"the server at {self.base_url} timed out") from None
        if resp.is_success:
            return resp.json()
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        if resp.status_code == 422:
            raise RequestError(detail if isinstance(detail, str) else str(detail))
        raise ServerError(f"server error {resp.status_code}: {detail}", status=resp.status_code)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
