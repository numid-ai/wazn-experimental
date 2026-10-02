"""Exceptions raised by the client and the runtime."""

from __future__ import annotations


class WaznError(Exception):
    """Base class for every error this library raises on purpose."""


class RequestError(WaznError, ValueError):
    """A request that cannot be answered as written (bad labels, unknown
    fields, an impossible tournament...). Raised before anything reaches the
    model, on the client as well as on the server."""


class ServerError(WaznError):
    """The server answered with an error, or could not be reached."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class RuntimeExtraMissing(WaznError, ImportError):
    """The model runtime was used without the `[server]` extra installed."""

    def __init__(self, missing: str) -> None:
        super().__init__(
            f"running the model needs {missing!r}, which is not installed. "
            "Install the runtime with: pip install 'wazn-experimental[server]'"
        )
