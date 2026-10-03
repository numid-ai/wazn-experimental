"""wazn-experimental: try out Wazn, a choice model, locally.

The client side imports nothing heavy:

    from wazn_experimental import Client, Request, Instruction, Label

The model itself (`Wazn`) needs the `[server]` extra and is imported on
first use:

    from wazn_experimental import Wazn
    model = Wazn.load()   # wazn-2b-v0.1, or a hub id / local path

Experimental: for testing the model, not for production. The API may change
between 0.x releases.
"""

from ._version import __version__
from .client import Client
from .errors import RequestError, RuntimeExtraMissing, ServerError, WaznError
from .request import Instruction, Label, Request, Tournament
from .response import Answer, Response, Usage

__all__ = [
    "Answer",
    "Client",
    "Instruction",
    "Label",
    "Request",
    "RequestError",
    "Response",
    "RuntimeExtraMissing",
    "ServerError",
    "Tournament",
    "Usage",
    "Wazn",
    "WaznError",
    "__version__",
]


def __getattr__(name: str):
    if name == "Wazn":
        from .runtime.api import Wazn

        return Wazn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
