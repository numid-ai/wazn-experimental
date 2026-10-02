"""The model runtime. Needs the `[server]` extra."""

from ..errors import RuntimeExtraMissing

for _module in ("torch", "transformers", "safetensors", "numpy"):
    try:
        __import__(_module)
    except ImportError:
        raise RuntimeExtraMissing(_module) from None
