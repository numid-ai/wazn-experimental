"""`Wazn`: the model, in process.

    from wazn_experimental import Wazn, Request, Instruction, Label

    model = Wazn.load()             # wazn-2b-v0.1, or a hub id / local checkpoint
    response = model.predict(request)
    model.serve(port=8000)          # or put it behind the HTTP API

`predict` takes the same arguments as `Client.predict` and returns the same
`Response`, so code moves between the two unchanged.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import torch

from .._defaults import DEFAULT_MODEL
from .._version import __version__
from ..request import PredictOptions, Request
from ..response import Response
from .engine import WaznEngine
from .loading import fast_kernels_available, load_checkpoint
from .model import WaznModel


class Wazn:
    def __init__(self, engine: WaznEngine, source: str = "", meta: dict | None = None) -> None:
        self.engine = engine
        self.source = source
        self.meta = meta or {}

    @classmethod
    def load(
        cls,
        source: str | Path = DEFAULT_MODEL,
        *,
        revision: str | None = None,
        device: str = "auto",
        dtype: str | None = "auto",
        candidate_chunk: int | None = None,
    ) -> "Wazn":
        """Load a checkpoint: a Hugging Face Hub repo id or a local directory
        (default: the current release, wazn-2b-v0.1).

        `device` is "auto" (CUDA, then Apple MPS, then CPU) or any torch
        device. `dtype` is "auto" (the checkpoint's own, bfloat16 for the
        released ones) or "bfloat16", "float16", "float32". LoRA adapters
        are merged into the backbone on load. `candidate_chunk` caps how many
        labels are encoded per forward pass on hybrid backbones; it is halved
        on out-of-memory anyway.
        """
        ckpt = load_checkpoint(source, revision=revision, device=device, dtype=dtype)
        engine = WaznEngine(ckpt.model, model_name=ckpt.name, candidate_chunk=candidate_chunk)
        return cls(engine, source=ckpt.source, meta=ckpt.meta)

    @classmethod
    def from_model(cls, model: WaznModel, name: str = "wazn", **engine_kwargs) -> "Wazn":
        return cls(WaznEngine(model, model_name=name, **engine_kwargs))

    @property
    def name(self) -> str:
        return self.engine.model_name

    def predict(
        self,
        request: Request | Mapping[str, Any],
        *,
        group_size: int | None = None,
        top_k: int = 1,
        seed: int | None = None,
        usage_detail: bool = False,
    ) -> Response:
        """Answer every instruction in one request. With `group_size`, by
        tournament (see `PredictOptions`)."""
        return self.predict_with(
            Request.coerce(request), PredictOptions(group_size, top_k, seed, usage_detail)
        )

    def predict_with(self, request: Request, options: PredictOptions) -> Response:
        start = time.perf_counter()
        response = self.engine.predict(request, options)
        # probabilities are read back to the CPU, which waits for the device
        response.prediction_seconds = round(time.perf_counter() - start, 4)
        return response

    def info(self) -> dict[str, Any]:
        model = self.engine.model
        config = model.config
        return {
            "model": self.name,
            "source": self.source,
            "library_version": __version__,
            "backbone": config.backbone.name_or_path,
            "backbone_revision": config.backbone.revision,
            "dtype": config.backbone.dtype,
            "device": str(self.engine.device),
            "lora": "merged" if config.lora.enabled and model.lora_merged else (
                "adapters" if config.lora.enabled else "none"),
            "fast_kernels": fast_kernels_available() if model.is_recurrent else None,
            "none_gate": model.has_gate,
            "none_threshold": config.none_threshold if model.has_gate else None,
            "max_prefix_tokens": config.prompt.max_prefix_tokens,
            "max_candidate_tokens": config.prompt.max_candidate_tokens,
            **{k: v for k, v in self.meta.items() if k != "name"},
        }

    def memory(self) -> dict[str, int]:
        """What this process holds on the GPU, in MiB."""
        if not torch.cuda.is_available():
            return {}
        mib = 1024**2
        return {
            "gpu_allocated_mib": round(torch.cuda.memory_allocated() / mib),
            "gpu_reserved_mib": round(torch.cuda.memory_reserved() / mib),
        }

    def serve(self, host: str = "127.0.0.1", port: int = 8000, **limits) -> None:
        """Serve this model over HTTP until interrupted. See `server.app`."""
        from ..server.app import run

        run(self, host=host, port=port, **limits)
