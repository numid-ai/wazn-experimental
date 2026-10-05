"""Finding and loading a checkpoint.

A checkpoint is a directory (local, or a Hugging Face Hub model repo) with

    config.json         the model config; its `wazn` block names the model
    head.safetensors    the head: judge, tag embeddings, LoRA adapters

The backbone's base weights are not in it: they are fetched from the hub at
the revision `config.json` pins. A training run directory with `head.pt` in
place of `head.safetensors` loads too.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import WaznConfig
from .model import WaznModel, is_head_param

HEAD_FILES = ("head.safetensors", "head.pt")

log = logging.getLogger("wazn_experimental")


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype: str | None, configured: str) -> str:
    """"auto" (or None) is the dtype the checkpoint was trained in, on every
    device."""
    return configured if dtype in (None, "auto") else dtype


def fast_kernels_available() -> bool:
    """Can transformers use the fused kernels for Qwen3.5's linear-attention
    layers? It binds `fla`'s `chunk_gated_delta_rule` whenever that imports
    (Linux + CUDA, the `[cuda]` extra); otherwise every one of those layers
    runs a reference PyTorch loop, an order of magnitude slower."""
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
    except Exception:
        return False
    return True


def fla_causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
    """transformers' `causal_conv1d_fn` on fla's Triton kernel.

    transformers reads `[batch, channels, time]` and returns the same;
    fla's `causal_conv1d` reads `[batch, time, channels]` and returns
    `(output, final_state)`. The weight is `[channels, kernel]` in both."""
    from fla.modules.conv import causal_conv1d

    out, _ = causal_conv1d(hidden_states.transpose(1, 2), weight=weight, bias=bias,
                           activation=activation)
    return out.transpose(1, 2)


def use_fla_causal_conv1d(backbone: torch.nn.Module) -> bool:
    """Point the backbone's short convolution at fla's Triton kernel.

    Without the `causal-conv1d` package (a CUDA extension that needs a
    toolkit matching torch's CUDA to build), transformers runs a PyTorch
    fallback and warns. fla, which the `[cuda]` extra installs anyway,
    ships the same operation in Triton. A real `causal-conv1d` install is
    left alone. -> whether the backbone now uses fla's kernel"""
    import importlib.util
    import sys

    if importlib.util.find_spec("causal_conv1d") is not None:
        return False
    try:
        from fla.modules.conv import causal_conv1d  # noqa: F401
    except Exception:
        return False
    modules = {sys.modules.get(type(m).__module__) for m in backbone.modules()}
    patched = False
    for module in modules:
        if module is not None and hasattr(module, "causal_conv1d_fn"):
            module.causal_conv1d_fn = fla_causal_conv1d_fn
            patched = True
    return patched


def resolve_source(source: str | Path, revision: str | None = None) -> Path:
    """A local checkpoint directory, or a hub repo id downloaded to the cache."""
    path = Path(source).expanduser()
    if path.is_dir():
        return path
    if path.exists() or str(source).startswith((".", "/", "~")):
        raise FileNotFoundError(f"{source} is not a checkpoint directory")
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=str(source),
            revision=revision,
            allow_patterns=["config.json", *HEAD_FILES],
        )
    )


def read_head(directory: Path) -> dict[str, torch.Tensor]:
    for name in HEAD_FILES:
        f = directory / name
        if not f.exists():
            continue
        if f.suffix == ".safetensors":
            from safetensors.torch import load_file

            return load_file(str(f))
        return torch.load(f, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"no head file in {directory}; expected one of {HEAD_FILES}")


@dataclass
class Checkpoint:
    model: WaznModel
    name: str
    source: str
    meta: dict[str, Any]


def load_checkpoint(
    source: str | Path,
    *,
    revision: str | None = None,
    device: str = "auto",
    dtype: str | None = "auto",
) -> Checkpoint:
    directory = resolve_source(source, revision)
    raw = json.loads((directory / "config.json").read_text())
    config = WaznConfig.from_dict(raw)
    device = resolve_device(device)
    config.backbone.dtype = resolve_dtype(dtype, config.backbone.dtype)

    model = WaznModel(config)
    if model.is_recurrent and torch.device(device).type == "cuda":
        if not fast_kernels_available():
            log.warning(
                "running on CUDA without the fused linear-attention kernels; predictions will be "
                "much slower than they need to be. Install them: pip install 'wazn-experimental[cuda]'"
            )
        else:
            use_fla_causal_conv1d(model.backbone)
    state = read_head(directory)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(f"the head file has keys this model does not: {unexpected[:10]}")
    absent = [k for k in missing if is_head_param(k)]
    if absent:
        raise ValueError(f"the head file is missing parameters: {absent[:10]}")
    model.merge_lora()
    model = model.to(device).eval()

    meta = dict(raw.get("wazn") or {})
    name = meta.get("name") or Path(str(source)).name
    return Checkpoint(model=model, name=name, source=str(source), meta=meta)
