"""Turn a training run directory into a publishable checkpoint.

    wazn-experimental export runs/qwen_3.5_2b_V0.4 dist/wazn-2b-v0.1 \
        --name wazn-2b-v0.1 --repo numid/wazn-2b-v0.1

reads the run's `config.json` and `head.pt` and writes

    config.json        inference fields only, the backbone pinned to a commit
    head.safetensors   the same tensors, without pickle
    README.md          a model card to fill in before uploading

Nothing is loaded onto a device and the backbone weights are not touched.
Upload the result with `hf upload numid/wazn-2b-v0.1 dist/wazn-2b-v0.1`.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from .._version import __version__
from .config import WaznConfig
from .model import is_head_param

FORMAT_VERSION = 1

CARD = """\
---
license: apache-2.0
base_model: {backbone}
library_name: wazn-experimental
tags:
  - wazn
  - classification
  - zero-shot-classification
---

# {name}

Wazn is a decision model: given a context, optional rules and an instruction, it
returns a probability distribution over a set of labels you supply, each with an
optional definition and few-shot examples. Nothing is generated.

**Experimental.** For trying the model out, not for production.

## Use

```bash
pip install "wazn-experimental[server]"
wazn-experimental serve --model {repo}
```

```python
from wazn_experimental import Client, Request, Instruction, Label

request = Request(
    context="My running shoes arrived in the wrong size. Can I swap them?",
    instructions=[Instruction(
        "Which team should handle this?",
        labels=[Label("returns", "Exchanges, refunds, wrong or damaged items"),
                Label("shipping", "Delivery status, delays, lost packages"),
                Label("billing", "Charges, invoices, payment problems")],
    )],
)
print(Client().predict(request).answer)
```

## Model

- Backbone: `{backbone}` at revision `{revision}`, loaded from the hub (not in this repo)
- This repo: the head ({n_params:,} parameters: relational judge, NONE gate, tag embeddings, LoRA adapters)
- Token budgets: {max_prefix} prefix tokens, {max_cand} tokens per label

## Limitations

TODO: known weak spots, training data summary, evaluation results.

## License

The head weights are released under Apache-2.0. The backbone keeps its own
license; see `{backbone}`.
"""


def backbone_revision(name_or_path: str) -> str | None:
    """The commit the hub currently serves for `name_or_path`, or None for a
    local backbone or when the hub cannot be reached."""
    if Path(name_or_path).exists():
        return None
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(name_or_path).sha
    except Exception:
        return None


def export(
    run_dir: str | Path,
    out_dir: str | Path,
    *,
    name: str | None = None,
    head: str = "head.pt",
    revision: str | None = None,
    repo: str | None = None,
) -> Path:
    run, out = Path(run_dir), Path(out_dir)
    raw = json.loads((run / "config.json").read_text())
    config = WaznConfig.from_dict(raw)

    local_backbone = Path(config.backbone.name_or_path).exists()
    if revision is None and config.backbone.revision is None and not local_backbone:
        revision = backbone_revision(config.backbone.name_or_path)
        if revision is None:
            raise SystemExit(
                f"could not resolve a commit for {config.backbone.name_or_path}; "
                "pass --backbone-revision so the head is pinned to its backbone"
            )
    if revision is not None:
        config.backbone.revision = revision

    state = torch.load(run / head, map_location="cpu", weights_only=True)
    stray = [k for k in state if not is_head_param(k)]
    if stray:
        raise SystemExit(f"{run / head} holds backbone base weights, not a head: {stray[:5]}")
    # safetensors refuses tensors that share storage
    state = {k: v.detach().contiguous().clone() for k, v in state.items()}

    name = name or out.name
    out.mkdir(parents=True, exist_ok=True)
    payload = asdict(config)
    payload["wazn"] = {
        "name": name,
        "format_version": FORMAT_VERSION,
        "exported_with": f"wazn-experimental {__version__}",
        "exported_from": run.name,
    }
    (out / "config.json").write_text(json.dumps(payload, indent=2) + "\n")

    from safetensors.torch import save_file

    save_file(state, str(out / "head.safetensors"), metadata={"format": "pt"})

    card = out / "README.md"
    if not card.exists():
        card.write_text(CARD.format(
            name=name,
            repo=repo or f"<org>/{name}",
            backbone=config.backbone.name_or_path,
            revision=config.backbone.revision,
            n_params=sum(v.numel() for v in state.values()),
            max_prefix=config.prompt.max_prefix_tokens,
            max_cand=config.prompt.max_candidate_tokens,
        ))
    return out
