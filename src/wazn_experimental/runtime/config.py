"""Model configuration, read from a checkpoint's `config.json`.

Only what inference needs is kept. A checkpoint's config also records how it
was trained (loss weights, invariance settings...); those fields are ignored
on read, so any training run's config loads as is.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping

# The section tags of the layout the model was trained on (see `formatting`).
# Each is one added token with its own trained embedding. The order is part
# of the checkpoint: the tags get consecutive ids in this order.
TAG_TOKENS = (
    "<rules>", "</rules>",
    "<examples>", "</examples>",
    "<example>", "</example>",
    "<context>", "</context>",
    "<instruction>", "</instruction>",
    "<choice>", "</choice>",
)
# `</choice>` closes every candidate, and its hidden state is the candidate's
# representation.
CHOICE_REP_TOKEN = "</choice>"


def _known(cls, d: Mapping[str, Any] | None) -> dict[str, Any]:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in (d or {}).items() if k in names}


@dataclass
class BackboneConfig:
    name_or_path: str = "Qwen/Qwen3.5-2B-Base"
    # A hub commit hash, so the head always meets the weights it was trained on.
    revision: str | None = None
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    trust_remote_code: bool = False


@dataclass
class LoRAConfig:
    enabled: bool = False
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

    def __post_init__(self) -> None:
        self.target_modules = tuple(self.target_modules)


@dataclass
class JudgeConfig:
    """The relational judge: scores come only from comparing candidate pairs,
    and an optional NONE gate estimates whether any candidate fits at all."""

    kind: str = "relational"
    dim: int = 768
    n_layers: int = 3
    n_heads: int = 8
    ffn_mult: int = 4
    dropout: float = 0.0
    local_layers: int = 2
    relational_features: tuple[str, ...] = ("diff", "prod", "cos")
    relational_aggregate: str = "mean"
    none_gate: bool = False
    gate_inputs: tuple[str, ...] = ("pair", "selected", "mean")
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.kind != "relational":
            raise ValueError(
                f"judge kind {self.kind!r} is not supported by wazn-experimental; "
                "only 'relational' checkpoints are released"
            )
        self.relational_features = tuple(self.relational_features)
        self.gate_inputs = tuple(self.gate_inputs)
        if set(self.relational_features) - {"diff", "prod", "cos"} or not self.relational_features:
            raise ValueError(f"bad relational_features {self.relational_features}")
        if set(self.gate_inputs) - {"pair", "selected", "mean", "contrast"} or not self.gate_inputs:
            raise ValueError(f"bad gate_inputs {self.gate_inputs}")


@dataclass
class PromptConfig:
    """Token budgets: the shared prefix (rules, context, instruction) and each
    candidate (a label with its own examples)."""

    max_prefix_tokens: int = 8192
    max_candidate_tokens: int = 640


@dataclass
class WaznConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    normalize_choice_rep: bool = True
    # A set is called NONE when P(a valid candidate exists) < none_threshold.
    none_threshold: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.none_threshold < 1.0:
            raise ValueError(f"none_threshold must be in (0, 1), got {self.none_threshold}")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "WaznConfig":
        old_layout = {"instruction", "state_template", "question_template", "candidate_template"}
        if old_layout & set(d.get("prompt") or {}):
            raise ValueError(
                "this checkpoint was trained on the pre-tag prompt layout, which "
                "wazn-experimental cannot render"
            )
        top = {k: v for k, v in _known(cls, d).items()
               if k not in {"backbone", "lora", "judge", "prompt"}}
        return cls(
            backbone=BackboneConfig(**_known(BackboneConfig, d.get("backbone"))),
            lora=LoRAConfig(**_known(LoRAConfig, d.get("lora"))),
            judge=JudgeConfig(**_known(JudgeConfig, d.get("judge"))),
            prompt=PromptConfig(**_known(PromptConfig, d.get("prompt"))),
            **top,
        )
