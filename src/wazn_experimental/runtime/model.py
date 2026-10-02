"""The Wazn model, inference only.

    (context, instruction, {label_1..label_K}) -> distribution over the K labels

A causal LM backbone reads the shared prefix once and each label as its own
suffix; the hidden state at each label's closing `</choice>` is that label's
representation, and the relational judge compares them. Nothing is
generated.

Module and parameter names match the training checkpoints and must not
change: the head file is loaded by name.
"""

from __future__ import annotations

import types

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PreTrainedTokenizerBase

from .config import WaznConfig
from .judge import RelationalJudge
from .tokenization import build_tokenizer, tag_token_ids

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


# --------------------------------------------------------------------------
# KV cache broadcast
#
# The prefix is encoded once and its cache is broadcast to every branch that
# continues from it. Hybrid backbones (Qwen3.5: linear-attention layers
# between attention layers) also carry recurrent state, broadcast the same
# way; such a cache is consumed by the one pass that reads it.
# --------------------------------------------------------------------------


def _cache_layers(cache) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if hasattr(cache, "layers"):
        return [(layer.keys, layer.values) for layer in cache.layers]
    return list(zip(cache.key_cache, cache.value_cache))


def _is_recurrent_layer(layer) -> bool:
    return hasattr(layer, "conv_states") and hasattr(layer, "recurrent_states")


def is_hybrid_cache(cache) -> bool:
    return hasattr(cache, "layers") and any(_is_recurrent_layer(l) for l in cache.layers)


def _branch_conv_update(self, conv_states, state_idx: int = 0, conv_kernel_size=None, **kwargs):
    # The branch pass never reads the cache again, so return the context
    # without writing the new window back into the shared tensor.
    if not self.has_previous_state[state_idx]:
        raise RuntimeError("branched recurrent cache has no prefix state to continue from")
    return torch.cat([self.conv_states[state_idx], conv_states], dim=-1)


def _branch_recurrent_update(self, recurrent_states, state_idx: int = 0, **kwargs):
    return recurrent_states


def _expand_hybrid_cache(cache, repeats: torch.Tensor):
    idx = torch.repeat_interleave(torch.arange(repeats.numel(), device=repeats.device), repeats)
    for layer in cache.layers:
        touched = False
        if _is_recurrent_layer(layer):
            for i in range(layer.number_of_states):
                if layer.is_conv_states_initialized[i]:
                    s = layer.conv_states[i]
                    layer.conv_states[i] = s.index_select(0, idx.to(s.device))
                if layer.is_recurrent_states_initialized[i]:
                    s = layer.recurrent_states[i]
                    layer.recurrent_states[i] = s.index_select(0, idx.to(s.device))
            layer.update_conv_state = types.MethodType(_branch_conv_update, layer)
            layer.update_recurrent_state = types.MethodType(_branch_recurrent_update, layer)
            touched = True
        if getattr(layer, "keys", None) is not None and layer.keys.numel():
            layer.keys = layer.keys.index_select(0, idx.to(layer.keys.device))
            layer.values = layer.values.index_select(0, idx.to(layer.values.device))
            touched = True
        if not touched:
            raise ValueError(f"don't know how to branch cache layer {type(layer).__name__}")
    return cache


def expand_cache(cache, repeats: torch.Tensor):
    """Broadcast a `[B, ...]` cache to `[N, ...]`, N = repeats.sum()."""
    if is_hybrid_cache(cache):
        return _expand_hybrid_cache(cache, repeats)
    from transformers import DynamicCache

    expanded = DynamicCache()
    for i, (k, v) in enumerate(_cache_layers(cache)):
        expanded.update(k.repeat_interleave(repeats, dim=0), v.repeat_interleave(repeats, dim=0), i)
    return expanded


def has_recurrent_layers(backbone: nn.Module) -> bool:
    cfg = getattr(backbone, "config", None)
    return any(t != "full_attention" for t in getattr(cfg, "layer_types", None) or [])


# --------------------------------------------------------------------------
# backbone
# --------------------------------------------------------------------------


def _vision_language_text_config(name_or_path: str, revision: str | None, trust_remote_code: bool):
    """The text config of a vision-language checkpoint, None for a plain LM."""
    cfg = AutoConfig.from_pretrained(
        name_or_path, revision=revision, trust_remote_code=trust_remote_code
    )
    text_config = getattr(cfg, "text_config", None)
    if text_config is None or getattr(cfg, "hidden_size", None) is not None:
        return None
    return text_config


def load_backbone(config: WaznConfig) -> nn.Module:
    """The bare transformer (`AutoModel`): the model never reads vocabulary
    logits, so the LM head is not loaded at all."""
    bc = config.backbone
    kwargs = dict(
        revision=bc.revision,
        attn_implementation=bc.attn_implementation,
        trust_remote_code=bc.trust_remote_code,
        dtype=DTYPES[bc.dtype],
    )
    text_config = _vision_language_text_config(bc.name_or_path, bc.revision, bc.trust_remote_code)
    if text_config is None:
        return AutoModel.from_pretrained(bc.name_or_path, **kwargs)
    # Qwen3.5 ships as a vision-language checkpoint. Handing AutoModel the
    # text config resolves to the bare text transformer; the skipped vision
    # weights would otherwise print a long unexpected-keys report.
    from transformers.utils import logging as hf_logging

    verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        return AutoModel.from_pretrained(bc.name_or_path, config=text_config, **kwargs)
    finally:
        hf_logging.set_verbosity(verbosity)


def is_head_param(name: str) -> bool:
    """Is this parameter stored in the checkpoint's head file? Everything
    outside the backbone, plus the LoRA adapters inside it."""
    return not name.startswith("backbone.") or "lora_" in name


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------


class WaznModel(nn.Module):
    def __init__(
        self,
        config: WaznConfig,
        tokenizer: PreTrainedTokenizerBase | None = None,
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        bc = config.backbone
        self.tokenizer = tokenizer or build_tokenizer(
            bc.name_or_path, bc.revision, bc.trust_remote_code
        )
        self.backbone = backbone if backbone is not None else load_backbone(config)

        tag_ids = tag_token_ids(self.tokenizer)
        self.first_tag_id = tag_ids[0]
        self.n_tags = len(tag_ids)

        d_model = self.backbone.config.hidden_size
        # The section tags' embeddings live here, not in the backbone's
        # embedding matrix (their ids may lie past it). Loaded from the head.
        self.tag_embedding = nn.Parameter(torch.zeros(self.n_tags, d_model))
        self.choice_rep_norm = nn.LayerNorm(d_model) if config.normalize_choice_rep else nn.Identity()
        self.judge = RelationalJudge(d_model, config.judge)

        self.judge_dtype = DTYPES[config.judge.dtype]
        self.choice_rep_norm.to(self.judge_dtype)
        self.judge.to(self.judge_dtype)

        if config.lora.enabled:
            self._apply_lora()
        self.requires_grad_(False)

    def _apply_lora(self) -> None:
        """Wrap the backbone exactly as training did, so the adapters in the
        head file land on the right modules. `merge_lora` folds them in."""
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            from ..errors import RuntimeExtraMissing

            raise RuntimeExtraMissing("peft") from None
        lc = self.config.lora
        self.backbone = get_peft_model(
            self.backbone,
            LoraConfig(
                r=lc.r,
                lora_alpha=lc.alpha,
                lora_dropout=lc.dropout,
                target_modules=list(lc.target_modules),
                bias="none",
                task_type=None,
            ),
        )

    def merge_lora(self) -> None:
        """Fold the adapters into the backbone weights: one matmul per
        projection instead of three."""
        if hasattr(self.backbone, "merge_and_unload"):
            self.backbone = self.backbone.merge_and_unload()

    @property
    def lora_merged(self) -> bool:
        """No LoRA wrapper left on the backbone (adapters folded in, or none)."""
        return not hasattr(self.backbone, "merge_and_unload")

    @property
    def has_gate(self) -> bool:
        return self.judge.has_gate

    @property
    def is_recurrent(self) -> bool:
        return has_recurrent_layers(self.backbone)

    def embed_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed tokens, swapping each section tag for its row of `tag_embedding`."""
        tag = input_ids - self.first_tag_id
        is_tag = (tag >= 0) & (tag < self.n_tags)
        embeds = self.backbone.get_input_embeddings()(input_ids.masked_fill(is_tag, 0))
        tags = self.tag_embedding.to(embeds.dtype)[tag.clamp(0, self.n_tags - 1)]
        return torch.where(is_tag.unsqueeze(-1), tags, embeds)

    @torch.no_grad()
    def judge_sets(
        self,
        c_flat: torch.Tensor,
        set_idx: torch.Tensor,
        slot: torch.Tensor,
        n_sets: int,
    ) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        """Score each set of candidate representations.

        `c_flat[i]` is candidate `slot[i]` of set `set_idx[i]`. Sets of
        different sizes are padded and masked; the softmax covers real slots
        only. -> (one probability vector per set, P(a valid candidate
        exists) per set, or None without a gate)
        """
        k_max = int(slot.max()) + 1
        c = c_flat.new_zeros((n_sets, k_max, c_flat.size(-1)))
        c[set_idx, slot] = c_flat
        mask = torch.zeros((n_sets, k_max), dtype=torch.bool, device=c_flat.device)
        mask[set_idx, slot] = True

        c = self.choice_rep_norm(c.to(self.judge_dtype))
        parts = self.judge(c, mask)
        p = torch.softmax(parts.scores.float(), dim=-1).cpu()
        widths = mask.sum(dim=-1).tolist()
        probs = [p[i, : int(w)] for i, w in enumerate(widths)]
        gate = None
        if parts.gate_logit is not None:
            gate = torch.sigmoid(parts.gate_logit.float()).cpu()
        return probs, gate
