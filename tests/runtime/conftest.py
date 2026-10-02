"""Tiny randomly initialised backbones. Only the tokenizer is fetched from the
hub (and cached); no real weights are ever downloaded."""

import os

import pytest

torch = pytest.importorskip("torch", reason="runtime tests need the [server] extra")

from wazn_experimental.runtime.api import Wazn  # noqa: E402
from wazn_experimental.runtime.config import (  # noqa: E402
    BackboneConfig,
    JudgeConfig,
    PromptConfig,
    WaznConfig,
)
from wazn_experimental.runtime.model import WaznModel  # noqa: E402
from wazn_experimental.runtime.tokenization import build_tokenizer  # noqa: E402

BACKBONE = os.environ.get("WAZN_TEST_BACKBONE", "Qwen/Qwen3-0.6B-Base")


def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker(pytest.mark.runtime)


def tiny_qwen3(vocab_size: int):
    from transformers import Qwen3Config, Qwen3Model

    torch.manual_seed(0)
    return Qwen3Model(Qwen3Config(
        vocab_size=vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        max_position_embeddings=1024, attn_implementation="eager",
    )).float()


def tiny_config(**prompt) -> WaznConfig:
    return WaznConfig(
        backbone=BackboneConfig(name_or_path=BACKBONE, dtype="float32", attn_implementation="eager"),
        judge=JudgeConfig(dim=32, n_layers=2, local_layers=1, ffn_mult=2,
                          none_gate=True, gate_inputs=("contrast",)),
        prompt=PromptConfig(**{"max_prefix_tokens": 256, "max_candidate_tokens": 48, **prompt}),
    )


def randomize(model: WaznModel) -> WaznModel:
    """Random head weights and tag embeddings, so outputs are not uniform."""
    torch.manual_seed(1)
    with torch.no_grad():
        model.tag_embedding.normal_(std=0.5)
        for p in model.judge.parameters():
            if p.dim() > 1:
                p.normal_(std=0.2)
    return model.eval()


@pytest.fixture(scope="session")
def tokenizer():
    try:
        return build_tokenizer(BACKBONE)
    except Exception as e:  # offline with an empty cache
        pytest.skip(f"tokenizer for {BACKBONE} unavailable: {e}")


@pytest.fixture(scope="session")
def make_model(tokenizer):
    def make(**prompt) -> WaznModel:
        return randomize(WaznModel(
            tiny_config(**prompt), tokenizer=tokenizer, backbone=tiny_qwen3(len(tokenizer)),
        ))

    return make


@pytest.fixture(scope="session")
def tiny_model(make_model):
    return make_model()


@pytest.fixture(scope="session")
def wazn(tiny_model):
    return Wazn.from_model(tiny_model, name="tiny")


@pytest.fixture(scope="session")
def wazn_flat(tiny_model):
    return Wazn.from_model(tiny_model, name="tiny", share_state=False)
