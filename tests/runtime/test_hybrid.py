"""The recurrent path, on a tiny Qwen3.5-style hybrid backbone (linear-
attention layers between full-attention layers), which is what the released
checkpoints run on. The shared prefill + per-row continuation must match
encoding each row on its own, at any chunk size."""

import pytest
import torch

from wazn_experimental import Instruction, Label, Request
from wazn_experimental.runtime.api import Wazn
from wazn_experimental.runtime.model import WaznModel

from .conftest import randomize, tiny_config

REQUEST = Request(
    rules="Severity is high when payments fail for every customer.",
    context="The checkout page returned a 500 for every card for eleven minutes.",
    instructions=[
        Instruction("How urgent is this?", name="severity",
                    labels=[Label("low", "Cosmetic", examples=["Typo on the help page"]),
                            Label("medium", "Degraded"), Label("high", "Immediate")]),
        Instruction("What is the tone?", name="tone", labels=["negative", "neutral", "positive"]),
    ],
)


@pytest.fixture(scope="module")
def hybrid(tokenizer):
    try:
        from transformers import Qwen3_5TextConfig, Qwen3_5TextModel
    except ImportError:
        pytest.skip("this transformers has no Qwen3.5")
    torch.manual_seed(0)
    backbone = Qwen3_5TextModel(Qwen3_5TextConfig(
        vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_conv_kernel_dim=4, max_position_embeddings=1024,
    )).float()
    model = randomize(WaznModel(tiny_config(), tokenizer=tokenizer, backbone=backbone))
    assert model.is_recurrent
    return model


@pytest.mark.parametrize("chunk", [None, 1, 3])
def test_recurrent_shared_path_matches_flat(hybrid, chunk):
    shared = Wazn.from_model(hybrid, candidate_chunk=chunk).predict(REQUEST)
    flat = Wazn.from_model(hybrid, share_state=False).predict(REQUEST)
    for name in shared.answers:
        for label, p in shared[name].probabilities.items():
            assert p == pytest.approx(flat[name].probabilities[label], abs=1e-4)
        assert shared[name].none_probability == pytest.approx(flat[name].none_probability, abs=1e-4)


def test_out_of_memory_halves_the_chunk_and_retries(hybrid, monkeypatch):
    wazn = Wazn.from_model(hybrid)
    expected = wazn.predict(REQUEST)
    real = wazn.engine._branch_rows
    seen = []

    def flaky(seg, rows, chunk):
        seen.append(chunk)
        if chunk > 2:
            raise torch.OutOfMemoryError("simulated")
        return real(seg, rows, chunk)

    monkeypatch.setattr(wazn.engine, "_branch_rows", flaky)
    got = wazn.predict(REQUEST)
    assert seen == [6, 3, 1]
    assert wazn.engine.oom_retries == 2
    for name in got.answers:
        assert got[name].probabilities == pytest.approx(expected[name].probabilities, abs=1e-5)
