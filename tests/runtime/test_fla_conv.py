"""fla's Triton causal conv1d stands in for the `causal-conv1d` package.

Runs only where fla imports and CUDA is available (the GPU server); it must
give what transformers' PyTorch fallback gives, at the shapes the branch
passes use: the conv window concatenated in front of a short row."""

import copy
import importlib.util

import pytest
import torch
import torch.nn.functional as F

from wazn_experimental.runtime.api import Wazn
from wazn_experimental.runtime.loading import fla_causal_conv1d_fn, use_fla_causal_conv1d
from wazn_experimental.runtime.model import WaznModel

from .conftest import randomize, tiny_config
from .test_hybrid import REQUEST

fla_conv = pytest.importorskip("fla.modules.conv", reason="fla is not installed")
if not torch.cuda.is_available():
    pytest.skip("fla's kernels need CUDA", allow_module_level=True)


def reference(hidden_states, weight, bias=None, activation=None):
    """transformers' fallback, verbatim in effect."""
    seq_len = hidden_states.shape[-1]
    out = F.conv1d(hidden_states.to(weight.dtype), weight=weight.unsqueeze(1), bias=bias,
                   padding=weight.shape[-1] - 1, groups=hidden_states.shape[1])[:, :, :seq_len]
    if activation is not None:
        out = F.silu(out)
    return out.to(hidden_states.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("activation", [None, "silu"])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("batch,seq", [(1, 73), (219, 3 + 10), (40, 3 + 157)])
def test_matches_the_pytorch_fallback(dtype, activation, with_bias, batch, seq):
    torch.manual_seed(0)
    channels, kernel = 384, 4
    x = torch.randn(batch, channels, seq, device="cuda", dtype=dtype)
    w = torch.randn(channels, kernel, device="cuda", dtype=dtype) * 0.5
    b = torch.randn(channels, device="cuda", dtype=dtype) if with_bias else None
    got = fla_causal_conv1d_fn(x, w, b, activation=activation)
    want = reference(x, w, b, activation=activation)
    assert got.shape == want.shape
    tol = 1e-4 if dtype == torch.float32 else 3e-2
    torch.testing.assert_close(got.float(), want.float(), atol=tol, rtol=tol)


def tiny_hybrid(vocab_size: int):
    from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

    torch.manual_seed(0)
    return Qwen3_5TextModel(Qwen3_5TextConfig(
        vocab_size=vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_conv_kernel_dim=4, max_position_embeddings=1024,
    )).float()


@pytest.mark.skipif(importlib.util.find_spec("causal_conv1d") is not None,
                    reason="causal-conv1d is installed, so it is used instead")
def test_patches_qwen3_5_layers():
    try:
        from transformers import Qwen3_5TextConfig, Qwen3_5TextModel
        from transformers.models.qwen3_5 import modeling_qwen3_5
    except ImportError:
        pytest.skip("this transformers has no Qwen3.5")
    backbone = Qwen3_5TextModel(Qwen3_5TextConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"], num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
    ))
    original = modeling_qwen3_5.causal_conv1d_fn
    try:
        assert use_fla_causal_conv1d(backbone)
        assert modeling_qwen3_5.causal_conv1d_fn is fla_causal_conv1d_fn
    finally:
        modeling_qwen3_5.causal_conv1d_fn = original


@pytest.mark.skipif(importlib.util.find_spec("causal_conv1d") is not None,
                    reason="causal-conv1d is installed, so it is used instead")
def test_predictions_match_with_fla_conv(tokenizer):
    """End to end on CUDA, shared prefill and branched rows included."""
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5
    except ImportError:
        pytest.skip("this transformers has no Qwen3.5")
    model = randomize(WaznModel(tiny_config(), tokenizer=tokenizer,
                                backbone=tiny_hybrid(len(tokenizer)))).to("cuda")
    original = modeling_qwen3_5.causal_conv1d_fn
    try:
        want = Wazn.from_model(copy.deepcopy(model)).predict(REQUEST)
        assert use_fla_causal_conv1d(model.backbone)
        got = Wazn.from_model(model).predict(REQUEST)
    finally:
        modeling_qwen3_5.causal_conv1d_fn = original
    for name in want.answers:
        assert got[name].probabilities == pytest.approx(want[name].probabilities, abs=1e-4)
        assert got[name].none_probability == pytest.approx(want[name].none_probability, abs=1e-4)
