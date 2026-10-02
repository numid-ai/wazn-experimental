"""Device and dtype selection, and the float16 safety net."""

import pytest
import torch

from wazn_experimental.runtime.api import Wazn
from wazn_experimental.runtime.loading import resolve_dtype

from .test_engine import MULTI


@pytest.mark.parametrize(
    "dtype, expected",
    [
        ("auto", "bfloat16"),  # what the checkpoint was trained in, on any device
        (None, "bfloat16"),
        ("float16", "float16"),  # an explicit choice always wins
        ("float32", "float32"),
    ],
)
def test_dtype_resolution(dtype, expected):
    assert resolve_dtype(dtype, "bfloat16") == expected


def test_non_finite_activations_fail_loudly(make_model):
    model = make_model()
    with torch.no_grad():
        model.tag_embedding.fill_(float("inf"))
    with pytest.raises(FloatingPointError, match="bfloat16"):
        Wazn.from_model(model).predict(MULTI)


def test_info_reports_the_runtime(wazn):
    info = wazn.info()
    assert info["lora"] == "none"
    assert info["fast_kernels"] is None  # tiny Qwen3 has no linear-attention layers
