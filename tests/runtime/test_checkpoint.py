"""The release path end to end: a training run directory (research-style
config with training fields, `head.pt` with LoRA adapters) -> `export` ->
`Wazn.load`, and the loaded model answers as the model that was saved."""

import json
from dataclasses import asdict

import pytest
import torch

from wazn_experimental.runtime.api import Wazn
from wazn_experimental.runtime.config import LoRAConfig
from wazn_experimental.runtime.export import export
from wazn_experimental.runtime.model import WaznModel, is_head_param

from .conftest import BACKBONE, randomize, tiny_config, tiny_qwen3
from .test_engine import MULTI


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory, tokenizer):
    from transformers import AutoTokenizer

    root = tmp_path_factory.mktemp("ckpt")
    backbone_dir = root / "backbone"
    tiny_qwen3(len(tokenizer)).save_pretrained(backbone_dir)
    AutoTokenizer.from_pretrained(BACKBONE).save_pretrained(backbone_dir)

    config = tiny_config()
    config.backbone.name_or_path = str(backbone_dir)
    config.lora = LoRAConfig(enabled=True, r=4, alpha=8, target_modules=("q_proj", "v_proj"))
    model = randomize(WaznModel(config, tokenizer=tokenizer,
                                backbone=tiny_qwen3(len(tokenizer))))
    with torch.no_grad():  # LoRA B starts at zero; make the adapters matter
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(std=0.2)

    run = root / "run"
    run.mkdir()
    raw = asdict(config)
    raw["backbone"].pop("revision")
    raw.update(gate_loss_weight=3.0, invariance_loss_weight=0.0, share_prefix=True)  # training-only
    raw["backbone"]["freeze"] = True
    (run / "config.json").write_text(json.dumps(raw))
    head = {k: v for k, v in model.state_dict().items() if is_head_param(k)}
    assert any("lora_" in k for k in head)
    torch.save(head, run / "head.pt")
    return run, Wazn.from_model(model, name="saved").predict(MULTI)


def test_export_then_load_answers_like_the_saved_model(run_dir, tmp_path):
    run, expected = run_dir
    out = export(run, tmp_path / "wazn-tiny", name="wazn-tiny")
    assert sorted(p.name for p in out.iterdir()) == ["README.md", "config.json", "head.safetensors"]
    meta = json.loads((out / "config.json").read_text())
    assert meta["wazn"]["name"] == "wazn-tiny" and "gate_loss_weight" not in meta

    wazn = Wazn.load(out, device="cpu")
    assert wazn.name == "wazn-tiny"
    assert wazn.info()["lora"] == "merged"
    assert not any("lora_" in n for n, _ in wazn.engine.model.named_parameters())
    got = wazn.predict(MULTI)
    for name in expected.answers:  # LoRA merged on load: same function, new rounding
        assert got[name].probabilities == pytest.approx(expected[name].probabilities, abs=1e-4)


def test_a_run_dir_loads_directly(run_dir):
    run, expected = run_dir
    got = Wazn.load(run, device="cpu").predict(MULTI)
    for name in expected.answers:
        assert got[name].probabilities == pytest.approx(expected[name].probabilities, abs=1e-4)


def test_a_head_with_missing_parameters_is_refused(run_dir, tmp_path):
    run, _ = run_dir
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "config.json").write_text((run / "config.json").read_text())
    head = torch.load(run / "head.pt", weights_only=True)
    head.pop("tag_embedding")
    torch.save(head, bad / "head.pt")
    with pytest.raises(ValueError, match="missing"):
        Wazn.load(bad, device="cpu")
