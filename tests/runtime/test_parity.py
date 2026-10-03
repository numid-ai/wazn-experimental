"""Parity with the research code this library was ported from.

The same tiny weights are loaded into the research `ArbiterModel` and into
`WaznModel` (which also checks the parameter names still line up, since
checkpoints are loaded by name), and every shipped example must get the same
distribution from both engines.

Skipped unless the research package is importable: set WAZN_RESEARCH_SRC to
its `src` directory (defaults to the parent repo's when this project sits
inside it).
"""

import copy
import os
import sys
from pathlib import Path

import pytest

from wazn_experimental import Request
from wazn_experimental.runtime.api import Wazn
from wazn_experimental.runtime.model import WaznModel

from .conftest import BACKBONE, randomize, tiny_config, tiny_qwen3

SRC = Path(os.environ.get("WAZN_RESEARCH_SRC", Path(__file__).parents[3] / "src"))
EXAMPLES = sorted((Path(__file__).parents[2] / "examples" / "requests").glob("*.json"))


@pytest.fixture(scope="module")
def pair(tokenizer):
    if not (SRC / "arbiter" / "inference.py").exists():
        pytest.skip(f"research code not found at {SRC}")
    sys.path.insert(0, str(SRC))
    try:
        from arbiter.config import ArbiterConfig, BackboneConfig, JudgeConfig, PromptConfig
        from arbiter.inference import ArbiterEngine
        from arbiter.model import ArbiterModel
        from arbiter.tokenization import build_tokenizer
    except ImportError as e:
        pytest.skip(f"research code not importable: {e}")
    finally:
        sys.path.remove(str(SRC))

    ours = tiny_config()
    theirs = ArbiterConfig(
        backbone=BackboneConfig(name_or_path=BACKBONE, dtype="float32", attn_implementation="eager"),
        judge=JudgeConfig(kind="relational", dim=ours.judge.dim, n_layers=ours.judge.n_layers,
                          local_layers=ours.judge.local_layers, ffn_mult=ours.judge.ffn_mult,
                          none_gate=True, gate_inputs=ours.judge.gate_inputs),
        prompt=PromptConfig(max_prefix_tokens=ours.prompt.max_prefix_tokens,
                            max_candidate_tokens=ours.prompt.max_candidate_tokens),
    )
    research = ArbiterModel(theirs, tokenizer=build_tokenizer(BACKBONE),
                            backbone=tiny_qwen3(len(tokenizer)))
    randomize(research)  # same random head draw as ours, on their module names

    wazn_model = WaznModel(ours, tokenizer=tokenizer, backbone=copy.deepcopy(research.backbone))
    wazn_model.load_state_dict(research.state_dict(), strict=True)
    return ArbiterEngine(research.eval(), model_name="tiny"), Wazn.from_model(wazn_model.eval())


def research_format(request: Request) -> dict:
    """The research engine's request: definitions only under `criteria`,
    examples at question level as [{input, label}], and no per-question
    tournament (it takes one for the whole request, as arguments)."""
    payload = request.to_dict()
    for ins in request.instructions:
        q = payload["questions"][ins.name]
        q.pop("tournament", None)
        q["criteria"] = {lab.name: lab.definition for lab in ins.labels}
        shots = [{"input": x, "label": lab.name} for lab in ins.labels for x in lab.examples]
        if shots:
            q["examples"] = shots
    return payload


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_same_distributions_as_the_research_engine(pair, path):
    research, wazn = pair
    request = Request.from_file(path)
    tournaments = {i.tournament for i in request.instructions if i.tournament is not None}

    payload = research_format(request)
    if tournaments:
        # the research engine runs one tournament for every question, so only
        # requests whose questions share it can be compared
        assert len(tournaments) == 1 and all(i.tournament for i in request.instructions)
        t = tournaments.pop()
        theirs = research.infer_tournament(payload, t.group_size, top_k=t.top_k, seed=t.seed)
    else:
        theirs = research.infer(payload)
    ours = wazn.predict(request)

    for name, answer in theirs.answers.items():
        assert ours[name].choice == answer.choice
        for label, p in answer.probabilities.items():
            assert ours[name].probabilities[label] == pytest.approx(p, abs=1e-5)
    # the research engine splits the same total into input + output tokens
    mine, ref = ours.usage.to_dict(detail=True), theirs.usage.to_dict(detail=True)
    assert mine.pop("input_tokens") == ref.pop("input_tokens") + ref.pop("output_tokens")
    assert mine == ref
