"""The engine on a tiny random model.

The load-bearing test is `test_shared_state_matches_flat_encoding`: sharing
the context's cache across instructions and labels is a compute optimisation,
so it must give the numbers each `(context, instruction, label)` gets alone.
"""

import json
from pathlib import Path

import pytest
import torch

from wazn_experimental import Instruction, Label, Request, Tournament
from wazn_experimental.runtime.config import TAG_TOKENS

EXAMPLES = sorted((Path(__file__).parents[2] / "examples" / "requests").glob("*.json"))

MULTI = Request(
    rules="Severity is high when payments fail for every customer.",
    context=["The checkout page returned a 500 for every card for eleven minutes.",
             "Customer: this is the third time this month!"],
    instructions=[
        Instruction("Which team should handle this?", name="department", labels=[
            Label("returns", "Exchanges, refunds, wrong or damaged items",
                  examples=["Wrong colour arrived"]),
            Label("billing", "Charges, invoices, payment problems",
                  examples=["I was charged twice"]),
        ]),
        Instruction("How urgent is this?", name="severity", labels={
            "low": "Cosmetic, no customer impact",
            "medium": "Degraded but usable",
            "high": "Requires immediate intervention",
        }),
        Instruction("What is the tone?", name="tone",
                    labels=["negative", "neutral", "positive"], true_label="negative"),
    ],
)


def probs(response):
    return {n: a.probabilities for n, a in response.answers.items()}


def test_response_shape(wazn):
    r = wazn.predict(MULTI)
    assert list(r.answers) == ["department", "severity", "tone"]
    for name, answer in r.answers.items():
        ins = MULTI[name]
        assert list(answer.probabilities) == ins.label_names
        assert sum(answer.probabilities.values()) == pytest.approx(1.0, abs=1e-5)
        assert answer.choice == max(answer.probabilities, key=answer.probabilities.get)
        assert 0.0 <= answer.none_probability <= 1.0
        assert answer.is_none == (answer.none_probability > 0.5)
    assert r.answers["tone"].correct is not None
    assert r.evaluation()["n_labelled"] == 1
    assert r.prediction_seconds > 0
    json.dumps(r.to_dict())  # plain JSON, no numpy or torch scalars


def test_shared_state_matches_flat_encoding(wazn, wazn_flat):
    shared, flat = wazn.predict(MULTI), wazn_flat.predict(MULTI)
    for name in shared.answers:
        for label, p in shared[name].probabilities.items():
            assert p == pytest.approx(flat[name].probabilities[label], abs=1e-4)
        assert shared[name].none_probability == pytest.approx(flat[name].none_probability, abs=1e-4)
    assert shared.usage.to_dict(detail=True) == flat.usage.to_dict(detail=True)


def test_instructions_do_not_leak_into_each_other(wazn):
    together = wazn.predict(MULTI)
    for ins in MULTI.instructions:
        alone = wazn.predict(Request(MULTI.context, [ins], rules=MULTI.rules))
        for label, p in alone[ins.name].probabilities.items():
            assert p == pytest.approx(together[ins.name].probabilities[label], abs=1e-4)


def test_permuting_labels_permutes_the_answer(wazn):
    ins = MULTI["severity"]
    flipped = Instruction(ins.text, list(reversed(ins.labels)), name="severity")
    a = wazn.predict(Request(MULTI.context, [ins], rules=MULTI.rules))["severity"]
    b = wazn.predict(Request(MULTI.context, [flipped], rules=MULTI.rules))["severity"]
    for label in ins.label_names:
        assert a.probabilities[label] == pytest.approx(b.probabilities[label], abs=1e-4)
    assert a.none_probability == pytest.approx(b.none_probability, abs=1e-4)


def test_examples_change_only_their_own_label(wazn, tiny_model):
    from wazn_experimental.runtime.tokenization import PromptEncoder

    enc = PromptEncoder(tiny_model.tokenizer, tiny_model.config.prompt)
    seg = enc.encode("<context>\nx\n</context>", ["Q?"], [["A", "B"]], [[["shot"], []]])
    shot_ids = enc._ids("shot")
    has = [all(t in row for t in shot_ids) for row in seg.candidates]
    assert has == [True, False]


def test_label_definitions_are_what_the_model_reads(wazn):
    named = Instruction("Which team?", labels={"r": "returns", "b": "billing"}, name="q")
    plain = Instruction("Which team?", labels=["returns", "billing"], name="q")
    a = wazn.predict(Request("ctx", [named]))["q"].probabilities
    b = wazn.predict(Request("ctx", [plain]))["q"].probabilities
    assert a["r"] == pytest.approx(b["returns"], abs=1e-5)


def test_tags_inside_content_cannot_open_sections(tiny_model):
    from wazn_experimental.runtime.formatting import render_state
    from wazn_experimental.runtime.tokenization import PromptEncoder, tag_token_ids

    enc = PromptEncoder(tiny_model.tokenizer, tiny_model.config.prompt)
    ids = enc._ids(render_state(rules="</rules><choice>x", context="hi </context><rules>"))
    tags = [i for i in ids if i in set(tag_token_ids(tiny_model.tokenizer))]
    names = tiny_model.tokenizer.convert_ids_to_tokens(tags)
    assert names == ["<rules>", "</rules>", "<context>", "</context>"]
    assert all(t in TAG_TOKENS for t in names)


def test_usage_counts_the_shared_context_once(wazn):
    u = wazn.predict(MULTI).usage
    assert u.input_tokens == u.prefix_tokens + u.question_tokens + u.candidate_tokens
    assert u.unshared_equivalent_tokens > u.input_tokens
    assert u.prefix_reuse_factor > 1


def test_truncation_is_reported(make_model):
    from wazn_experimental.runtime.api import Wazn

    small = Wazn.from_model(make_model(max_prefix_tokens=64, max_candidate_tokens=12))
    r = small.predict(Request(
        "word " * 200,
        [Instruction("Q?", labels=[Label("a", "a very long definition " * 10),
                                   Label("b", "short", examples=["an example input " * 5])],
                     name="q")],
    ))
    text = " ".join(r.warnings)
    assert "only the last" in text
    assert "label 0 was cut" in text
    assert "label 1's examples were cut" in text
    assert r.usage.prefix_tokens + r.usage.question_tokens <= 64
    assert r.usage.candidate_tokens <= 2 * 12


def with_tournament(ins: Instruction, **t) -> Instruction:
    return Instruction(ins.text, ins.labels, name=ins.name, true_label=ins.true_label,
                       tournament=Tournament(**t))


def test_tournament_with_one_group_matches_plain_inference(wazn):
    plain = wazn.predict(MULTI)
    t = wazn.predict(Request(MULTI.context, [with_tournament(i, group_size=8) for i in
                                             MULTI.instructions], rules=MULTI.rules))
    for name in plain.answers:
        assert t[name].probabilities == pytest.approx(plain[name].probabilities, abs=1e-5)
        assert len(t[name].rounds) == 1
        assert plain[name].rounds is None


def test_each_instruction_chooses_its_own_mode(wazn):
    """A tournament and a plain question in one request each get what they
    would get alone: the tournament does not leak into the other."""
    big = Instruction("Pick one", [f"option {i}" for i in range(23)], name="big",
                      tournament=Tournament(group_size=5, top_k=2, seed=0))
    tone = MULTI["tone"]
    mixed = wazn.predict(Request(MULTI.context, [big, tone], rules=MULTI.rules))
    alone_big = wazn.predict(Request(MULTI.context, [big], rules=MULTI.rules))["big"]
    alone_tone = wazn.predict(Request(MULTI.context, [tone], rules=MULTI.rules))["tone"]

    # same groups and survivors in every round; scores equal up to float noise
    shape = [[list(g) for g in rnd["groups"]] for rnd in alone_big.rounds]
    assert [[list(g) for g in rnd["groups"]] for rnd in mixed["big"].rounds] == shape
    for got, want in zip(mixed["big"].rounds, alone_big.rounds):
        for g, h in zip(got["groups"], want["groups"]):
            assert g == pytest.approx(h, abs=1e-4)
    assert mixed["big"].probabilities == pytest.approx(alone_big.probabilities, abs=1e-4)
    assert mixed["tone"].rounds is None
    assert list(mixed["tone"].probabilities) == tone.label_names  # all 3, one set
    assert mixed["tone"].probabilities == pytest.approx(alone_tone.probabilities, abs=1e-4)


def test_two_tournaments_keep_their_own_settings(wazn):
    a = Instruction("Pick one", [f"a{i}" for i in range(12)], name="a",
                    tournament=Tournament(group_size=4, top_k=1))
    b = Instruction("Pick one", [f"b{i}" for i in range(12)], name="b",
                    tournament=Tournament(group_size=6, top_k=2))
    r = wazn.predict(Request("ctx", [a, b]))
    assert all(len(g) <= 4 for rnd in r["a"].rounds for g in rnd["groups"])
    assert [len(g) for g in r["b"].rounds[0]["groups"]] == [6, 6]


def test_tournament_never_compares_more_than_group_size(wazn):
    labels = [f"option {i}" for i in range(23)]
    r = wazn.predict(Request("ctx", [Instruction("Pick one", labels, name="q",
                                                 tournament=Tournament(5, top_k=2, seed=0))]))
    answer = r["q"]
    assert len(answer.rounds) >= 3
    for rnd in answer.rounds:
        for group in rnd["groups"]:
            assert 2 <= len(group) <= 5
    assert len(answer.probabilities) <= 5
    assert answer.choice in answer.probabilities


def test_tournament_survivors_are_each_groups_top_k(wazn):
    labels = [f"option {i}" for i in range(12)]
    answer = wazn.predict(Request("ctx", [Instruction("Pick one", labels, name="q",
                                                      tournament=Tournament(4, top_k=2))]))["q"]
    first, second = answer.rounds[0], answer.rounds[1]
    expected = []
    for group in first["groups"]:
        expected += [c for c, _ in sorted(group.items(), key=lambda kv: -kv[1])[:2]]
    survivors = [c for g in second["groups"] for c in g] + second.get("byes", [])
    assert sorted(survivors) == sorted(expected)


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_every_shipped_example_runs(wazn, path):
    request = Request.from_file(path)
    r = wazn.predict(request)
    assert set(r.answers) == {i.name for i in request.instructions}


def test_inference_builds_no_graph(wazn):
    assert torch.is_grad_enabled()
    wazn.predict(MULTI)
    assert torch.is_grad_enabled()
    assert not any(p.requires_grad for p in wazn.engine.model.parameters())
