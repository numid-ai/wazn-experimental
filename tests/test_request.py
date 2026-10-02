"""Building and validating requests. No torch needed."""

import json
from pathlib import Path

import pytest

from wazn_experimental import Instruction, Label, PredictOptions, Request, RequestError

EXAMPLES = sorted((Path(__file__).parents[1] / "examples" / "requests").glob("*.json"))


def make(**kw) -> Request:
    base = dict(
        context="My shoes arrived in the wrong size.",
        instructions=[Instruction("Which team?", labels=["returns", "billing"])],
    )
    return Request(**{**base, **kw})


def test_labels_accept_objects_names_and_mappings():
    a = Instruction("Q?", labels=[Label("x", "def x"), "y"])
    b = Instruction("Q?", labels={"x": "def x", "y": ""})
    assert [(l.name, l.definition) for l in a.labels] == [(l.name, l.definition) for l in b.labels]
    assert a.label("x").text == "def x"
    assert a.label("y").text == "y"  # no definition: the model reads the name


def test_each_label_carries_its_own_examples():
    ins = Instruction("Q?", labels=[
        Label("a", "def a", examples=["one", "three"]),
        Label("b", examples=["two"]),
        "c",
    ])
    assert [lab.examples for lab in ins.labels] == [["one", "three"], ["two"], []]
    same = Instruction("Q?", labels={"a": {"definition": "def a", "examples": ["one", "three"]},
                                     "b": {"examples": ["two"]}, "c": ""})
    assert [(lab.definition, lab.examples) for lab in same.labels] == \
        [(lab.definition, lab.examples) for lab in ins.labels]


@pytest.mark.parametrize(
    "label, match",
    [
        (lambda: Label("a", examples="one"), "list of strings, not a string"),
        (lambda: Label("a", examples=["ok", 3]), "must be a string"),
        (lambda: Instruction("Q?", {"a": {"definition": "x", "shots": []}, "b": ""}), "unknown fields"),
        (lambda: Instruction("Q?", {"a": 3, "b": ""}), "definition string or an object"),
    ],
)
def test_bad_label_examples_are_refused(label, match):
    with pytest.raises(RequestError, match=match):
        label()


def test_examples_outside_labels_are_refused():
    with pytest.raises(RequestError, match="belong to their label"):
        Request.from_dict({"state": "s", "questions": {"q": {
            "instructions": "Q?",
            "criteria": {"a": "def a", "b": ""},
            "examples": [{"input": "one", "label": "a"}],
        }}})
    with pytest.raises(RequestError, match="belong to their label"):
        Request.from_dict({"state": {"context": "s", "examples": []}, "questions": {}})


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(labels=["only"]), "at least 2"),
        (dict(labels=["a", "a"]), "duplicate"),
        (dict(labels={"a": "same", "b": "same"}), "identically"),
        (dict(labels=["a", "b"], true_label="c"), "not one of"),
        (dict(labels="ab"), "not a single string"),
    ],
)
def test_bad_instructions_are_refused_when_built(kwargs, match):
    with pytest.raises(RequestError, match=match):
        Instruction("Q?", **kwargs)


def test_unnamed_instructions_get_positional_names():
    r = make(instructions=[
        Instruction("A?", labels=["x", "y"]),
        Instruction("B?", labels=["x", "y"], name="b"),
        Instruction("C?", labels=["x", "y"]),
    ])
    assert [i.name for i in r.instructions] == ["instruction_0", "b", "instruction_2"]


def test_a_single_instruction_needs_no_list():
    r = Request("ctx", Instruction("Q?", labels=["x", "y"]))
    assert len(r.instructions) == 1


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(context="  "), "empty"),
        (dict(context=[]), "non-empty list"),
        (dict(instructions=[]), "at least one"),
        (dict(instructions=[Instruction("A?", ["x", "y"], name="n"),
                            Instruction("B?", ["x", "y"], name="n")]), "unique"),
    ],
)
def test_bad_requests_are_refused_when_built(kwargs, match):
    with pytest.raises(RequestError, match=match):
        make(**kwargs)


def test_wire_format_round_trips():
    r = make(
        rules="Returns handles exchanges.",
        context=["a document", "a message about it"],
        instructions=[Instruction(
            "Which team?",
            labels=[Label("returns", "Exchanges"), Label("billing", examples=["charged twice"])],
            name="team", true_label="returns",
        )],
    )
    d = r.to_dict()
    assert d == {
        "state": {"rules": "Returns handles exchanges.",
                  "context": ["a document", "a message about it"]},
        "questions": {"team": {
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {
                "returns": "Exchanges",  # no examples: just the definition
                "billing": {"definition": "", "examples": ["charged twice"]},
            },
            "true_label": "returns",
        }},
    }
    assert Request.from_json(json.dumps(d)).to_dict() == d


def test_a_plain_string_state_is_the_context():
    r = Request.from_dict({"state": "hello", "questions": {"q": {
        "instructions": "Q?", "criteria": ["a", "b"]}}})
    assert r.context == "hello" and r.rules is None


@pytest.mark.parametrize(
    "payload, match",
    [
        ({"questions": {"q": {"instructions": "Q?", "criteria": ["a", "b"]}}}, "context"),
        ({"state": "s", "questions": {}}, "at least one"),
        ({"state": "s", "questions": {"q": {"instructions": "Q?", "criteria": ["a", "b"],
                                            "shots": []}}}, "unknown fields"),
        ({"state": {"context": "s", "examples": []}, "questions": {}}, "belong to their label"),
        ({"state": {"context": "s", "notes": ""}, "questions": {}}, "unknown state sections"),
        ({"state": "s", "questions": {"q": {"type": "number", "instructions": "Q?",
                                            "criteria": ["a", "b"]}}}, "not supported"),
        ({"state": "s", "questions": {}, "extra": 1}, "unknown request fields"),
    ],
)
def test_bad_wire_requests_are_refused(payload, match):
    with pytest.raises(RequestError, match=match):
        Request.from_dict(payload)


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_every_shipped_example_is_a_valid_request(path):
    r = Request.from_file(path)
    assert Request.from_dict(r.to_dict()).to_dict() == r.to_dict()


@pytest.mark.parametrize("group_size, top_k", [(1, 1), (4, 4), (4, 0)])
def test_impossible_tournaments_are_refused(group_size, top_k):
    with pytest.raises(RequestError):
        PredictOptions(group_size=group_size, top_k=top_k)
