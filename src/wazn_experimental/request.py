"""Building requests.

A request is a context (plus optional rules) and one or more instructions to
answer about it. Each instruction carries the labels to choose between; each
label has an optional definition and optional few-shot examples of its own:

    Request(
        rules="Returns handles exchanges; billing handles charges.",
        context="My running shoes arrived in the wrong size. Can I swap them?",
        instructions=[
            Instruction(
                "Which team should handle this?",
                labels=[
                    Label("returns", "Exchanges, refunds, wrong or damaged items"),
                    Label("shipping", "Delivery status, delays, lost packages"),
                    Label("billing", "Charges, invoices, payment problems",
                          examples=["I was charged twice for one order"]),
                ],
                name="department",
            ),
        ],
    )

Everything is validated when it is built, so a mistake fails here and not
as a server error. `Request.to_dict()` is the wire format and
`Request.from_file` reads it back. In the wire format a label is its
definition string, or `{"definition": ..., "examples": [...]}` when it has
examples. Examples anywhere else are refused.

Pure Python: nothing here needs torch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .errors import RequestError

CHOICE_TYPE = "choice"
_STATE_KEYS = {"rules", "context"}
_QUESTION_KEYS = {"type", "instructions", "criteria", "true_label"}
_LABEL_KEYS = {"definition", "examples"}
_REQUEST_KEYS = {"state", "questions"}


def _text(value: Any, what: str) -> str:
    if not isinstance(value, str):
        raise RequestError(f"{what} must be a string, got {type(value).__name__}")
    return value


@dataclass
class Label:
    """One answer an instruction can take.

    `definition` says when the label applies; the model reads it in place of
    the name, so a good definition matters more than a good name. Leave it
    empty and the model reads the name itself.

    `examples` are few-shot inputs that should get this label. They are read
    by this label only, never by the others, so they help the model tell the
    labels apart without biasing the rest of the set.
    """

    name: str
    definition: str = ""
    examples: Sequence[str] = ()

    def __post_init__(self) -> None:
        self.name = _text(self.name, "a label name")
        if not self.name.strip():
            raise RequestError("a label name cannot be empty")
        self.definition = _text(self.definition or "", f"the definition of label {self.name!r}")
        if isinstance(self.examples, str):
            raise RequestError(
                f"the examples of label {self.name!r} must be a list of strings, not a string"
            )
        self.examples = [_text(e, f"an example of label {self.name!r}") for e in self.examples or ()]

    @property
    def text(self) -> str:
        """What the model reads for this label."""
        return self.definition or self.name

    def to_wire(self) -> str | dict[str, Any]:
        if not self.examples:
            return self.definition
        return {"definition": self.definition, "examples": list(self.examples)}

    @classmethod
    def from_wire(cls, name: str, value: Any) -> "Label":
        if value is None or isinstance(value, str):
            return cls(name, value or "")
        if not isinstance(value, Mapping):
            raise RequestError(
                f"label {name!r} must be a definition string or an object with "
                f"{sorted(_LABEL_KEYS)}, got {type(value).__name__}"
            )
        unknown = set(value) - _LABEL_KEYS
        if unknown:
            raise RequestError(
                f"label {name!r} has unknown fields {sorted(unknown)}; expected {sorted(_LABEL_KEYS)}"
            )
        examples = value.get("examples") or []
        if not isinstance(examples, list):
            raise RequestError(f"the examples of label {name!r} must be a list of strings")
        return cls(name, value.get("definition") or "", examples)


LabelsLike = Sequence["Label | str"] | Mapping[str, Any]


def _labels(labels: LabelsLike) -> list[Label]:
    if isinstance(labels, Mapping):
        return [Label.from_wire(name, value) for name, value in labels.items()]
    if isinstance(labels, str):
        raise RequestError("labels must be a list or a mapping, not a single string")
    out = []
    for lab in labels:
        if isinstance(lab, Label):
            out.append(lab)
        elif isinstance(lab, str):
            out.append(Label(lab))
        else:
            raise RequestError(f"a label must be a Label or a string, got {type(lab).__name__}")
    return out


_EXAMPLES_ON_LABELS = (
    'few-shot examples belong to their label: write the label as '
    '{"definition": "...", "examples": ["...", ...]}'
)


@dataclass
class Instruction:
    """A question about the context, and the labels to choose between.

    `labels` takes `Label`s, plain names, or a `{name: definition}` mapping.
    `name` keys the answer in the response; unnamed instructions are called
    `instruction_0`, `instruction_1`, ... in order. `true_label` is optional:
    when given, the response reports whether the model got it right. It is
    read after scoring and never reaches the model.
    """

    text: str
    labels: LabelsLike
    name: str | None = None
    true_label: str | None = None

    def __post_init__(self) -> None:
        self.text = _text(self.text, "an instruction")
        if not self.text.strip():
            raise RequestError("an instruction cannot be empty")
        self.labels = _labels(self.labels)
        where = f"instruction {self.name!r}" if self.name else "an instruction"

        if len(self.labels) < 2:
            raise RequestError(f"{where} needs at least 2 labels, got {len(self.labels)}")
        names = [lab.name for lab in self.labels]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise RequestError(f"{where} has duplicate labels {dupes}")
        texts = [lab.text for lab in self.labels]
        same = sorted({t for t in texts if texts.count(t) > 1})
        if same:
            # The model reads the text, so two labels with the same text are
            # the same candidate twice.
            raise RequestError(f"{where} has labels the model would read identically: {same}")
        if self.true_label is not None and self.true_label not in names:
            raise RequestError(f"{where}: true_label {self.true_label!r} is not one of {names}")

    @property
    def label_names(self) -> list[str]:
        return [lab.name for lab in self.labels]

    def label(self, name: str) -> Label:
        for lab in self.labels:
            if lab.name == name:
                return lab
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "type": CHOICE_TYPE,
            "instructions": self.text,
            "criteria": {lab.name: lab.to_wire() for lab in self.labels},
        }
        if self.true_label is not None:
            out["true_label"] = self.true_label
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], name: str | None = None) -> "Instruction":
        if not isinstance(d, Mapping):
            raise RequestError(f"question {name!r} must be an object")
        if "examples" in d:
            raise RequestError(f"question {name!r} has `examples`; {_EXAMPLES_ON_LABELS}")
        missing = {"instructions", "criteria"} - set(d)
        if missing:
            raise RequestError(f"question {name!r} is missing {sorted(missing)}")
        unknown = set(d) - _QUESTION_KEYS
        if unknown:
            raise RequestError(
                f"question {name!r} has unknown fields {sorted(unknown)}; "
                f"expected {sorted(_QUESTION_KEYS)}"
            )
        if d.get("type", CHOICE_TYPE) != CHOICE_TYPE:
            raise RequestError(f"question type {d['type']!r} is not supported; only 'choice' is")
        return cls(text=d["instructions"], labels=d["criteria"], name=name,
                   true_label=d.get("true_label"))


@dataclass
class Request:
    """A context, optional rules, and the instructions to answer about them.

    `context` is a string, or a list of parts (a document and a message about
    it, say) that are joined by a blank line. `rules` is the policy or
    procedure that should settle the answer, when there is one.
    """

    context: str | Sequence[str]
    instructions: Sequence[Instruction] | Instruction = field(default_factory=list)
    rules: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.context, str):
            if not self.context.strip():
                raise RequestError("the context cannot be empty")
        else:
            parts = list(self.context)
            if not parts or not all(isinstance(p, str) for p in parts):
                raise RequestError("the context must be a string or a non-empty list of strings")
            if not any(p.strip() for p in parts):
                raise RequestError("the context cannot be empty")
            self.context = parts
        if self.rules is not None:
            self.rules = _text(self.rules, "the rules")

        if isinstance(self.instructions, Instruction):
            self.instructions = [self.instructions]
        self.instructions = list(self.instructions)
        if not self.instructions:
            raise RequestError("a request needs at least one instruction")
        for i, ins in enumerate(self.instructions):
            if not isinstance(ins, Instruction):
                raise RequestError(f"instruction {i} is a {type(ins).__name__}, not an Instruction")
            if ins.name is None:
                ins.name = f"instruction_{i}"
        names = [ins.name for ins in self.instructions]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise RequestError(f"instruction names must be unique, got duplicates {dupes}")

    def __getitem__(self, name: str) -> Instruction:
        for ins in self.instructions:
            if ins.name == name:
                return ins
        raise KeyError(name)

    @property
    def num_labels(self) -> int:
        """Labels across every instruction: what the model has to read."""
        return sum(len(ins.labels) for ins in self.instructions)

    # ------------------------------------------------------------ wire format

    def to_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        if self.rules is not None:
            state["rules"] = self.rules
        state["context"] = self.context if isinstance(self.context, str) else list(self.context)
        return {
            "state": state,
            "questions": {ins.name: ins.to_dict() for ins in self.instructions},
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Request":
        if not isinstance(d, Mapping):
            raise RequestError("a request must be a JSON object")
        unknown = set(d) - _REQUEST_KEYS
        if unknown:
            raise RequestError(f"unknown request fields {sorted(unknown)}; expected {sorted(_REQUEST_KEYS)}")
        state = d.get("state")
        rules = None
        if isinstance(state, Mapping):
            if "examples" in state:
                raise RequestError(f"the state has `examples`; {_EXAMPLES_ON_LABELS}")
            bad = set(state) - _STATE_KEYS
            if bad:
                raise RequestError(f"unknown state sections {sorted(bad)}; expected {sorted(_STATE_KEYS)}")
            rules = state.get("rules")
            context = state.get("context")
        else:
            context = state
        if context is None:
            raise RequestError("a request needs a context (`state.context`, or `state` as a string)")
        questions = d.get("questions")
        if not isinstance(questions, Mapping):
            raise RequestError("`questions` must be an object of name -> question")
        return cls(
            context=context,
            rules=rules,
            instructions=[Instruction.from_dict(q, name=n) for n, q in questions.items()],
        )

    @classmethod
    def from_json(cls, text: str) -> "Request":
        try:
            return cls.from_dict(json.loads(text))
        except json.JSONDecodeError as e:
            raise RequestError(f"not valid JSON: {e}") from None

    @classmethod
    def from_file(cls, path: str | Path) -> "Request":
        return cls.from_json(Path(path).read_text())

    @classmethod
    def coerce(cls, value: "Request | Mapping[str, Any]") -> "Request":
        """A `Request`, or the wire-format dict of one."""
        return value if isinstance(value, Request) else cls.from_dict(value)


@dataclass
class PredictOptions:
    """How to answer. With `group_size` set the answer is a tournament: the
    labels are scored in groups of `group_size`, the `top_k` of each group
    advance, and the survivors are regrouped until one group is left. The
    model then never compares more than `group_size` labels at once, which is
    how to ask it about a large label set. `seed` shuffles the labels before
    the first round; without it the groups follow the request's order."""

    group_size: int | None = None
    top_k: int = 1
    seed: int | None = None
    usage_detail: bool = False

    def __post_init__(self) -> None:
        if self.group_size is None:
            return
        if self.group_size < 2:
            raise RequestError(f"group_size must be at least 2, got {self.group_size}")
        if not 1 <= self.top_k < self.group_size:
            raise RequestError(
                f"top_k must be in [1, group_size), got top_k={self.top_k} with "
                f"group_size={self.group_size}; otherwise a group never shrinks"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_size": self.group_size,
            "top_k": self.top_k,
            "seed": self.seed,
            "usage_detail": self.usage_detail,
        }
