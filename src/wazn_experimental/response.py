"""Reading responses.

A response carries one `Answer` per instruction, keyed by the instruction's
name: the chosen label, the full distribution over labels, and, for a model
with a NONE gate, how likely it is that no label applies at all.

The same classes come back from `Client.predict` and from the in-process
`Wazn.predict`, and `to_dict()` is the JSON the server sends.

Pure Python: nothing here needs torch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping


@dataclass
class Answer:
    """The model's answer to one instruction.

    `probabilities` sums to 1 over the labels it covers, and `choice` is its
    argmax. Those probabilities are *relative*: the model always ranks the
    labels, even when none of them fits. `none_probability` is the separate
    estimate that no label applies, and `is_none` is that estimate against the
    model's threshold; both are None for a model without a NONE gate.

    In a tournament (`group_size` set) `probabilities` covers the final group
    only, `none_probability` is read on that group, and `rounds` records
    every group along the way.
    """

    choice: str
    confidence: float
    probabilities: dict[str, float]
    none_probability: float | None = None
    is_none: bool | None = None
    true_label: str | None = None
    rounds: list[dict[str, Any]] | None = None

    def top(self, n: int = 3) -> list[tuple[str, float]]:
        """The `n` most likely labels, most likely first."""
        return sorted(self.probabilities.items(), key=lambda kv: -kv[1])[:n]

    @property
    def correct(self) -> bool | None:
        """None when the instruction carried no `true_label`."""
        if self.true_label is None:
            return None
        return self.choice == self.true_label

    @property
    def true_label_probability(self) -> float | None:
        """How much mass landed on the expected label (0 for a label knocked
        out earlier in a tournament: it never met the finalists)."""
        if self.true_label is None:
            return None
        return self.probabilities.get(self.true_label, 0.0)

    @property
    def eliminated_round(self) -> int | None:
        """The 1-based tournament round that knocked out `true_label`; None
        without a tournament or a label, or when the label reached the final."""
        if self.rounds is None or self.true_label is None:
            return None
        for r, entry in enumerate(self.rounds[1:], start=1):
            present = set(entry.get("byes", ()))
            present.update(c for dist in entry["groups"] for c in dist)
            if self.true_label not in present:
                return r
        return None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "type": "choice",
            "choice": self.choice,
            "confidence": self.confidence,
            "probabilities": dict(self.probabilities),
        }
        if self.none_probability is not None:
            out["none_probability"] = self.none_probability
            out["is_none"] = self.is_none
        if self.true_label is not None:
            out["true_label"] = self.true_label
            out["correct"] = self.correct
            out["true_label_probability"] = self.true_label_probability
            if self.rounds is not None:
                out["true_label_eliminated_round"] = self.eliminated_round
        if self.rounds is not None:
            out["rounds"] = self.rounds
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Answer":
        # The derived fields (`correct`, ...) are recomputed, not read.
        return cls(
            choice=d["choice"],
            confidence=d["confidence"],
            probabilities=dict(d["probabilities"]),
            none_probability=d.get("none_probability"),
            is_none=d.get("is_none"),
            true_label=d.get("true_label"),
            rounds=d.get("rounds"),
        )


@dataclass
class Usage:
    """What the model read, counted once per token actually encoded.

    Nothing is generated, so every token is input: `input_tokens` is the
    rules, context, instructions and labels (with their examples) together.
    The other fields are its breakdown, sent when `usage_detail` is asked
    for: the shared prefix (rules + context), the instructions, and the
    labels.
    """

    input_tokens: int
    prefix_tokens: int = 0
    question_tokens: int = 0
    candidate_tokens: int = 0
    unshared_equivalent_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            prefix_tokens=self.prefix_tokens + other.prefix_tokens,
            question_tokens=self.question_tokens + other.question_tokens,
            candidate_tokens=self.candidate_tokens + other.candidate_tokens,
            unshared_equivalent_tokens=self.unshared_equivalent_tokens
            + other.unshared_equivalent_tokens,
        )

    @property
    def prefix_reuse_factor(self) -> float:
        """How many times more the model would read if every label were sent
        as its own prompt instead of sharing the context."""
        return self.unshared_equivalent_tokens / max(1, self.input_tokens)

    def to_dict(self, detail: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {"input_tokens": self.input_tokens}
        if detail:
            out.update(
                prefix_tokens=self.prefix_tokens,
                question_tokens=self.question_tokens,
                candidate_tokens=self.candidate_tokens,
                unshared_equivalent_tokens=self.unshared_equivalent_tokens,
                prefix_reuse_factor=round(self.prefix_reuse_factor, 3),
            )
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Usage":
        return cls(
            input_tokens=d["input_tokens"],
            prefix_tokens=d.get("prefix_tokens", 0),
            question_tokens=d.get("question_tokens", 0),
            candidate_tokens=d.get("candidate_tokens", 0),
            unshared_equivalent_tokens=d.get("unshared_equivalent_tokens", 0),
        )


@dataclass
class Response:
    """The answers to one request.

    `warnings` lists anything the model did not read as sent: a context cut
    to fit the token budget, a label or its examples cut short. Check it
    before trusting an answer on a long input.
    """

    model: str
    answers: dict[str, Answer]
    usage: Usage
    warnings: list[str] = field(default_factory=list)
    prediction_seconds: float | None = None
    usage_detail: bool = False

    def __getitem__(self, name: str) -> Answer:
        return self.answers[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.answers)

    def __len__(self) -> int:
        return len(self.answers)

    @property
    def answer(self) -> Answer:
        """The only answer, for a request with a single instruction."""
        if len(self.answers) != 1:
            raise ValueError(
                f"this response has {len(self.answers)} answers; index it by name: "
                f"{list(self.answers)}"
            )
        return next(iter(self.answers.values()))

    def evaluation(self) -> dict[str, Any] | None:
        """Accuracy over the instructions that carried a `true_label`.

        For spotting which instructions a request got wrong, not a metric:
        a handful of instructions is far too few to measure a model with.
        """
        scored = {n: a for n, a in self.answers.items() if a.true_label is not None}
        if not scored:
            return None
        correct = [n for n, a in scored.items() if a.correct]
        return {
            "n_labelled": len(scored),
            "n_correct": len(correct),
            "accuracy": len(correct) / len(scored),
            "incorrect": sorted(set(scored) - set(correct)),
            "mean_true_label_probability": sum(a.true_label_probability for a in scored.values())
            / len(scored),
        }

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "model": self.model,
            "answers": {n: a.to_dict() for n, a in self.answers.items()},
            "usage": self.usage.to_dict(detail=self.usage_detail),
        }
        evaluation = self.evaluation()
        if evaluation is not None:
            out["evaluation"] = evaluation
        if self.warnings:
            out["warnings"] = list(self.warnings)
        if self.prediction_seconds is not None:
            out["prediction_seconds"] = self.prediction_seconds
        return out

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Response":
        usage = d["usage"]
        return cls(
            model=d["model"],
            answers={n: Answer.from_dict(a) for n, a in d["answers"].items()},
            usage=Usage.from_dict(usage),
            warnings=list(d.get("warnings", [])),
            prediction_seconds=d.get("prediction_seconds"),
            usage_detail="prefix_tokens" in usage,
        )
