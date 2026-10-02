"""Tokenizer setup and the token rows the model reads.

The prefix (rules, context, instruction) is tokenized once and shared; each
candidate is tokenized as its own suffix, so a candidate never appears in
another candidate's token stream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from tokenizers import AddedToken
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .config import CHOICE_REP_TOKEN, TAG_TOKENS, PromptConfig
from .formatting import STATE_SEP, render_choice, render_choice_context, render_question


def build_tokenizer(
    name_or_path: str, revision: str | None = None, trust_remote_code: bool = False
) -> PreTrainedTokenizerBase:
    """The backbone's tokenizer with the section tags added, one token each."""
    tok = AutoTokenizer.from_pretrained(
        name_or_path, revision=revision, trust_remote_code=trust_remote_code
    )
    clash = [t for t in TAG_TOKENS if t in tok.get_vocab()]
    if clash:
        raise ValueError(f"the backbone tokenizer already defines {clash}")
    tok.add_tokens([AddedToken(t, special=True) for t in TAG_TOKENS], special_tokens=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def tag_token_ids(tokenizer: PreTrainedTokenizerBase) -> list[int]:
    """The tags' ids, in TAG_TOKENS order: consecutive, as added."""
    ids = tokenizer.convert_tokens_to_ids(list(TAG_TOKENS))
    if ids != list(range(ids[0], ids[0] + len(ids))):
        raise ValueError(f"the section tags must have consecutive ids, got {ids}")
    return ids


@dataclass
class Segments:
    """A request as token rows: the shared state, one row per question, and
    one row per candidate (with the question it belongs to and its slot)."""

    state: list[int]
    questions: list[list[int]]
    candidates: list[list[int]] = field(default_factory=list)
    question_idx: list[int] = field(default_factory=list)
    slot: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class PromptEncoder:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, prompt: PromptConfig) -> None:
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.choice_rep_id = tokenizer.convert_tokens_to_ids(CHOICE_REP_TOKEN)
        self.pad_id = tokenizer.pad_token_id

    def _ids(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def encode(
        self,
        state: str,
        questions: Sequence[str],
        candidates: Sequence[Sequence[str]],
        shots: Sequence[Sequence[Sequence[str]]] | None = None,
        names: Sequence[str] | None = None,
    ) -> Segments:
        """`candidates[j][k]` is question j's k-th candidate text and
        `shots[j][k]` that candidate's own example inputs, which only it reads.

        The state is cut from the LEFT so that `[state, question]` fits the
        prefix budget for the longest question: every question then shares
        the same state, which is what lets it be encoded once. Note that the
        rules come first, so a long context pushes them out first.
        """
        names = names or [f"#{j}" for j in range(len(questions))]
        question_ids = [self._ids(render_question(q)) for q in questions]
        longest = max(len(q) for q in question_ids)
        budget = self.prompt.max_prefix_tokens - longest
        if budget <= 0:
            raise ValueError(
                f"the longest instruction is {longest} tokens, which leaves no room "
                f"for the context inside max_prefix_tokens={self.prompt.max_prefix_tokens}"
            )
        seg = Segments(state=[], questions=question_ids)
        if state:
            ids = self._ids(state + STATE_SEP)
            if len(ids) > budget:
                seg.warnings.append(
                    f"rules and context are {len(ids)} tokens; only the last {budget} were "
                    f"read (max_prefix_tokens={self.prompt.max_prefix_tokens})"
                )
            seg.state = ids[-budget:]

        for j, texts in enumerate(candidates):
            for k, text in enumerate(texts):
                row, cut_label, cut_shots = self.candidate(text, shots[j][k] if shots else ())
                if cut_label:
                    seg.warnings.append(
                        f"instruction {names[j]!r}: label {k} was cut to "
                        f"max_candidate_tokens={self.prompt.max_candidate_tokens}"
                    )
                if cut_shots:
                    seg.warnings.append(
                        f"instruction {names[j]!r}: label {k}'s examples were cut from the "
                        f"left to fit max_candidate_tokens={self.prompt.max_candidate_tokens}"
                    )
                seg.candidates.append(row)
                seg.question_idx.append(j)
                seg.slot.append(k)
        return seg

    def candidate(self, text: str, shot_inputs: Sequence[str] = ()) -> tuple[list[int], bool, bool]:
        """`[examples, <choice>text, </choice>]` within the candidate budget.

        The label is kept whole up to the budget and its examples are cut from
        the left to fit, so what is lost first is the oldest example.
        -> (ids, label was cut, examples were cut)
        """
        room = self.prompt.max_candidate_tokens - 1
        ids = self._ids(render_choice(text))
        cut_label = len(ids) > room
        ids = ids[:room]
        cut_shots = False
        if shot_inputs:
            ctx = self._ids(render_choice_context(shot_inputs))
            left = room - len(ids)
            cut_shots = len(ctx) > max(left, 0)
            ids = (ctx[-left:] if left > 0 else []) + ids
        ids.append(self.choice_rep_id)
        return ids, cut_label, cut_shots
