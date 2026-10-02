"""The tagged layout the model was trained on.

    prefix (shared by every candidate)
        <rules>        the policy or procedure that settles the choice
        <context>      the text to choose about (several parts: blank line)
        <instruction>  the question

    branch (one per candidate, never seen by the others)
        <examples>     this candidate's own few-shot examples
        <choice>candidate</choice>   the closing tag is the readout

None of this is configurable: a model reads the layout it was trained on.
"""

from __future__ import annotations

import re
from typing import Sequence

from .config import TAG_TOKENS

_TAGS = re.compile("|".join(re.escape(t) for t in TAG_TOKENS))

# Between the state and the instruction.
STATE_SEP = "\n\n"


def strip_tags(text: str) -> str:
    """Content must never contain a section tag: the tokenizer would read it as
    the added token and the text would open or close a section."""
    return _TAGS.sub("", text)


def render_state(rules: str | None = None, context: str | Sequence[str] | None = None) -> str:
    if context is not None and not isinstance(context, str):
        context = "\n\n".join(strip_tags(c).strip() for c in context if c and c.strip())
    parts = []
    if rules and rules.strip():
        parts.append(f"<rules>\n{strip_tags(rules).strip()}\n</rules>")
    if context and context.strip():
        parts.append(f"<context>\n{strip_tags(context).strip()}\n</context>")
    return "\n\n".join(parts)


def render_question(question: str) -> str:
    return f"<instruction>\n{strip_tags(question).strip()}\n</instruction>\n"


def render_choice(candidate: str) -> str:
    """A candidate without its closing `</choice>`, which is appended as an id
    so truncation can never cut it off."""
    return f"<choice>{strip_tags(candidate)}"


def render_choice_context(inputs: Sequence[str]) -> str:
    """One candidate's own few-shot examples, read in its branch only."""
    ex = "\n".join(f"<example>\n{strip_tags(i).strip()}\n</example>" for i in inputs)
    return f"<examples>\n{ex}\n</examples>\n"
