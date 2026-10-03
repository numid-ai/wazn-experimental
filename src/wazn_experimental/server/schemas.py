"""The wire format, as pydantic models: they check the shape of a request
body. The semantic checks (at least 2 labels, unique names...) are
`Request`'s, shared with the client."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StateModel(_Strict):
    rules: str | None = Field(None, description="Policy or procedure that settles the answer.")
    context: str | list[str] = Field(
        ..., description="The text to answer about; a list of parts is joined by a blank line."
    )


class LabelModel(_Strict):
    definition: str = Field("", description="When the label applies; the model reads this.")
    examples: list[str] = Field(
        default_factory=list, description="Few-shot inputs that get this label; only it reads them."
    )


class TournamentModel(_Strict):
    group_size: int = Field(..., ge=2, description="Labels compared at once in each group.")
    top_k: int = Field(1, ge=1, description="Labels advancing from each group.")
    seed: int | None = Field(None, description="Shuffle the labels before the first round.")


class QuestionModel(_Strict):
    type: Literal["choice"] = "choice"
    instructions: str = Field(..., description="The question to answer about the context.")
    criteria: dict[str, str | LabelModel] | list[str] = Field(
        ...,
        description="Label name -> definition, or -> {definition, examples}; or bare names.",
    )
    true_label: str | None = Field(None, description="Optional expected answer, for spot checks.")
    tournament: TournamentModel | None = Field(None, description="Answer this question by tournament.")


class RequestModel(_Strict):
    state: str | StateModel = Field(..., description="The context, or {rules, context}.")
    questions: dict[str, QuestionModel] = Field(..., description="Name -> question.")


class OptionsModel(_Strict):
    usage_detail: bool = Field(False, description="Break token usage down.")


class PredictBody(_Strict):
    request: RequestModel
    options: OptionsModel = Field(default_factory=OptionsModel)
