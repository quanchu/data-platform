"""Source provenance and strict model output contracts."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Span(Record):
    start: int
    end: int


class Sentence(Span):
    index: int
    section_index: int
    text: str


class Section(Span):
    index: int
    subtitle: str
    text: str
    sentences: list[Sentence] = Field(default_factory=list)
    excluded_spans: list[Span] = Field(default_factory=list)


class Document(Record):
    title: str
    sections: list[Section] = Field(default_factory=list)


class Rewrite(Record):
    text: Text


class GeneratedClaims(Record):
    claims: list[Text]


class Claim(Span):
    text: str
    section_index: int
    sentence_index: int


class StageRecord(Record):
    stage: str
    output: dict[str, Any]
    model: str | None = None
    confidence: Any = None
    usage: dict[str, Any] = Field(default_factory=dict)


class SentenceResult(Sentence):
    selected_text: str | None = None
    clarified_text: str | None = None
    claims: list[Claim] = Field(default_factory=list)
    stages: list[StageRecord] = Field(default_factory=list)
    status: Literal["extracted", "no_verifiable_claims", "unresolved_ambiguity", "validation_failed"] = "no_verifiable_claims"


class ExtractionResult(Record):
    document: Document
    sentences: list[SentenceResult] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    requested_models: dict[str, str] = Field(default_factory=dict)
