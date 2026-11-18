"""Source provenance and strict model output contracts."""

from typing import Any, Literal

from pydantic import Field

from ..source import Document, Record, Sentence, Span, Text
from ..source import Section as Section


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
    status: Literal[
        "extracted", "no_verifiable_claims", "unresolved_ambiguity", "validation_failed"
    ] = "no_verifiable_claims"


class ExtractionResult(Record):
    document: Document
    sentences: list[SentenceResult] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    requested_models: dict[str, str] = Field(default_factory=dict)
