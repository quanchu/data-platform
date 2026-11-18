"""Typed contracts shared by hosted and caller-supplied OpenIE backends."""
from typing import Annotated, Any, Literal

from pydantic import Field, StringConstraints, model_validator

from ..source import Document, Record, Sentence, Span

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Evidence(Record):
    source: Literal['sentence', 'title', 'subtitle']
    sentence_index: int | None = None

    @model_validator(mode='after')
    def check_source(self):
        if (self.source == 'sentence') != (self.sentence_index is not None):
            raise ValueError('Only sentence evidence requires a sentence_index')
        return self


class Replacement(Record):
    original: Text
    replacement: Text
    evidence: list[Evidence] = Field(min_length=1)


class ResolvedSentence(Record):
    sentence_index: int
    text: Annotated[str, StringConstraints(min_length=1)]
    status: Literal['resolved', 'unchanged', 'unresolved']
    replacements: list[Replacement] = Field(default_factory=list)


class BackendMetadata(Record):
    backend: Text
    model: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)


class ResolutionOutput(Record):
    sentences: list[ResolvedSentence]


class ResolutionResponse(ResolutionOutput):
    metadata: BackendMetadata


class Qualifier(Record):
    kind: Literal['time', 'location', 'condition', 'negation', 'modality', 'attribution', 'other']
    text: Text


class Proposition(Record):
    subject: Text
    relation: Text
    object: Text | None = None
    qualifiers: list[Qualifier] = Field(default_factory=list)
    text: Text


class PropositionOutput(Record):
    propositions: list[Proposition]


class ExtractionResponse(PropositionOutput):
    metadata: BackendMetadata


class Claim(Proposition, Span):
    section_index: int
    sentence_index: int


class StageRecord(Record):
    stage: Literal['resolution', 'extraction']
    output: dict[str, Any]
    metadata: BackendMetadata


class SentenceResult(Sentence):
    resolution: ResolvedSentence
    status: Literal['extracted', 'no_claims', 'unresolved_ambiguity']
    claims: list[Claim] = Field(default_factory=list)
    stages: list[StageRecord] = Field(default_factory=list)


class ExtractionResult(Record):
    document: Document
    sentences: list[SentenceResult] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    resolution_stages: dict[int, StageRecord] = Field(default_factory=dict)
