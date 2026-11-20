"""Persisted claims and typed retrieval/comparison results."""

from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from . import prompts
from ..source import Record, Text

Status = Literal["unverified", "verified"]
Verdict = Literal["conflict", "equivalent", "compatible", "unrelated", "uncertain"]


class ClaimRecord(Record):
    id: str = Field(pattern=r"^[0-9a-f]{64}$")
    text: Text
    source: Text
    document_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_title: str
    original_sentence: Text
    section_title: str
    section_index: int = Field(ge=0)
    sentence_index: int = Field(ge=0)
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    created_at: AwareDatetime

    @model_validator(mode="after")
    def ordered_offsets(self):
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self


class ComparisonDecision(Record):
    verdict: Verdict = Field(
        description=prompts.COMPARISON_VERDICT,
    )


class RetrievalMatch(Record):
    claim: ClaimRecord
    similarity: float


class PairComparison(Record):
    claim: ClaimRecord
    verified_claim: ClaimRecord
    similarity: float
    decision: ComparisonDecision


class CheckReport(Record):
    claims: list[ClaimRecord]
    comparisons: list[PairComparison]
    top_k: int
    verified_count: int
    scope: str = "No conflict found means no conflict found among retrieved candidates; this is not exhaustive and does not approve claims."

    @property
    def conflicts(self) -> list[PairComparison]:
        return [p for p in self.comparisons if p.decision.verdict == "conflict"]

    @property
    def uncertain(self) -> list[PairComparison]:
        return [p for p in self.comparisons if p.decision.verdict == "uncertain"]

    @property
    def review_required(self) -> list[PairComparison]:
        return [
            p
            for p in self.comparisons
            if p.decision.verdict in {"conflict", "uncertain"}
        ]


class SyncResult(Record):
    verified_count: int
    embedded_count: int
    removed_count: int
