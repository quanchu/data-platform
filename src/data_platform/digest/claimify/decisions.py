"""One explicit semantic question per Jev field; no generative fields."""

from typing import Literal

from pydantic import Field

from .models import Record
from . import prompts


class Selection(Record):
    verdict: Literal["fully_verifiable", "mixed", "none"] = Field(
        description=prompts.SELECTION_VERDICT
    )


class FilterCheck(Record):
    supported: bool = Field(description=prompts.FILTER_CHECK_SUPPORTED)
    coverage: bool = Field(description=prompts.FILTER_CHECK_COVERAGE)
    factual_only: bool = Field(description=prompts.FILTER_CHECK_FACTUAL_ONLY)


class Ambiguity(Record):
    referential: Literal["absent", "resolvable", "unresolvable"] = Field(
        description=prompts.AMBIGUITY_REFERENTIAL
    )
    structural: Literal["absent", "resolvable", "unresolvable"] = Field(
        description=prompts.AMBIGUITY_STRUCTURAL
    )
    expand_names: bool = Field(description=prompts.AMBIGUITY_EXPAND_NAMES)


class ClarificationCheck(Record):
    supported: bool = Field(description=prompts.CLARIFICATION_CHECK_SUPPORTED)
    meaning_preserved: bool = Field(
        description=prompts.CLARIFICATION_CHECK_MEANING_PRESERVED
    )
    ambiguity_resolved: bool = Field(
        description=prompts.CLARIFICATION_CHECK_AMBIGUITY_RESOLVED
    )


class ClaimCheck(Record):
    supported: bool = Field(description=prompts.CLAIM_CHECK_SUPPORTED)
    verifiable: bool = Field(description=prompts.CLAIM_CHECK_VERIFIABLE)
    contextual: bool = Field(description=prompts.CLAIM_CHECK_CONTEXTUAL)


class CoverageCheck(Record):
    complete: bool = Field(description=prompts.COVERAGE_CHECK_COMPLETE)
