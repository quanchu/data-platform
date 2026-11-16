"""One explicit semantic question per Jev field; no generative fields."""

from typing import Literal

from pydantic import Field

from .models import Record


class Selection(Record):
    verdict: Literal["fully_verifiable", "mixed", "none"] = Field(description="Does the target sentence contain only verifiable factual content, both verifiable and unverifiable content, or no verifiable content? Judge verifiability, not truth; assume ambiguity can be resolved for this decision.")


class FilterCheck(Record):
    supported: bool = Field(description="Is every assertion in candidate supported by the original target sentence in its supplied context?")
    coverage: bool = Field(description="Does candidate retain all verifiable factual content of the original target sentence?")
    factual_only: bool = Field(description="Has candidate removed all explicitly unverifiable content from the target sentence?")


class Ambiguity(Record):
    referential: Literal["absent", "resolvable", "unresolvable"] = Field(description="Does selected_text have referential ambiguity, and if so can all such ambiguity be resolved using only the supplied title, subtitle, section and question?")
    structural: Literal["absent", "resolvable", "unresolvable"] = Field(description="Does selected_text have structural ambiguity, and if so can all such ambiguity be resolved using only the supplied context? This includes ambiguity between a speaker assertion and author interpretation.")
    expand_names: bool = Field(description="Does selected_text contain a partial name or acronym whose full form is explicitly supplied in context and should be expanded?")


class ClarificationCheck(Record):
    supported: bool = Field(description="Is candidate supported by selected_text and supplied context without invented details?")
    meaning_preserved: bool = Field(description="Does candidate preserve all factual meaning of selected_text including attribution, uncertainty, quantities and tense?")
    ambiguity_resolved: bool = Field(description="Does candidate resolve all referential and structural ambiguity using only supplied context?")


class ClaimCheck(Record):
    supported: bool = Field(description="Is candidate entailed by the original target sentence in its supplied context, rather than merely asserted elsewhere in the section?")
    verifiable: bool = Field(description="Does candidate express verifiable factual content without explicit unverifiable assertions?")
    contextual: bool = Field(description="Can candidate be understood independently with the qualifiers, entities and attribution necessary to preserve its original meaning?")


class CoverageCheck(Record):
    complete: bool = Field(description="Does the candidate claim set cover all verifiable factual content of clarified_text? An empty set is complete only if clarified_text contains no verifiable content.")
