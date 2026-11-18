"""One explicit semantic question per Jev field; no generative fields."""

from typing import Literal

from pydantic import Field

from .models import Record
from .prompts import VERIFIABILITY


class Selection(Record):
    verdict: Literal["fully_verifiable", "mixed", "none"] = Field(description="Does the target sentence contain only verifiable factual content, both verifiable and unverifiable content, or no verifiable content? Judge verifiability, not truth; assume ambiguity can be resolved for this decision. " + VERIFIABILITY)


class FilterCheck(Record):
    supported: bool = Field(description="Is every assertion in candidate supported by the original target sentence in its supplied context?")
    coverage: bool = Field(description="Does candidate retain all verifiable factual content of the original target sentence? " + VERIFIABILITY)
    factual_only: bool = Field(description="Has candidate removed all unverifiable content, including personal opinions, feelings and subjective evaluations, from the target sentence? " + VERIFIABILITY)


class Ambiguity(Record):
    referential: Literal["absent", "resolvable", "unresolvable"] = Field(description="Does selected_text contain context-dependent references (pronouns, possessives, demonstratives, omitted subjects or relative time/place references), and if so can all referents be uniquely identified using only the supplied title, subtitle, section and question? Treat a reference as resolvable even if its referent is obvious from context but not explicit in selected_text. Mark unresolvable if any required referent or temporal/spatial anchor is missing or admits multiple interpretations; never guess.")
    structural: Literal["absent", "resolvable", "unresolvable"] = Field(description="Does selected_text have structural ambiguity, and if so can all such ambiguity be resolved using only the supplied context? This includes ambiguity between a speaker assertion and author interpretation.")
    expand_names: bool = Field(description="Does selected_text contain a partial name or acronym whose full form is explicitly supplied in context and should be expanded?")


class ClarificationCheck(Record):
    supported: bool = Field(description="Is candidate supported by selected_text and supplied context without invented details?")
    meaning_preserved: bool = Field(description="Does candidate preserve all factual meaning of selected_text including attribution, uncertainty, quantities and tense?")
    ambiguity_resolved: bool = Field(description="Does candidate replace every context-dependent reference with an explicit, uniquely supported referent and resolve structural ambiguity, so the text can be understood independently? Reject remaining vague references, guessed referents and relative dates without a supplied anchor. Use only supplied context.")


class ClaimCheck(Record):
    supported: bool = Field(description="Is candidate entailed by the original target sentence in its supplied context, rather than merely asserted elsewhere in the section?")
    verifiable: bool = Field(description="Does candidate express only verifiable factual content, excluding personal opinions, feelings and subjective evaluations even when attributed? " + VERIFIABILITY)
    contextual: bool = Field(description="Can candidate be understood independently with the qualifiers, entities and attribution necessary to preserve its original meaning?")


class CoverageCheck(Record):
    complete: bool = Field(description="Does the candidate claim set cover all verifiable factual content of clarified_text? An empty set is complete only if clarified_text contains no verifiable content. Subjective content does not need coverage. " + VERIFIABILITY)
