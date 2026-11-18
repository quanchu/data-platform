"""Async adapter protocols and default Pydantic AI implementations."""
import dataclasses
import json
from typing import Protocol

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model

from ..source import Section, Sentence
from .models import (
    BackendMetadata, ExtractionResponse, PropositionOutput, ResolutionOutput,
    ResolutionResponse, ResolvedSentence,
)
from .prompts import EXTRACT, RESOLVE

DEFAULT_MODEL = 'openai:gpt-6-luna'


class ReferenceResolver(Protocol):
    async def resolve(self, *, title: str, section: Section) -> ResolutionResponse: ...


class PropositionExtractor(Protocol):
    async def extract(self, *, sentence: Sentence, resolved: ResolvedSentence) -> ExtractionResponse: ...


def validate_resolution(output: ResolutionOutput, section: Section) -> None:
    originals = {s.index: s for s in section.sentences}
    ids = [s.sentence_index for s in output.sentences]
    if len(ids) != len(set(ids)) or set(ids) != set(originals):
        raise ValueError('Resolution must cover each target sentence ID exactly once')
    for resolved in output.sentences:
        original = originals[resolved.sentence_index].text
        if resolved.status != 'resolved':
            if resolved.text != original or resolved.replacements:
                raise ValueError('Unchanged/unresolved sentences must preserve source text without replacements')
        elif resolved.text == original or not resolved.replacements:
            raise ValueError('Resolved sentences require a rewrite and replacement evidence')
        for replacement in resolved.replacements:
            if replacement.original not in original or replacement.replacement not in resolved.text:
                raise ValueError('Replacement mentions must occur in original and resolved text')
            for evidence in replacement.evidence:
                if evidence.source == 'sentence' and evidence.sentence_index not in originals:
                    raise ValueError('Evidence must reference a sentence in this section')


def metadata(response) -> BackendMetadata:
    return BackendMetadata(backend='pydantic-ai', model=response.response.model_name,
                           usage=dataclasses.asdict(response.usage))


class LLMReferenceResolver:
    def __init__(self, model: str | Model = DEFAULT_MODEL):
        self.model = model

    async def resolve(self, *, title: str, section: Section) -> ResolutionResponse:
        agent = Agent(self.model, output_type=ResolutionOutput, instructions=RESOLVE, retries={'output': 2})

        @agent.output_validator
        def validate(ctx: RunContext, output: ResolutionOutput) -> ResolutionOutput:
            try:
                validate_resolution(output, section)
            except ValueError as exc:
                raise ModelRetry(str(exc)) from exc
            return output

        response = await agent.run(json.dumps({'document_title': title, 'section': section.model_dump()}, ensure_ascii=False))
        return ResolutionResponse(**response.output.model_dump(), metadata=metadata(response))


class LLMPropositionExtractor:
    def __init__(self, model: str | Model = DEFAULT_MODEL):
        self.model = model

    async def extract(self, *, sentence: Sentence, resolved: ResolvedSentence) -> ExtractionResponse:
        agent = Agent(self.model, output_type=PropositionOutput, instructions=EXTRACT, retries={'output': 2})
        response = await agent.run(json.dumps({'sentence': sentence.model_dump(), 'resolved': resolved.model_dump()}, ensure_ascii=False))
        return ExtractionResponse(**response.output.model_dump(), metadata=metadata(response))
