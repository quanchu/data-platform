"""Independent two-stage, section-scoped OpenIE orchestration."""
import asyncio

from pydantic_ai.models import Model

from ..source import parse_document
from .backends import (
    DEFAULT_MODEL, LLMPropositionExtractor, LLMReferenceResolver,
    PropositionExtractor, ReferenceResolver, validate_resolution,
)
from .models import Claim, ExtractionResponse, ExtractionResult, ResolutionResponse, SentenceResult, StageRecord


class OpenIEError(RuntimeError):
    def __init__(self, section_index: int, sentence_index: int | None, stage: str):
        self.section_index = section_index
        self.sentence_index = sentence_index
        self.stage = stage
        super().__init__(f'OpenIE failed at section {section_index}, sentence {sentence_index}, stage {stage}')


async def extract_claims(
    markdown_doc: str, *, resolver: ReferenceResolver | None = None,
    extractor: PropositionExtractor | None = None,
    resolution_model: str | Model = DEFAULT_MODEL,
    extraction_model: str | Model = DEFAULT_MODEL,
    max_concurrency: int = 4,
) -> ExtractionResult:
    if isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int) or max_concurrency < 1:
        raise ValueError('max_concurrency must be a positive integer')
    document = parse_document(markdown_doc)
    result = ExtractionResult(document=document)
    resolver = resolver if resolver is not None else LLMReferenceResolver(resolution_model)
    extractor = extractor if extractor is not None else LLMPropositionExtractor(extraction_model)
    sections = iter(s for s in document.sections if s.sentences)
    resolved_by_id = {}

    async def run_workers(worker, count):
        tasks = [asyncio.create_task(worker()) for _ in range(min(max_concurrency, count))]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def resolve_worker():
        for section in sections:
            try:
                response = await resolver.resolve(title=document.title, section=section.model_copy(deep=True))
                response = ResolutionResponse.model_validate(response.model_dump())
                validate_resolution(response, section)
                result.resolution_stages[section.index] = StageRecord(
                    stage='resolution', output={'sentences': [s.model_dump() for s in response.sentences]},
                    metadata=response.metadata,
                )
                resolved_by_id.update((s.sentence_index, s) for s in response.sentences)
            except Exception as exc:
                raise OpenIEError(section.index, None, 'resolution') from exc

    await run_workers(resolve_worker, len(document.sections))
    pending = iter(s for section in document.sections for s in section.sentences)
    outputs = {}

    async def extract_worker():
        for sentence in pending:
            resolved = resolved_by_id[sentence.index]
            output = SentenceResult(**sentence.model_dump(), resolution=resolved,
                                    status='unresolved_ambiguity' if resolved.status == 'unresolved' else 'no_claims')
            outputs[sentence.index] = output
            if resolved.status == 'unresolved':
                continue
            try:
                response = await extractor.extract(sentence=sentence.model_copy(deep=True), resolved=resolved.model_copy(deep=True))
                response = ExtractionResponse.model_validate(response.model_dump())
                output.stages.append(StageRecord(stage='extraction', output={'propositions': [p.model_dump() for p in response.propositions]}, metadata=response.metadata))
                output.claims = [Claim(**p.model_dump(), section_index=sentence.section_index,
                                       sentence_index=sentence.index, start=sentence.start, end=sentence.end)
                                 for p in response.propositions]
                if output.claims:
                    output.status = 'extracted'
            except Exception as exc:
                raise OpenIEError(sentence.section_index, sentence.index, 'extraction') from exc

    await run_workers(extract_worker, len(resolved_by_id))
    result.resolution_stages = dict(sorted(result.resolution_stages.items()))
    result.sentences = [outputs[i] for i in sorted(outputs)]
    result.claims = [claim for sentence in result.sentences for claim in sentence.claims]
    return result
