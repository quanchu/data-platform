"""Independent two-stage, section-scoped OpenIE orchestration."""

from pydantic_ai.models import Model

from .._events import emit
from .prompts import RESOLVE, EXTRACT
from .._async import ordered_map, positive_int
from ..source import parse_document
from .backends import (
    DEFAULT_MODEL,
    LLMPropositionExtractor,
    LLMReferenceResolver,
    PropositionExtractor,
    ReferenceResolver,
    validate_resolution,
)
from .models import (
    Claim,
    ExtractionResponse,
    ExtractionResult,
    ResolutionResponse,
    SentenceResult,
    StageRecord,
)


class OpenIEError(RuntimeError):
    def __init__(self, section_index: int, sentence_index: int | None, stage: str):
        self.section_index = section_index
        self.sentence_index = sentence_index
        self.stage = stage
        super().__init__(
            f"OpenIE failed at section {section_index}, sentence {sentence_index}, stage {stage}"
        )


async def extract_claims(
    markdown_doc: str,
    *,
    resolver: ReferenceResolver | None = None,
    extractor: PropositionExtractor | None = None,
    resolution_model: str | Model = DEFAULT_MODEL,
    extraction_model: str | Model = DEFAULT_MODEL,
    max_concurrency: int = 20,
) -> ExtractionResult:
    positive_int(max_concurrency, "max_concurrency")
    document = parse_document(markdown_doc)
    result = ExtractionResult(document=document)
    resolver = (
        resolver if resolver is not None else LLMReferenceResolver(resolution_model)
    )
    extractor = (
        extractor
        if extractor is not None
        else LLMPropositionExtractor(extraction_model)
    )
    sections = iter(s for s in document.sections if s.sentences)

    async def resolve_section(section):
        spans = [(section.start, section.end)]
        emit("resolution", status="running", spans=spans, prompt=RESOLVE)
        try:
            response = await resolver.resolve(
                title=document.title, section=section.model_copy(deep=True)
            )
            response = ResolutionResponse.model_validate(response.model_dump())
            validate_resolution(response, section)
            record = StageRecord(
                stage="resolution",
                output={"sentences": [s.model_dump() for s in response.sentences]},
                metadata=response.metadata,
            )
            emit("resolution", spans=spans, output=response.model_dump(mode="json"))
            return section.index, record, response.sentences
        except Exception as exc:
            emit("resolution", status="failed", spans=spans, error=str(exc))
            raise OpenIEError(section.index, None, "resolution") from exc

    resolutions = await ordered_map(
        resolve_section, sections, max_concurrency=max_concurrency
    )
    result.resolution_stages = {index: record for index, record, _ in resolutions}
    resolved_by_id = {
        sentence.sentence_index: sentence
        for _, _, sentences in resolutions
        for sentence in sentences
    }
    pending = iter(s for section in document.sections for s in section.sentences)

    async def extract_sentence(sentence):
        resolved = resolved_by_id[sentence.index]
        output = SentenceResult(
            **sentence.model_dump(),
            resolution=resolved,
            status="unresolved_ambiguity"
            if resolved.status == "unresolved"
            else "no_claims",
        )
        if resolved.status == "unresolved":
            emit(
                "extraction",
                status="skipped",
                spans=[(sentence.start, sentence.end)],
                reason="Unresolved ambiguity",
            )
            return output
        spans = [(sentence.start, sentence.end)]
        emit("extraction", status="running", spans=spans, prompt=EXTRACT)
        try:
            response = await extractor.extract(
                sentence=sentence.model_copy(deep=True),
                resolved=resolved.model_copy(deep=True),
            )
            response = ExtractionResponse.model_validate(response.model_dump())
            output.stages.append(
                StageRecord(
                    stage="extraction",
                    output={
                        "propositions": [p.model_dump() for p in response.propositions]
                    },
                    metadata=response.metadata,
                )
            )
            output.claims = [
                Claim(
                    **p.model_dump(),
                    section_index=sentence.section_index,
                    sentence_index=sentence.index,
                    start=sentence.start,
                    end=sentence.end,
                )
                for p in response.propositions
            ]
            if output.claims:
                output.status = "extracted"
            emit("extraction", spans=spans, output=output.model_dump(mode="json"))
            return output
        except Exception as exc:
            emit("extraction", status="failed", spans=spans, error=str(exc))
            raise OpenIEError(
                sentence.section_index, sentence.index, "extraction"
            ) from exc

    outputs = await ordered_map(
        extract_sentence, pending, max_concurrency=max_concurrency
    )
    result.resolution_stages = dict(sorted(result.resolution_stages.items()))
    result.sentences = outputs
    result.claims = [
        claim for sentence in result.sentences for claim in sentence.claims
    ]
    return result
