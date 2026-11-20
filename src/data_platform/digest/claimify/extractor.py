"""Section-scoped Claimify orchestration with explicit model routing."""

import dataclasses
import json

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import Model

from .._events import emit
from ..source import Section, Sentence
from .._async import ordered_map, positive_int
from . import decisions as d
from . import prompts
from .document import parse_document
from .models import (
    Claim,
    ExtractionResult,
    GeneratedClaims,
    Rewrite,
    SentenceResult,
    StageRecord,
)


class ClaimifyError(RuntimeError):
    def __init__(self, section_index: int, sentence_index: int, stage: str):
        self.section_index = section_index
        self.sentence_index = sentence_index
        self.stage = stage
        super().__init__(
            f"Claimify failed at section {section_index}, sentence {sentence_index}, stage {stage}"
        )


@dataclasses.dataclass
class _SentenceContext:
    section: Section
    sentence: Sentence
    payload: dict
    output: SentenceResult
    decision_model: str | Model
    generation_model: str | Model


async def _call[Output: BaseModel](
    context: _SentenceContext,
    stage_name: str,
    schema: type[Output],
    payload: dict,
    *,
    instructions: str | None = None,
) -> Output:
    spans = [(context.sentence.start, context.sentence.end)]
    emit(
        stage_name,
        status="running",
        spans=spans,
        sentence_index=context.sentence.index,
        prompt=instructions or prompts.DECISION_INSTRUCTIONS,
        request={**context.payload, **payload},
        schema=schema.model_json_schema(),
    )
    try:
        agent = Agent(
            context.generation_model if instructions else context.decision_model,
            output_type=schema,
            retries={"output": 2},
            instructions=instructions or prompts.DECISION_INSTRUCTIONS,
        )
        response = await agent.run(
            json.dumps({**context.payload, **payload}, ensure_ascii=False)
        )
        context.output.stages.append(
            StageRecord(
                stage=stage_name,
                output=response.output.model_dump(),
                model=response.response.model_name,
                confidence=(response.response.provider_details or {}).get("confidence"),
                usage=dataclasses.asdict(response.usage),
            )
        )
        emit(
            stage_name,
            spans=spans,
            sentence_index=context.sentence.index,
            output=response.output.model_dump(),
            model=response.response.model_name,
        )
        return response.output
    except Exception as exc:
        emit(stage_name, status="failed", spans=spans, error=str(exc))
        raise ClaimifyError(
            context.section.index, context.sentence.index, stage_name
        ) from exc


async def _rewrite(
    context: _SentenceContext, stage: str, instructions: str, check_schema
) -> str | None:
    repair = {}
    for _ in range(2):
        candidate = await _call(
            context, stage, Rewrite, repair, instructions=instructions
        )
        check = await _call(
            context,
            stage + "_check",
            check_schema,
            {"candidate": candidate.text},
        )
        failures = [key for key, value in check.model_dump().items() if not value]
        if not failures:
            return candidate.text
        repair = {
            "previous_candidate": candidate.text,
            "failed_criteria": failures,
        }
    context.output.status = "validation_failed"
    return None


async def _generate_claims(context: _SentenceContext) -> SentenceResult:
    output, section, sentence = context.output, context.section, context.sentence
    repair = {}
    for _ in range(2):
        generated = await _call(
            context, "generate", GeneratedClaims, repair, instructions=prompts.GENERATE
        )
        failures = []
        for index, claim in enumerate(generated.claims):
            check = await _call(
                context,
                "claim_check",
                d.ClaimCheck,
                {"candidate": claim, "claim_index": index},
            )
            failures.extend(
                f"claim[{index}].{key}"
                for key, value in check.model_dump().items()
                if not value
            )
        coverage = await _call(
            context, "coverage", d.CoverageCheck, {"candidate_claims": generated.claims}
        )
        if not coverage.complete:
            failures.append("set_coverage")
        if not failures:
            output.claims = [
                Claim(
                    text=c,
                    section_index=section.index,
                    sentence_index=sentence.index,
                    start=sentence.start,
                    end=sentence.end,
                )
                for c in generated.claims
            ]
            output.status = "extracted" if output.claims else "no_verifiable_claims"
            return output
        repair = {
            "previous_candidates": generated.claims,
            "failed_criteria": failures,
        }
    output.status = "validation_failed"
    return output


async def _process_sentence(context: _SentenceContext) -> SentenceResult:
    output, sentence = context.output, context.sentence
    selection = await _call(context, "selection", d.Selection, {})
    if selection.verdict == "none":
        return output
    selected = sentence.text
    if selection.verdict == "mixed":
        selected = await _rewrite(context, "filter", prompts.FILTER, d.FilterCheck)
        if selected is None:
            return output
    output.selected_text = selected
    context.payload["selected_text"] = selected
    ambiguity = await _call(context, "ambiguity", d.Ambiguity, {})
    if "unresolvable" in (ambiguity.referential, ambiguity.structural):
        output.status = "unresolved_ambiguity"
        return output
    clarified = selected
    if ambiguity.expand_names or "resolvable" in (
        ambiguity.referential,
        ambiguity.structural,
    ):
        clarified = await _rewrite(
            context, "clarify", prompts.CLARIFY, d.ClarificationCheck
        )
        if clarified is None:
            return output
    output.clarified_text = clarified
    context.payload["clarified_text"] = clarified
    return await _generate_claims(context)


async def extract_claims(
    markdown_doc: str,
    *,
    question: str | None = None,
    decision_model: str | Model = "typesafe:jev-latest",
    generation_model: str | Model = "openai:gpt-6-luna",
    max_concurrency: int = 20,
) -> ExtractionResult:
    positive_int(max_concurrency, "max_concurrency")
    document = parse_document(markdown_doc)
    result = ExtractionResult(
        document=document,
        requested_models={
            "decision": decision_model
            if isinstance(decision_model, str)
            else decision_model.model_name,
            "generation": generation_model
            if isinstance(generation_model, str)
            else generation_model.model_name,
        },
    )

    async def process_item(item):
        section, sentence = item
        context = _SentenceContext(
            section=section,
            sentence=sentence,
            payload={
                "document_title": document.title,
                "section_subtitle": section.subtitle,
                "section_content": section.text,
                "target_sentence": sentence.text,
                "sentence_index": sentence.index,
                "question": question,
            },
            output=SentenceResult(**sentence.model_dump()),
            decision_model=decision_model,
            generation_model=generation_model,
        )
        return await _process_sentence(context)

    result.sentences = await ordered_map(
        process_item,
        (
            (section, sentence)
            for section in document.sections
            for sentence in section.sentences
        ),
        max_concurrency=max_concurrency,
    )
    result.claims = [
        claim for sentence in result.sentences for claim in sentence.claims
    ]
    return result
