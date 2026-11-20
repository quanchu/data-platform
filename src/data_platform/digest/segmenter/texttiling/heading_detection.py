"""Bounded, type-safe decisions for rule-filtered unformatted headings."""

import json
import logging
import os
from dataclasses import replace

from pydantic import BaseModel, StrictBool
from pydantic_ai import Agent
from pydantic_ai.models import Model

from .prompts import DECISION_INSTRUCTIONS as DECISION_INSTRUCTIONS

from ..._events import emit
from ..._async import ordered_map, positive_int
from .document_structure import (
    DocumentStructure,
    confirm_candidates,
    parse_document_structure,
)

_LOG = logging.getLogger(__name__)


class HeadingDecision(BaseModel):
    is_heading: StrictBool


class HeadingDetectionError(RuntimeError):
    def __init__(self, candidate_index: int):
        self.candidate_index = candidate_index
        super().__init__(f"Heading detection failed for candidate {candidate_index}")


async def detect_document_structure(
    text: str,
    *,
    decision_model: str | Model = "openai:gpt-6-luna",
    max_concurrency: int = 20,
) -> DocumentStructure:
    """Classify candidates; without OpenAI credentials leave candidates unchanged.

    Configured model instances use their own credentials. An attempted request
    failure is surfaced, never treated as a negative classification.
    """
    positive_int(max_concurrency, "max_concurrency")
    structure = parse_document_structure(text)
    if not structure.candidates:
        return structure
    if (
        isinstance(decision_model, str)
        and decision_model.startswith("openai:")
        and not os.getenv("OPENAI_API_KEY", "").strip()
    ):
        _LOG.warning(
            "Unformatted heading classification skipped: OPENAI_API_KEY is missing"
        )
        return replace(structure, classification_skipped=True)
    try:
        agent = Agent(
            decision_model,
            output_type=HeadingDecision,
            retries={"output": 2},
            instructions=DECISION_INSTRUCTIONS,
        )
    except Exception as exc:
        raise HeadingDetectionError(0) from exc
    title = next(
        (text[b.start : b.end] for b in structure.blocks if b.kind == "title"), ""
    )

    async def process(item):
        index, candidate = item
        spans = [(candidate.start, candidate.end)]
        emit(
            "heading_detection",
            status="running",
            spans=spans,
            prompt=DECISION_INSTRUCTIONS,
        )
        try:
            response = await agent.run(
                json.dumps(
                    {
                        "document_title": title,
                        "candidate": candidate.text,
                        "previous_paragraph": candidate.previous_paragraph,
                        "next_paragraph": candidate.next_paragraph,
                    },
                    ensure_ascii=False,
                )
            )
            emit(
                "heading_detection", spans=spans, is_heading=response.output.is_heading
            )
            return response.output.is_heading
        except Exception as exc:
            emit("heading_detection", status="failed", spans=spans, error=str(exc))
            raise HeadingDetectionError(index) from exc

    decisions = await ordered_map(
        process, enumerate(structure.candidates), max_concurrency=max_concurrency
    )
    return confirm_candidates(structure, decisions)
