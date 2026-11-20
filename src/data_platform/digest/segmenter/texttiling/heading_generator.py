"""Concurrent, article-grounded headings for independently segmented sections."""

import json
from collections.abc import Sequence
from dataclasses import replace

from pydantic import BaseModel, field_validator
from pydantic_ai import Agent
from pydantic_ai.models import Model

from .prompts import HEADING_INSTRUCTIONS as HEADING_INSTRUCTIONS

from ..._events import emit
from ..._async import ordered_map, positive_int
from .models import Section

DEFAULT_MODEL = "openai:gpt-6-luna"


class Heading(BaseModel):
    heading: str

    @field_validator("heading")
    @classmethod
    def validate_heading(cls, value: str) -> str:
        if (
            any(c in value for c in "\r\n\v\f\x1c\x1d\x1e\x85\u2028\u2029")
            or not value.strip()
        ):
            raise ValueError("Heading must be nonempty and single-line")
        return value.strip()


class HeadingGenerationError(RuntimeError):
    """A heading request failed; section_index is zero-based."""

    def __init__(self, section_index: int):
        self.section_index = section_index
        super().__init__(f"Heading generation failed for section {section_index}")


def _eligible_sections(sections: Sequence[Section]) -> list[tuple[int, Section]]:
    """Skip titles, metadata, and the first body below an existing heading."""
    eligible: list[tuple[int, Section]] = []
    suppress_next = False
    before_title = any(section.kind == "title" for section in sections)
    for index, section in enumerate(sections):
        if section.kind == "title":
            before_title = False
            suppress_next = False
        elif section.kind == "heading":
            suppress_next = True
        elif section.kind == "metadata":
            continue
        elif not before_title:
            if not suppress_next:
                eligible.append((index, section))
            suppress_next = False
    return eligible


async def generate_headings(
    sections: Sequence[Section],
    *,
    model: str | Model = DEFAULT_MODEL,
    article_text: str | None = None,
    max_concurrency: int = 20,
) -> list[Section]:
    """Return new sections in input order, with two validation retries per section.

    Context defaults to all supplied sections; pass article_text for subsets.
    On failure, cancel and await remaining workers, preserving the failing index.
    """
    positive_int(max_concurrency, "max_concurrency")
    if not sections:
        return []
    sections = tuple(sections)
    context = (
        article_text if article_text is not None else "".join(s.text for s in sections)
    )
    eligible = _eligible_sections(sections)
    if not eligible:
        return list(sections)
    try:
        agent = Agent(
            model,
            output_type=Heading,
            retries={"output": 2},
            instructions=HEADING_INSTRUCTIONS,
        )
    except Exception as exc:
        raise HeadingGenerationError(eligible[0][0]) from exc
    results = list(sections)

    async def process(item):
        index, section = item
        spans = [(section.start, section.end)]
        emit(
            "heading_generation",
            status="running",
            spans=spans,
            prompt=HEADING_INSTRUCTIONS,
        )
        try:
            response = await agent.run(
                json.dumps(
                    {
                        "article_context": context,
                        "target_section": {
                            "index": index,
                            "kind": section.kind,
                            "text": section.text,
                        },
                    },
                    ensure_ascii=False,
                )
            )
            emit("heading_generation", spans=spans, heading=response.output.heading)
            return index, replace(section, heading=response.output.heading)
        except Exception as exc:
            emit("heading_generation", status="failed", spans=spans, error=str(exc))
            raise HeadingGenerationError(index) from exc

    generated = await ordered_map(process, eligible, max_concurrency=max_concurrency)
    for index, section in generated:
        results[index] = section
    return results
