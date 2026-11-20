"""Generate section boundaries and headings without rewriting source blocks."""

import json
import re
from collections.abc import Sequence

from pydantic_ai import Agent, ModelRetry
from pydantic_ai.models import Model

from .prompts import INSTRUCTIONS as INSTRUCTIONS

from ..._events import emit
from ..._text.quotations import quotation_spans
from ..._text.markdown import list_spans

DEFAULT_MODEL = "openai:gpt-6-astra"
RESERVED_MARKER = "<SECTION_BOUNDARY>"
_BLANK_LINE = re.compile(r"\r?\n[^\S\r\n]*\r?\n")


def split_blocks(markdown_doc: str) -> list[str]:
    """Split blank lines and quote boundaries, preserving text within each block.

    Blank-line pairs outside quotations become separators. Consecutive pairs and
    trailing pairs retain empty blocks. Quote boundaries introduce no empty blocks.
    Reconstruction normalizes separators; it does not preserve whole-file bytes.
    """
    protected_lists = list_spans(markdown_doc)
    quotes = [
        quote
        for quote in quotation_spans(markdown_doc)
        if not any(
            start < quote.end and quote.start < end for start, end in protected_lists
        )
    ]
    events = [(q.start, q.start) for q in quotes]
    events.extend((offset, offset) for span in protected_lists for offset in span)
    events.extend((q.end, q.end) for q in quotes)
    events.extend(
        (m.start(), m.end())
        for m in _BLANK_LINE.finditer(markdown_doc)
        if not any(q.start < m.end() and m.start() < q.end for q in quotes)
        and not any(
            start < m.end() and m.start() < end for start, end in protected_lists
        )
    )
    blocks = []
    cursor = 0
    previous_separator = False
    for start, end in sorted(set(events)):
        if end > start:
            if cursor < start or previous_separator or start == 0:
                blocks.append(markdown_doc[cursor:start])
            cursor = end
            previous_separator = True
        elif start > cursor:
            blocks.append(markdown_doc[cursor:start])
            cursor = start
            previous_separator = False
    if cursor < len(markdown_doc) or previous_separator or not blocks:
        blocks.append(markdown_doc[cursor:])
    return blocks


def _validate(blocks: Sequence[str], headings: Sequence[str]) -> None:
    if len(headings) != len(blocks):
        raise ValueError(
            f"Expected exactly {len(blocks)} entries; received {len(headings)}."
        )
    for index, (block, heading) in enumerate(zip(blocks, headings, strict=True)):
        if not isinstance(heading, str):
            raise TypeError(f"Entry {index} must be a string.")
        if not block.strip() and heading != "":
            raise ValueError(f"Entry {index} must be empty because its block is empty.")
        if heading and (
            not heading.startswith("## ")
            or not heading[3:].strip()
            or "\n" in heading
            or "\r" in heading
            or RESERVED_MARKER in heading
        ):
            raise ValueError(
                f"Entry {index} must be empty or start with '## ', contain heading "
                "text, have no newline/carriage return, and exclude "
                f"{RESERVED_MARKER}."
            )


async def generate_block_headings(
    blocks: Sequence[str],
    *,
    model: str | Model = DEFAULT_MODEL,
    retries: int = 2,
) -> list[str]:
    """Request all headings together; retry format errors through Pydantic AI."""
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ValueError("retries must be a nonnegative integer")
    blocks = tuple(blocks)
    if not any(block.strip() for block in blocks):
        return [""] * len(blocks)
    agent = Agent(
        model,
        output_type=list[str],
        instructions=INSTRUCTIONS,
        retries={"output": retries},
    )

    @agent.output_validator
    def validate_output(headings: list[str]) -> list[str]:
        try:
            _validate(blocks, headings)
        except ValueError as exc:
            emit("block_headings", status="retrying", error=str(exc))
            raise ModelRetry(str(exc)) from exc
        return headings

    emit("block_headings", status="running", prompt=INSTRUCTIONS, blocks=list(blocks))
    result = await agent.run(
        json.dumps(
            [
                {"index": index, "marker": RESERVED_MARKER, "text": block}
                for index, block in enumerate(blocks)
            ],
            ensure_ascii=False,
        )
    )
    emit("block_headings", headings=result.output)
    return result.output


def reconstruct_document(blocks: Sequence[str], headings: Sequence[str]) -> str:
    """Validate headings, then prepend them to untouched source blocks."""
    _validate(blocks, headings)
    return "\n\n".join(
        heading + "\n\n" + block if heading else block
        for block, heading in zip(blocks, headings, strict=True)
    )


async def llm_segment(
    markdown_doc: str,
    *,
    model: str | Model = DEFAULT_MODEL,
    retries: int = 2,
) -> str:
    """Return segmented Markdown; credentials and file operations belong to callers."""
    blocks = split_blocks(markdown_doc)
    spans = []
    cursor = 0
    for index, block in enumerate(blocks):
        start = markdown_doc.find(block, cursor)
        end = start + len(block)
        spans.append((start, end))
        cursor = end
        emit("input_block", spans=[(start, end)], block_index=index, text=block)
    headings = await generate_block_headings(blocks, model=model, retries=retries)
    for span, heading in zip(spans, headings, strict=True):
        if heading:
            emit("heading_generation", spans=[span], heading=heading)
    return reconstruct_document(blocks, headings)
