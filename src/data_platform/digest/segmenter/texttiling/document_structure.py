"""Source-aligned Markdown structure and conservative heading candidates."""

import re
from dataclasses import dataclass, field, replace
from typing import Literal

from markdown_it import MarkdownIt

from ..._text.markdown import line_offsets
from ..._text.quotations import quotation_spans
from ..._text.sentences import sentence_spans

_METADATA = re.compile(
    r"^(?:published(?:\s+date|\s+on)?|publication date|date|ngày đăng(?: bài)?|"
    r"ngày xuất bản|đăng ngày|updated|cập nhật|author|by|tác giả)\b",
    re.IGNORECASE,
)
_DATE = re.compile(r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}(?:\s+.*)?$")
_URL_ONLY = re.compile(r"^(?:https?://|www\.)\S+$", re.IGNORECASE)
_WHOLE_QUOTE = re.compile(
    r"""^(?:“[\s\S]*”|‘[\s\S]*’|"[\s\S]*"|'[\s\S]*'|«[\s\S]*»)$"""
)


@dataclass(frozen=True)
class StructureBlock:
    start: int
    end: int
    kind: Literal["title", "metadata", "heading"]
    heading_level: int | None = None
    is_markdown: bool = False


@dataclass(frozen=True)
class HeadingCandidate:
    start: int
    end: int
    text: str
    previous_paragraph: str
    next_paragraph: str
    is_heading: bool | None = None


@dataclass(frozen=True)
class DocumentStructure:
    source: str = field(repr=False)
    blocks: tuple[StructureBlock, ...] = ()
    candidates: tuple[HeadingCandidate, ...] = ()
    classification_skipped: bool = False


def _is_metadata(line: str) -> bool:
    line = line.strip()
    if _DATE.fullmatch(line):
        return True
    if re.match(r"^(?i:by|author:|tác giả:)\s+[A-ZÀ-Ỹ][^.!?]{0,100}$", line):
        return True
    return bool(_METADATA.match(line) and re.search(r"\d", line))


def parse_document_structure(text: str) -> DocumentStructure:
    """Recognize explicit headings and metadata locally, and collect candidates.

    Candidates are not headings until a decision model confirms them.
    Markdown token line maps keep exclusions and offsets tied to source text.
    """
    lines = text.splitlines(keepends=True)
    offsets = line_offsets(text)
    tokens = MarkdownIt("commonmark").parse(text)
    blocks = []
    top_blocks = []
    title_seen = False
    metadata_allowed = True
    for token in tokens:
        if token.level != 0 or token.map is None:
            continue
        first, last = token.map
        start, end = offsets[first], offsets[last]
        if token.type == "heading_open":
            level = int(token.tag[1:])
            is_title = level == 1 and not title_seen
            blocks.append(
                StructureBlock(
                    start,
                    end,
                    "title" if is_title else "heading",
                    1 if is_title else max(2, level),
                    True,
                )
            )
            title_seen |= is_title
            metadata_allowed = is_title
            top_blocks.append((start, end, "heading"))
        elif token.type == "paragraph_open":
            paragraph_start = start
            if metadata_allowed:
                while first < last and _is_metadata(lines[first]):
                    first += 1
                if offsets[first] > start:
                    blocks.append(StructureBlock(start, offsets[first], "metadata"))
                    paragraph_start = offsets[first]
            if paragraph_start < end:
                metadata_allowed = False
                top_blocks.append((paragraph_start, end, "paragraph"))
            else:
                top_blocks.append((start, end, "metadata"))
        elif token.type in {
            "fence",
            "code_block",
            "blockquote_open",
            "bullet_list_open",
            "ordered_list_open",
            "html_block",
            "hr",
        }:
            top_blocks.append((start, end, "excluded"))
            metadata_allowed = False
    quotes = quotation_spans(text)
    candidates = []
    for i, (start, end, kind) in enumerate(top_blocks):
        raw = text[start:end]
        line = raw.strip()
        if (
            kind != "paragraph"
            or any(q.start < end and q.end > start for q in quotes)
            or len(line.splitlines()) != 1
            or len(line) > 240
            or len(re.findall(r"\w+", line)) > 40
            or not re.search(r"\w", line)
            or _is_metadata(line)
            or _URL_ONLY.fullmatch(line)
            or _WHOLE_QUOTE.fullmatch(line)
        ):
            continue
        # A punctuation-free short label is one sentence span as well.
        if len(sentence_spans(line, 0, len(line))) != 1:
            continue
        if (
            re.fullmatch(r"!?\[[^\]]*\]\([^)]*\)", line)
            or re.fullmatch(r"`+[^`]+`+", line)
            or re.fullmatch(r"<(?:https?://|www\.)[^>]+>", line)
        ):
            continue
        previous = text[slice(*top_blocks[i - 1][:2])] if i else ""
        following = (
            text[slice(*top_blocks[i + 1][:2])] if i + 1 < len(top_blocks) else ""
        )
        candidates.append(HeadingCandidate(start, end, raw, previous, following))
    return DocumentStructure(text, tuple(blocks), tuple(candidates))


def confirm_candidates(
    structure: DocumentStructure, decisions: list[bool]
) -> DocumentStructure:
    """Apply ordered decisions without altering source content."""
    if len(decisions) != len(structure.candidates):
        raise ValueError("Decision count must match heading candidate count")
    blocks = list(structure.blocks)
    has_title = any(block.kind == "title" for block in blocks)
    candidates = []
    for candidate, decision in zip(structure.candidates, decisions, strict=True):
        if type(decision) is not bool:
            raise TypeError("Heading decisions must be booleans")
        candidates.append(replace(candidate, is_heading=decision))
        if decision:
            is_title = not has_title and not structure.source[: candidate.start].strip()
            blocks.append(
                StructureBlock(
                    candidate.start,
                    candidate.end,
                    "title" if is_title else "heading",
                    1 if is_title else 2,
                )
            )
            has_title |= is_title
    # A newly confirmed opening title may expose metadata that local parsing
    # previously treated as part of the body.
    for title in [block for block in blocks if block.kind == "title"]:
        cursor = title.end
        metadata_start = None
        for line in structure.source[cursor:].splitlines(keepends=True):
            if not line.strip():
                cursor += len(line)
                continue
            if not _is_metadata(line):
                break
            if metadata_start is None:
                metadata_start = cursor
            cursor += len(line)
        if metadata_start is not None and not any(
            block.start <= metadata_start < block.end for block in blocks
        ):
            blocks.append(StructureBlock(metadata_start, cursor, "metadata"))
    return replace(
        structure,
        blocks=tuple(sorted(blocks, key=lambda b: b.start)),
        candidates=tuple(candidates),
    )


def document_parts(text: str, structure: DocumentStructure | None = None):
    """Return contiguous (start, end, structural block or None) source partitions."""
    structure = structure if structure is not None else parse_document_structure(text)
    if structure.source != text:
        raise ValueError("Document structure belongs to different source text")
    parts = []
    cursor = 0

    def append_body(start, end):
        if parts:
            leading = re.match(r"(?:[ \t]*\r?\n)+", text[start:end])
            if leading:
                start += leading.end()
                a, _, previous = parts[-1]
                parts[-1] = (a, start, previous)
        if start < end:
            parts.append((start, end, None))

    for block in structure.blocks:
        if not (cursor <= block.start < block.end <= len(text)):
            raise ValueError(
                "Structural spans must be ordered, nonoverlapping source spans"
            )
        start = block.start
        if text[cursor:start].strip():
            append_body(cursor, start)
        elif parts:
            a, _, previous = parts[-1]
            parts[-1] = (a, start, previous)
        else:
            start = 0
        parts.append((start, block.end, block))
        cursor = block.end
    if text[cursor:].strip() or not parts:
        append_body(cursor, len(text))
    elif cursor < len(text):
        a, _, previous = parts[-1]
        parts[-1] = (a, len(text), previous)
    return parts
