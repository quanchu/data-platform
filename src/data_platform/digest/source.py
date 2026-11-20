"""Shared source-aligned document records and Markdown parsing."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from ._events import emit
from ._text.markdown import line_offsets, parse_tokens
from ._text.metadata import is_metadata_line
from ._text.sentences import sentence_spans

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Span(Record):
    start: int
    end: int


class Sentence(Span):
    index: int
    section_index: int
    text: str


class Section(Span):
    index: int
    subtitle: str
    text: str
    sentences: list[Sentence] = Field(default_factory=list)
    excluded_spans: list[Span] = Field(default_factory=list)


class Document(Record):
    title: str
    sections: list[Section] = Field(default_factory=list)


def _excluded_spans(tokens, offsets: list[int], start: int, end: int) -> list[Span]:
    excluded = []
    for token in tokens:
        if token.map is None or token.type not in {
            "fence",
            "code_block",
            "html_block",
            "table_open",
        }:
            continue
        block_start, block_end = (offsets[line] for line in token.map)
        if start <= block_start < end:
            excluded.append(Span(start=block_start, end=block_end))
    return excluded


def _paragraph_ranges(tokens, offsets: list[int], section: Section):
    """Yield eligible paragraph source ranges, including paragraphs in list items."""
    for index, token in enumerate(tokens):
        if (
            token.type != "inline"
            or token.map is None
            or tokens[index - 1].type != "paragraph_open"
        ):
            continue
        start, end = (offsets[line] for line in token.map)
        if section.start <= start < section.end and not any(
            span.start <= start < span.end for span in section.excluded_spans
        ):
            yield start, end


def parse_document(source: str) -> Document:
    if not source.strip():
        return Document(title="")
    offsets = line_offsets(source)
    tokens = parse_tokens(source, tables=True)
    headings = []
    for i, token in enumerate(tokens):
        if token.type == "heading_open" and token.level == 0:
            headings.append((token, tokens[i + 1].content.strip()))
    titles = [(t, s) for t, s in headings if t.tag == "h1"]
    if len(titles) != 1 or not titles[0][1]:
        raise ValueError("Document requires exactly one nonempty H1 title")
    title, title_text = titles[0]
    subtitles = [(t, s) for t, s in headings if t.tag == "h2"]
    if not subtitles or any(not s for _, s in subtitles):
        raise ValueError("Document requires nonempty H2 section subtitles")
    if source[: offsets[title.map[0]]].strip() or subtitles[0][0].map[0] < title.map[1]:
        raise ValueError("H1 must precede all section content")
    preamble = source[offsets[title.map[1]] : offsets[subtitles[0][0].map[0]]]
    if any(
        line.strip() and not is_metadata_line(line) for line in preamble.splitlines()
    ):
        raise ValueError("Body content must be beneath an H2 subtitle")
    sections = []
    sentence_index = 0
    for index, (heading, subtitle) in enumerate(subtitles):
        start = offsets[heading.map[1]]
        end = (
            offsets[subtitles[index + 1][0].map[0]]
            if index + 1 < len(subtitles)
            else len(source)
        )
        section = Section(
            index=index, subtitle=subtitle, start=start, end=end, text=source[start:end]
        )
        section.excluded_spans = _excluded_spans(tokens, offsets, start, end)
        for paragraph_start, paragraph_end in _paragraph_ranges(
            tokens, offsets, section
        ):
            for left, right in sentence_spans(source, paragraph_start, paragraph_end):
                section.sentences.append(
                    Sentence(
                        index=sentence_index,
                        section_index=index,
                        start=left,
                        end=right,
                        text=source[left:right],
                    )
                )
                sentence_index += 1
        sections.append(section)
    document = Document(title=title_text, sections=sections)
    emit(
        "source_parsing",
        spans=[(s.start, s.end) for s in sections],
        document=document.model_dump(),
    )
    return document
