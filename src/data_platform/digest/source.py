"""Shared source-aligned document records and Markdown parsing."""

from typing import Annotated

from markdown_it import MarkdownIt
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from ._text.markdown import line_offsets
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


def parse_document(source: str) -> Document:
    if not source.strip():
        return Document(title="")
    offsets = line_offsets(source)
    tokens = MarkdownIt("commonmark").enable("table").parse(source)
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
    if source[offsets[title.map[1]] : offsets[subtitles[0][0].map[0]]].strip():
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
        excluded = []
        for token in tokens:
            if token.map is None:
                continue
            a, b = (offsets[n] for n in token.map)
            if start <= a < end and token.type in {
                "fence",
                "code_block",
                "html_block",
                "table_open",
            }:
                excluded.append(Span(start=a, end=b))
        section.excluded_spans = excluded
        for i, token in enumerate(tokens):
            if (
                token.type != "inline"
                or token.map is None
                or tokens[i - 1].type != "paragraph_open"
            ):
                continue
            a, b = (offsets[n] for n in token.map)
            if not start <= a < end or any(s.start <= a < s.end for s in excluded):
                continue
            for left, right in sentence_spans(source, a, b):
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
    return Document(title=title_text, sections=sections)
