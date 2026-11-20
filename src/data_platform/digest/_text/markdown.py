"""Source-aligned Markdown blocks; parser policy stays with each consumer."""

from dataclasses import dataclass

from markdown_it import MarkdownIt
from markdown_it.token import Token
from markdown_it.tree import SyntaxTreeNode


@dataclass(frozen=True)
class MarkdownBlock:
    kind: str
    start: int
    end: int
    children: tuple["MarkdownBlock", ...] = ()


def parse_tokens(source: str, *, tables: bool = False) -> list[Token]:
    parser = MarkdownIt("commonmark")
    if tables:
        parser.enable("table")
    return parser.parse(source)


def parse_blocks(source: str, *, tables: bool = False) -> tuple[MarkdownBlock, ...]:
    """Keep original character spans, including list continuation paragraphs."""
    offsets = line_offsets(source)

    def blocks(nodes):
        return tuple(
            MarkdownBlock(
                node.type,
                offsets[node.map[0]],
                offsets[node.map[1]],
                blocks(node.children),
            )
            for node in nodes
            if node.map is not None and node.type != "inline"
        )

    return blocks(SyntaxTreeNode(parse_tokens(source, tables=tables)).children)


def list_spans(source: str) -> list[tuple[int, int]]:
    """Protect outermost lists and their immediately preceding colon introduction."""
    blocks = parse_blocks(source)
    spans = []
    for index, block in enumerate(blocks):
        if block.kind not in {"bullet_list", "ordered_list"}:
            continue
        start = block.start
        if index:
            previous = blocks[index - 1]
            if (
                previous.kind == "paragraph"
                and source[previous.start : previous.end].rstrip().endswith(":")
                and not source[previous.end : start].strip()
            ):
                start = previous.start
        spans.append((start, block.end))
    return spans


def line_offsets(source: str) -> list[int]:
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets
