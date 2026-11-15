"""Shared deterministic sentence spans for candidate filtering and embeddings."""

import re

_END = re.compile(r"""[.!?…]+["'’”»\)\]]*(?=\s|$)""")
_URL = re.compile(r"(?:https?://|www\.)\S+")
_ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "vs",
    "etc",
    "e.g",
    "i.e",
    "ts",
    "ths",
    "pgs",
    "gs",
    "tp",
    "q",
    "p",
    "st",
    "no",
}


def sentence_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Return source spans, protecting common abbreviations, initials and URLs."""
    paragraph = text[start:end]
    urls = [
        (m.start(), m.start() + len(m.group().rstrip(".,!?;:")))
        for m in _URL.finditer(paragraph)
    ]
    spans = []
    cursor = 0
    for match in _END.finditer(paragraph):
        if any(a <= match.start() < b for a, b in urls):
            continue
        before = paragraph[: match.start()]
        word = re.search(r"([\w.]+)$", before)
        if match.group().startswith(".") and word:
            token = word.group(1)
            if token.casefold() in _ABBREVIATIONS or (
                len(token) == 1 and token.isupper()
            ):
                continue
        if paragraph[cursor : match.end()].strip():
            spans.append((start + cursor, start + match.end()))
        cursor = match.end()
    if paragraph[cursor:].strip():
        spans.append((start + cursor, end))
    return spans
