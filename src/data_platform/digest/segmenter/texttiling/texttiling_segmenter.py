"""Deterministic Unicode TextTiling with lossless paragraph-aligned sections."""

import itertools
import math
import re
import unicodedata
from collections import Counter
from dataclasses import replace
from typing import Literal

from .document_structure import (
    DocumentStructure,
    document_parts,
    parse_document_structure,
)
from .models import Section
from .quotations import body_spans, prose_paragraphs

_WORD = re.compile(r"[^\W_]+(?:[\u0300-\u036f][^\W_]*)*", re.UNICODE)


def _cosine(left: Counter, right: Counter) -> float:
    denominator = math.sqrt(
        sum(v * v for v in left.values()) * sum(v * v for v in right.values())
    )
    return (
        sum(v * right[k] for k, v in left.items()) / denominator if denominator else 0.0
    )


def _topic_boundaries(
    text: str, paragraphs: list[tuple[int, int]], tfidf=None
) -> list[int]:
    """Return paragraph indices at depth peaks in smoothed lexical similarity.

    Use 20-token blocks and up to four blocks per adjacent comparison window.
    Smooth with a centered three-gap mean; select positive local depth maxima
    above mean minus half a standard deviation. Snap each to the nearest actual
    paragraph break. Short inputs and flat similarity curves have no boundaries.
    """
    if len(paragraphs) < 2:
        return []
    start, end = paragraphs[0][0], paragraphs[-1][1]
    matches = list(_WORD.finditer(text, start, end))
    tokens = [unicodedata.normalize("NFC", m.group()).casefold() for m in matches]
    blocks = [Counter(tokens[i : i + 20]) for i in range(0, len(tokens), 20)]
    if len(blocks) < 4:
        return []
    scores = []
    for gap in range(1, len(blocks)):
        width = min(4, gap, len(blocks) - gap)
        left = sum(blocks[gap - width : gap], Counter())
        right = sum(blocks[gap : gap + width], Counter())
        if tfidf is None:
            scores.append(_cosine(left, right))
        else:
            windows = tfidf.transform(
                [
                    " ".join(tokens[(gap - width) * 20 : gap * 20]),
                    " ".join(tokens[gap * 20 : (gap + width) * 20]),
                ]
            )
            scores.append(float(windows[0].multiply(windows[1]).sum()))
    positions = [matches[(i + 1) * 20].start() for i in range(len(scores))]
    return _select_boundaries(scores, positions, paragraphs)


def _select_boundaries(scores, positions, paragraphs) -> list[int]:
    """Select depth peaks and align them to original paragraph offsets."""
    if len(scores) < 3:
        return []
    smooth = [
        sum(scores[max(0, i - 1) : i + 2]) / len(scores[max(0, i - 1) : i + 2])
        for i in range(len(scores))
    ]
    depths = []
    for i, value in enumerate(smooth):
        left = right = i
        while left > 0 and smooth[left - 1] >= smooth[left]:
            left -= 1
        while right + 1 < len(smooth) and smooth[right + 1] >= smooth[right]:
            right += 1
        depths.append(smooth[left] + smooth[right] - 2 * value)
    mean = sum(depths) / len(depths)
    std = math.sqrt(sum((d - mean) ** 2 for d in depths) / len(depths))
    boundaries = set()
    for i in range(1, len(depths) - 1):
        depth = depths[i]
        # A strict comparison on one side chooses only one point on a plateau.
        if (
            depth > max(1e-10, mean - std / 2)
            and depth > depths[i - 1]
            and depth >= depths[i + 1]
        ):
            position = positions[i]
            boundary = min(
                range(1, len(paragraphs)),
                key=lambda j: abs(paragraphs[j][0] - position),
            )
            boundaries.add(boundary)
    return sorted(boundaries)


def _segment_body(text: str, boundary_function) -> list[Section]:
    """Isolate every quotation before selecting topic boundaries in prose runs."""
    sections = []
    for start, end, kind in body_spans(text):
        if kind == "quote":
            sections.append(Section(text[start:end], start, end, "quote"))
            continue
        paragraphs = prose_paragraphs(text, start, end)
        boundaries = [0, *boundary_function(text, paragraphs), len(paragraphs)]
        for begin, finish in itertools.pairwise(boundaries):
            for cursor in range(begin, finish, 4):
                a, b = paragraphs[cursor][0], paragraphs[min(cursor + 4, finish) - 1][1]
                sections.append(Section(text[a:b], a, b, "prose"))
    return sections


def _segment(
    text: str, boundary_function, structure: DocumentStructure | None = None
) -> list[Section]:
    sections = []
    for start, end, block in document_parts(text, structure):
        if block is not None:
            sections.append(
                Section(
                    text[start:end],
                    start,
                    end,
                    block.kind,
                    heading_level=block.heading_level,
                    heading_is_markdown=block.is_markdown,
                )
            )
            continue

        def boundaries(_, paragraphs, start=start):
            return boundary_function(
                text, [(a + start, b + start) for a, b in paragraphs]
            )

        sections.extend(
            replace(section, start=section.start + start, end=section.end + start)
            for section in _segment_body(text[start:end], boundaries)
        )
    return sections


def _lexical_tokens(text: str) -> list[str]:
    return [
        unicodedata.normalize("NFC", m.group()).casefold() for m in _WORD.finditer(text)
    ]


def _fit_tfidf(text: str, structure: DocumentStructure | None = None):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:
        raise ImportError("TF-IDF requires: uv sync --extra offline") from exc
    blocks = []
    for begin, finish, structural in document_parts(text, structure):
        if structural is not None:
            continue
        body = text[begin:finish]
        for start, end, kind in body_spans(body):
            if kind == "quote":
                continue
            for a, b in prose_paragraphs(body, start, end):
                tokens = _lexical_tokens(body[a:b])
                blocks.extend(
                    " ".join(tokens[i : i + 20]) for i in range(0, len(tokens), 20)
                )
    if not blocks:
        return None
    return TfidfVectorizer(
        tokenizer=_lexical_tokens,
        # A custom tokenizer disables the regex; sklearn accepts None at runtime.
        token_pattern=None,  # pyright: ignore[reportArgumentType]
        lowercase=False,
        ngram_range=(1, 2),
        sublinear_tf=True,
        smooth_idf=True,
        norm="l2",
    ).fit(blocks)


def segment_article(
    text: str,
    *,
    vectorizer: Literal["counts", "tfidf"] = "counts",
    structure: DocumentStructure | None = None,
) -> list[Section]:
    """Offline segmentation; TF-IDF is an optional alternative to raw counts."""
    if vectorizer not in {"counts", "tfidf"}:
        raise ValueError("vectorizer must be 'counts' or 'tfidf'")
    structure = structure if structure is not None else parse_document_structure(text)
    if vectorizer == "counts":
        return _segment(text, _topic_boundaries, structure)
    if not text.strip():
        return []
    tfidf = _fit_tfidf(text, structure)
    return _segment(
        text,
        lambda source, paragraphs: (
            _topic_boundaries(source, paragraphs, tfidf) if tfidf is not None else []
        ),
        structure,
    )
