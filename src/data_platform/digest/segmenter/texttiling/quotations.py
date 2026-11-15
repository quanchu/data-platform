"""Deterministic, source-aligned quotation and attribution spans."""

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from markdown_it import MarkdownIt

from .sentences import sentence_spans

_PARAGRAPH_BREAK = re.compile(r"(?:\r?\n)[^\S\r\n]*(?:\r?\n)(?:[^\S\r\n]*(?:\r?\n))*")
_PAIRS = {"“": "”", "‘": "’", "«": "»", '"': '"', "'": "'"}
_REPORT_WORDS = (
    r"said|says?|state(?:s|d)?|comment(?:s|ed)?|affirm(?:s|ed)?|add(?:s|ed)?|"
    r"explain(?:s|ed)?|note(?:s|d)?|told|tells?|announce(?:s|d)?|emphasize(?:s|d)?|"
    r"stress(?:es|ed)?|replied|replies|ask(?:s|ed)?|warn(?:s|ed)?|wrote|writes?|"
    r"declare(?:s|d)?|confirm(?:s|ed)?|cho biết(?: thêm)?|cho hay|nói(?: thêm)?|"
    r"chia sẻ(?: thêm)?|bổ sung|khẳng định|nhấn mạnh|nhận định|đánh giá|bày tỏ|"
    r"tuyên bố|phát biểu|nêu rõ|kết luận"
)
_REPORT = re.compile(rf"\b(?:{_REPORT_WORDS})\b", re.IGNORECASE)
_PREFIX_REPORT = re.compile(
    rf"\b(?:{_REPORT_WORDS})(?:\s+(?:that|rằng))?[\s,:—–-]*$", re.IGNORECASE
)


@dataclass(frozen=True)
class QuotationSpan:
    start: int
    end: int


def paragraph_spans(text: str) -> list[tuple[int, int]]:
    """Attach separators to the preceding paragraph and retain edge whitespace."""
    spans = []
    start = 0
    for match in _PARAGRAPH_BREAK.finditer(text):
        if text[start : match.start()].strip():
            spans.append((start, match.end()))
            start = match.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    elif spans:
        spans[-1] = (spans[-1][0], len(text))
    return spans


def _markdown_spans(text):
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    blocks, excluded = [], []
    for token in MarkdownIt("commonmark").parse(text):
        if token.map is None or token.level != 0:
            continue
        start, end = (offsets[i] for i in token.map)
        if token.type == "blockquote_open":
            blocks.append((start, end))
        elif token.type in {"fence", "code_block", "html_block", "heading_open"}:
            excluded.append((start, end))
    # Inline code is source data, not spoken quotation.
    excluded.extend(
        (m.start(), m.end()) for m in re.finditer(r"(`+)([^`]|(?!\1)`)*?\1", text)
    )
    return blocks, sorted(excluded)


def _marked_spans(text, excluded):
    stack = []
    result = []
    for i, char in enumerate(text):
        if char not in {*_PAIRS, *_PAIRS.values()}:
            continue
        if any(a <= i < b for a, b in excluded):
            continue
        # Backslash-escaped marks and apostrophes inside words are not delimiters.
        preceding = i - 1
        while preceding >= 0 and text[preceding] == "\\":
            preceding -= 1
        if (i - preceding - 1) % 2:
            continue
        if (
            char in {"'", "’"}
            and i
            and text[i - 1].isalnum()
            and i + 1 < len(text)
            and text[i + 1].isalnum()
        ):
            continue
        if stack and char == stack[-1][1]:
            start, _ = stack.pop()
            if not any(a < i + 1 and b > start for a, b in excluded):
                result.append((start, i + 1))
        elif char in _PAIRS:
            # Do not treat trailing possessive apostrophes as new quotations.
            if char == "'" and i and text[i - 1].isalnum():
                continue
            stack.append((i, _PAIRS[char]))
    # Keep outer completed quotations. If an outer mark is unmatched, recover
    # independently completed inner spans rather than losing later quotations.
    return sorted(
        (a, b) for a, b in result if not any(c < a and b <= d for c, d in result)
    )


def _prefix_start(text, left, opening):
    spans = sentence_spans(text, left, opening)
    if not spans:
        return None
    start, end = spans[-1]
    candidate = text[start:end]
    if _PREFIX_REPORT.search(unicodedata.normalize("NFC", candidate)) and re.search(
        r"\w", candidate
    ):
        return start + len(candidate) - len(candidate.lstrip())
    return None


def _attribution_only(text):
    """Recognize a reporting clause, not a subsequent narrative sentence."""
    clause = unicodedata.normalize("NFC", text.strip(" \t\r\n,;:—–-"))
    if not clause or len(clause) > 240:
        return False
    report = _REPORT.search(clause)
    if report is None:
        return False
    before, after = (
        clause[: report.start()].strip(),
        clause[report.end() :].strip(" .,;:—–-"),
    )
    # Subject + reporting verb, or reporting verb + subject/name/title.
    if before and not after:
        return not re.search(r"[!?;]", before)
    if not before and after:
        return not re.search(r"\b(?:that|because|when|rằng|vì)\b", after, re.IGNORECASE)
    return False


def _suffix_end(text, closing, right):
    spans = sentence_spans(text, closing, right)
    if not spans:
        return None
    _, end = spans[0]
    candidate = text[closing:end]
    # A trailing colon introduces the next quotation rather than closing this one.
    if candidate.rstrip().endswith(":") or (
        _PREFIX_REPORT.search(unicodedata.normalize("NFC", candidate))
        and not candidate.rstrip().endswith((".", "!", "?", ","))
    ):
        return None
    if _attribution_only(candidate):
        return end
    return None


def quotation_spans(text: str) -> list[QuotationSpan]:
    """Find independent utterances, including adjacent reporting clauses.

    This is a conservative bilingual heuristic, not a speaker-identity model.
    Separate utterances never merge merely because the speaker is the same.
    """
    paragraphs = paragraph_spans(text)
    blockquotes, excluded = _markdown_spans(text)
    marked = _marked_spans(text, [*excluded, *blockquotes])
    # Merge only a continuous utterance interrupted by its attribution.
    grouped = []
    for start, end in marked:
        if grouped:
            previous_start, previous_end = grouped[-1]
            bridge = text[previous_end:start]
            if (
                text[previous_end - 2] not in ".!?…"
                and not _PARAGRAPH_BREAK.search(bridge)
                and bridge.rstrip().endswith((",", "—", "–"))
                and _attribution_only(bridge)
            ):
                grouped[-1] = (previous_start, end)
                continue
        grouped.append((start, end))
    result = []
    previous_end = 0
    for index, (opening, closing) in enumerate(grouped):
        para_start = next((a for a, b in paragraphs if a <= opening < b), 0)
        para_end = next((b for a, b in paragraphs if a < closing <= b), len(text))
        left = max(para_start, previous_end)
        right = min(
            para_end, grouped[index + 1][0] if index + 1 < len(grouped) else len(text)
        )
        prefix = _prefix_start(text, left, opening)
        # Support a reporting clause in the immediately preceding paragraph.
        if (
            prefix is None
            and not text[para_start:opening].strip()
            and para_start > previous_end
        ):
            prior = next(((a, b) for a, b in paragraphs if b == para_start), None)
            if prior:
                prefix = _prefix_start(text, max(prior[0], previous_end), opening)
        suffix = _suffix_end(text, closing, right)
        outside = text[para_start:opening] + text[closing:para_end]
        standalone = not outside.strip(" \t\r\n.!?,;")
        # A paragraph consisting solely of several quoted utterances retains
        # separate sections, even without explicit attribution on every one.
        cursor = para_start
        residue = []
        for a, b in grouped:
            if para_start <= a and b <= para_end:
                residue.append(text[cursor:a])
                cursor = b
        residue.append(text[cursor:para_end])
        standalone |= not "".join(residue).strip(" \t\r\n.!?,;")
        standalone |= bool(
            result
            and result[-1][1] >= para_start
            and not text[result[-1][1] : opening].strip()
        )
        if prefix is not None or suffix is not None or standalone:
            start, end = (
                prefix if prefix is not None else opening,
                suffix if suffix is not None else closing,
            )
            # Punctuation outside the closing delimiter belongs to the quote.
            while end < right and text[end] in ".!?":
                end += 1
            result.append((start, end))
            previous_end = end
        else:
            previous_end = closing
    for start, end in blockquotes:
        left = next((a for a, b in paragraphs if b == start), start)
        prefix = _prefix_start(text, left, start) if left < start else None
        result.append((prefix if prefix is not None else start, end))
    return [QuotationSpan(a, b) for a, b in sorted(result)]


def body_spans(text: str) -> list[tuple[int, int, Literal["prose", "quote"]]]:
    """Partition a body into contiguous prose/quote spans with no discarded text."""
    result = []
    cursor = 0
    for quote in quotation_spans(text):
        start = quote.start
        if start < cursor:
            continue
        if text[cursor:start].strip():
            result.append((cursor, start, "prose"))
        elif result:
            a, _, kind = result[-1]
            result[-1] = (a, start, kind)
        else:
            start = cursor
        end = quote.end
        while end < len(text) and text[end].isspace():
            end += 1
        result.append((start, end, "quote"))
        cursor = end
    if text[cursor:].strip():
        result.append((cursor, len(text), "prose"))
    elif result:
        a, _, kind = result[-1]
        result[-1] = (a, len(text), kind)
    return result


def prose_paragraphs(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Intersect a prose run with original paragraph spans (not artificial breaks)."""
    return [
        (max(a, start), min(b, end))
        for a, b in paragraph_spans(text)
        if a < end and b > start and text[max(a, start) : min(b, end)].strip()
    ]
