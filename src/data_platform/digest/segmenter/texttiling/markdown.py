"""Render source-preserving sections with document and section heading structure."""

import re
import string
from collections.abc import Sequence

from .models import Section


def _structural_heading(section: Section) -> str:
    text = section.text
    level = section.heading_level or (1 if section.kind == "title" else 2)
    if section.heading_is_markdown:
        # Preserve existing H2-H6 and the title; only later H1s need demotion.
        if section.kind == "heading" and level == 2:
            atx = re.search(r"(?m)^( {0,3})#(?=\s|$)", text)
            if atx:
                index = atx.end()
                return text[:index] + "#" + text[index:]
            return re.sub(r"(?m)^( {0,3})=+[ \t]*(?=\r?$)", r"\1---", text, count=1)
        return text
    # Rule-filtered candidates contain one nonblank line; preserve their wording.
    return re.sub(
        r"(?m)^([ \t]*)(\S)", lambda m: m[1] + "#" * level + " " + m[2], text, count=1
    )


def render_markdown(sections: Sequence[Section]) -> str:
    """Render existing structural headings and generated H2s only for body sections."""
    escapes = str.maketrans({char: "\\" + char for char in string.punctuation})
    parts = []
    before_title = any(s.kind == "title" for s in sections)
    suppress_next = False
    for section in sections:
        if section.kind in {"title", "heading"}:
            parts.append(_structural_heading(section))
            if section.kind == "title":
                before_title = False
                suppress_next = False
            else:
                suppress_next = True
            continue
        if section.kind == "metadata":
            parts.append(section.text)
            continue
        if section.heading is not None and not before_title and not suppress_next:
            if parts and not re.search(r"\r?\n[ \t]*\r?\n$", parts[-1]):
                parts.append("\n" if parts[-1].endswith("\n") else "\n\n")
            parts.append(f"## {section.heading.translate(escapes)}\n\n")
        parts.append(section.text)
        suppress_next = False
    return "".join(parts)
