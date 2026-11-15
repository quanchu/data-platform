"""Shared section data for segmentation and heading generation."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Section:
    """A contiguous source slice; end is exclusive, in Python character offsets."""

    text: str
    start: int
    end: int
    kind: Literal["prose", "quote", "title", "metadata", "heading"]
    heading: str | None = None
    heading_level: int | None = None
    heading_is_markdown: bool = False
