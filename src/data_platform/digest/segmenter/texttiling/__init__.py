"""Independent local segmentation and optional asynchronous heading generation."""

from .document_structure import (
    DocumentStructure,
    HeadingCandidate,
    StructureBlock,
    parse_document_structure,
)
from .markdown import render_markdown
from .models import Section
from .texttiling_segmenter import segment_article

__all__ = [
    "DocumentStructure",
    "EmbeddingGenerationError",
    "HeadingCandidate",
    "HeadingDetectionError",
    "HeadingGenerationError",
    "Section",
    "StructureBlock",
    "detect_document_structure",
    "generate_headings",
    "parse_document_structure",
    "render_markdown",
    "resolve_embedding_model",
    "segment_article",
    "segment_article_semantic",
    "segment_with_headings",
    "texttiling_segement",
]


def __getattr__(name: str):
    if name in {"detect_document_structure", "HeadingDetectionError"}:
        from . import heading_detection

        return getattr(heading_detection, name)
    if name in {
        "segment_article_semantic",
        "resolve_embedding_model",
        "EmbeddingGenerationError",
    }:
        from . import semantic_segmenter

        return getattr(semantic_segmenter, name)
    if name in {"generate_headings", "HeadingGenerationError"}:
        from . import heading_generator

        return getattr(heading_generator, name)
    if name in {"segment_with_headings", "texttiling_segement"}:
        from . import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
