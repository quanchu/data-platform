"""Compatibility helpers combining the two independent components."""

from pydantic_ai.embeddings import EmbeddingModel
from pydantic_ai.models import Model

from .heading_detection import detect_document_structure
from .heading_generator import (
    DEFAULT_MODEL,
    Heading,
    HeadingGenerationError,
    generate_headings,
)
from .markdown import render_markdown
from .models import Section
from .semantic_segmenter import segment_article_semantic

__all__ = [
    "Heading",
    "HeadingGenerationError",
    "generate_headings",
    "segment_with_headings",
    "texttiling_segement",
]


async def segment_with_headings(
    text: str,
    *,
    model: str | Model = DEFAULT_MODEL,
    embedding_model: str | EmbeddingModel | None = None,
    max_concurrency: int = 4,
    local_files_only: bool = False,
    local_model_path: str | None = None,
    decision_model: str | Model = DEFAULT_MODEL,
) -> list[Section]:
    """Embed and segment, then generate article-grounded headings concurrently."""
    structure = await detect_document_structure(
        text, decision_model=decision_model, max_concurrency=max_concurrency
    )
    sections = await segment_article_semantic(
        text,
        structure=structure,
        embedding_model=embedding_model,
        local_files_only=local_files_only,
        local_model_path=local_model_path,
    )
    return await generate_headings(
        sections, model=model, article_text=text, max_concurrency=max_concurrency
    )


async def texttiling_segement(
    markdown_doc: str,
    *,
    model: str | Model = DEFAULT_MODEL,
    embedding_model: str | EmbeddingModel | None = None,
    max_concurrency: int = 4,
    local_files_only: bool = False,
    local_model_path: str | None = None,
    decision_model: str | Model = DEFAULT_MODEL,
) -> str:
    """Return Markdown using semantic segmentation and generated section headings."""
    return render_markdown(
        await segment_with_headings(
            markdown_doc,
            model=model,
            embedding_model=embedding_model,
            max_concurrency=max_concurrency,
            local_files_only=local_files_only,
            local_model_path=local_model_path,
            decision_model=decision_model,
        )
    )
