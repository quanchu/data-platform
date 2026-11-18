"""Sentence-embedding TextTiling with automatic hosted/local backend selection."""

import asyncio
import logging
import math
import os
from functools import lru_cache
from threading import Lock
from typing import TYPE_CHECKING

from ..._text.quotations import body_spans, prose_paragraphs
from ..._text.sentences import sentence_spans as _sentences
from .document_structure import (
    DocumentStructure,
    document_parts,
    parse_document_structure,
)
from .models import Section
from .texttiling_segmenter import _segment, _select_boundaries

if TYPE_CHECKING:
    from pydantic_ai.embeddings import EmbeddingModel

OPENAI_EMBEDDING_MODEL = "openai:text-embedding-3-small"
LOCAL_EMBEDDING_MODEL = "sentence-transformers:intfloat/multilingual-e5-small"
_LOG = logging.getLogger(__name__)
_LOAD_LOCK = Lock()


class EmbeddingGenerationError(RuntimeError):
    """Embedding failure with zero-based batch and sentence context."""

    def __init__(self, batch_index: int, sentence_index: int, detail: str):
        self.batch_index = batch_index
        self.sentence_index = sentence_index
        super().__init__(
            f"Embedding batch {batch_index}, sentence {sentence_index}: {detail}"
        )


def resolve_embedding_model(embedding_model: str | EmbeddingModel | None = None):
    """Resolve automatic selection without loading weights or accessing the network."""
    if embedding_model is not None:
        return embedding_model
    return (
        OPENAI_EMBEDDING_MODEL
        if os.getenv("OPENAI_API_KEY", "").strip()
        else LOCAL_EMBEDDING_MODEL
    )


@lru_cache(maxsize=2)
def _cached_local_model(name: str, local_files_only: bool, use_e5_prefix: bool):
    try:
        from pydantic_ai.embeddings.sentence_transformers import (
            SentenceTransformerEmbeddingModel,
        )
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("Local embeddings require: uv sync --extra offline") from exc
    try:
        encoder = SentenceTransformer(
            name, device="cpu", local_files_only=local_files_only
        )
    except OSError as exc:
        raise RuntimeError(
            f"Cannot load local embedding model {name!r}. Prepare its weights with network access "
            "or supply local_model_path; set local_files_only=False to permit the first download."
        ) from exc
    # E5's symmetric similarity task requires this prefix, including local copies.
    if use_e5_prefix:
        encoder.prompts["query"] = "query: "
    encoder.default_prompt_name = None

    tokenizer = encoder.tokenizer

    class UntruncatedEmbeddingModel(SentenceTransformerEmbeddingModel):
        async def count_tokens(self, text: str) -> int:
            # SentenceTransformer.tokenize truncates; budget using its raw tokenizer.
            def count():
                return len(
                    tokenizer(
                        text, add_special_tokens=True, truncation=False, verbose=False
                    )["input_ids"]
                )

            return await asyncio.to_thread(count)

    return UntruncatedEmbeddingModel(
        encoder,
        settings={
            "sentence_transformers_device": "cpu",
            "sentence_transformers_normalize_embeddings": True,
            "sentence_transformers_batch_size": 32,
        },
    )


def _load_local_model(name: str, local_files_only: bool, use_e5_prefix: bool):
    with _LOAD_LOCK:
        return _cached_local_model(name, local_files_only, use_e5_prefix)


async def _make_embedder(selected, local_files_only, local_model_path):
    from pydantic_ai import Embedder

    prefix = ""
    if isinstance(selected, str) and selected.startswith("sentence-transformers:"):
        use_e5_prefix = "multilingual-e5" in selected
        name = local_model_path or selected.split(":", 1)[1]
        selected = await asyncio.to_thread(
            _load_local_model, str(name), local_files_only, use_e5_prefix
        )
        prefix = "query: " if use_e5_prefix else ""
    return Embedder(selected), prefix


def _normalize(vector):
    if not vector or not all(math.isfinite(value) for value in vector):
        raise ValueError("Embedding vectors must be nonempty and finite")
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("Embedding vectors must have a finite, nonzero norm")
    return [value / norm for value in vector]


async def _chunks(text, count_tokens, limit, prefix=""):
    """Split at Python character boundaries without dropping any source characters."""
    remaining = text
    while remaining:
        total = await count_tokens(prefix + remaining)
        if total <= limit:
            if remaining.strip():
                yield remaining, max(1, total)
            return
        low, high = 1, len(remaining)
        best = 0
        while low <= high:
            middle = (low + high) // 2
            if await count_tokens(prefix + remaining[:middle]) <= limit:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        if not best:
            raise ValueError("Embedding token budget cannot accommodate a character")
        chunk, remaining = remaining[:best], remaining[best:]
        if chunk.strip():
            yield chunk, max(1, await count_tokens(prefix + chunk))


async def _embed_sentences(
    text, sentences, selected, local_files_only, local_model_path
):
    batch_index = sentence_index = 0
    try:
        embedder, prefix = await _make_embedder(
            selected, local_files_only, local_model_path
        )
        limit = min(8000, await embedder.max_input_tokens() or 8000)
        pieces = []
        for sentence_index, (start, end) in enumerate(sentences):
            async for chunk, tokens in _chunks(
                text[start:end].strip(), embedder.count_tokens, limit, prefix
            ):
                pieces.append((sentence_index, chunk, tokens))
        sums = [None] * len(sentences)
        dimension = None
        cursor = 0
        while cursor < len(pieces):
            batch = []
            total = 0
            while cursor < len(pieces) and len(batch) < 64:
                item = pieces[cursor]
                if batch and total + item[2] > 64000:
                    break
                batch.append(item)
                total += item[2]
                cursor += 1
            sentence_index = batch[0][0]
            response = await embedder.embed(
                [item[1] for item in batch], input_type="query"
            )
            if len(response.embeddings) != len(batch):
                raise ValueError("Embedding count does not match input count")
            for (index, _, weight), vector in zip(
                batch, response.embeddings, strict=True
            ):
                sentence_index = index
                vector = _normalize(vector)
                if dimension is None:
                    dimension = len(vector)
                if len(vector) != dimension:
                    raise ValueError("Embedding dimensions are inconsistent")
                if sums[index] is None:
                    sums[index] = [0.0] * dimension
                for j, value in enumerate(vector):
                    sums[index][j] += weight * value
            batch_index += 1
        return [_normalize(vector) for vector in sums]
    except Exception as exc:
        raise EmbeddingGenerationError(batch_index, sentence_index, str(exc)) from exc


def _semantic_boundaries(vectors, sentences, paragraphs):
    scores = []
    for gap in range(1, len(vectors)):

        def mean(window):
            values = [sum(column) / len(window) for column in zip(*window, strict=True)]
            return _normalize(values) if any(values) else values

        left = mean(vectors[max(0, gap - 3) : gap])
        right = mean(vectors[gap : gap + 3])
        scores.append(sum(a * b for a, b in zip(left, right, strict=True)))
    return _select_boundaries(scores, [start for start, _ in sentences[1:]], paragraphs)


async def segment_article_semantic(
    text: str,
    *,
    embedding_model: str | EmbeddingModel | None = None,
    local_files_only: bool = False,
    local_model_path: str | None = None,
    structure: DocumentStructure | None = None,
) -> list[Section]:
    """Segment using sentence meaning; use OpenAI when a key exists, otherwise local E5.

    No silent fallback on provider failures. Local weights may download on first
    use unless local_files_only=True. Single-paragraph runs need no embeddings.
    """
    structure = structure if structure is not None else parse_document_structure(text)
    runs = []
    for begin, finish, block in document_parts(text, structure):
        if block is not None:
            continue
        body = text[begin:finish]
        for start, end, kind in body_spans(body):
            if kind == "quote":
                continue
            paragraphs = [
                (a + begin, b + begin) for a, b in prose_paragraphs(body, start, end)
            ]
            if len(paragraphs) > 1:
                runs.append(paragraphs)
    if not runs:
        return _segment(text, lambda *_: [], structure)
    selected = resolve_embedding_model(embedding_model)
    _LOG.info(
        "Embedding backend: %s",
        selected if isinstance(selected, str) else selected.model_name,
    )
    sentences = []
    offsets = []
    for run in runs:
        start = len(sentences)
        for a, b in run:
            sentences.extend(_sentences(text, a, b))
        offsets.append((start, len(sentences)))
    vectors = await _embed_sentences(
        text, sentences, selected, local_files_only, local_model_path
    )
    boundaries = {}
    for run, (a, b) in zip(runs, offsets, strict=True):
        boundaries[run[0][0]] = _semantic_boundaries(vectors[a:b], sentences[a:b], run)
    return _segment(text, lambda _, spans: boundaries.get(spans[0][0], []), structure)
