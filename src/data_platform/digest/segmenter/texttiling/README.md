# TextTiling segmentation and heading generation

Split article text into source-aligned sections using lexical similarity or
sentence embeddings. Structure detection and heading generation are separate
steps that can be combined with the convenience pipeline.

## Setup and segmentation methods

Run `uv sync` from the repository root. Install optional TF-IDF/local-embedding
dependencies with `uv sync --extra offline`.

```python
from data_platform.digest.segmenter.texttiling import (
    segment_article, segment_article_semantic,
)

text = "Acme builds cars.\n\nAcme also makes batteries."
lexical = segment_article(text)  # local token counts; no credentials
weighted = segment_article(text, vectorizer="tfidf")  # optional scikit-learn
semantic = await segment_article_semantic(text)  # sentence embeddings
```

Counts and TF-IDF compare lexical windows; semantic segmentation compares sentence
embedding windows. Boundaries follow paragraphs, existing structure, and quotations.
Prose sections have a four-paragraph cap; paragraphs and quotation sections have no
size cap. English/Vietnamese sentence and quotation handling is heuristic.

With `embedding_model=None`, semantic segmentation selects
`openai:text-embedding-3-small` if `OPENAI_API_KEY` is nonblank, otherwise
`sentence-transformers:intfloat/multilingual-e5-small`. Explicit model identifiers
or configured Pydantic AI embedding models override selection. Local weights may
download on first use; `local_files_only=True` requires prepared weights, and
`local_model_path` can identify them. Provider errors do not trigger a silent local
fallback. `resolve_embedding_model()` reports selection without loading weights.

## Structure, headings, and Markdown

```python
from data_platform.digest.segmenter.texttiling import (
    detect_document_structure, generate_headings, render_markdown,
)

structure = await detect_document_structure(text)
sections = await segment_article_semantic(text, structure=structure)
headed = await generate_headings(sections, article_text=text, max_concurrency=4)
markdown_doc = render_markdown(headed)
```

`detect_document_structure` parses explicit Markdown locally and classifies eligible
unformatted heading candidates with a model (`decision_model="openai:gpt-6-luna"`).
Without an OpenAI key, unformatted classification is skipped. Use
`parse_document_structure` for a local-only pass. Segmentation without supplied
structure parses locally and does not make hidden heading-classification calls.

Heading generation defaults to `model="openai:gpt-6-luna"`, accepts configured model
instances, and uses article context to resolve references. Local embeddings do not
provide a local heading model. Structural headings are retained; generated headings
use the source language and preserve attribution and planned/completed status.
Semantic heading requirements are prompt-driven and need review.

`segment_with_headings(text, ...)` combines detection, semantic segmentation, and
heading generation, returning sections. `texttiling_segement(text, ...)` (existing
API spelling) additionally renders Markdown. Both accept `model`, `decision_model`,
`embedding_model`, `local_files_only`, `local_model_path`, and `max_concurrency`.
Use `await` in Jupyter or wrap async calls in `asyncio.run` for scripts.

## Output and limitations

Sections are immutable records with `text`, `start`, `end`, `kind`, optional
`heading`, `heading_level`, and `heading_is_markdown`. Offsets are half-open Python
character indices. For nonblank input, concatenating section text reconstructs the
original input. Blank input returns no sections. Kinds include prose, quote, title,
metadata, and heading. Attributed quotations stay intact and are excluded from
similarity fitting; unrelated narration remains separate.

Rendering inserts headings and normalizes heading structure; do not treat rendered
Markdown offsets as offsets into the original input. It does not invent a missing
article title: downstream claim extraction still requires a valid H1/H2 document.
`HeadingDetectionError`, `EmbeddingGenerationError`, and `HeadingGenerationError`
retain contextual indices and original causes. Failed async heading work cancels
and awaits remaining workers. No automatic file persistence occurs.

## Verification

From the repository root:

```sh
uv run python -m unittest discover -s tests -p 'test_texttiling_segmenter.py' -v
uv run python -m unittest discover -s tests -p 'test_semantic_segmenter.py' -v
uv run python -m unittest discover -s tests -p 'test_document_structure.py' -v
uv run python -m unittest discover -s tests -p 'test_quotations.py' -v
```

These use offline fixtures by default. See
[the notebook](../../../../../notebooks/article_segmentation.ipynb) and
[LLM segmentation](../llm_e2e/README.md) for the alternative approach.
