# TextTiling segmentation and headings

Segmentation and heading generation are independent. In a notebook:

```python
from data_platform.digest.segmenter.texttiling import (
    detect_document_structure,
    segment_article_semantic,
    generate_headings,
    render_markdown,
)

structure = await detect_document_structure(markdown_doc)
sections = await segment_article_semantic(markdown_doc, structure=structure)
headed = await generate_headings(sections, article_text=markdown_doc, max_concurrency=4)
segmented_doc = render_markdown(headed)
```

Open [the notebook](../../../../../notebooks/article_segmentation.ipynb) for article loading, section inspection,
backend comparisons, headings, display, and saving. In scripts use `asyncio.run`.

### Document titles and existing headings

`detect_document_structure(markdown_doc, decision_model="openai:gpt-6-luna")`
recognizes explicit Markdown structure locally, then classifies rule-filtered
unformatted heading candidates with a Pydantic-validated `is_heading: StrictBool`
response. It accepts configured model instances and `max_concurrency=4`, with two
output-validation retries. A candidate is a standalone, single-line paragraph
containing at most one sentence, up to 40 lexical tokens and 240 characters;
short labels without punctuation are allowed. Metadata, bylines, lists, quotations,
code, and standalone URLs are excluded. Capitalization alone does not confirm a
heading. The classifier sees the document title and neighboring paragraphs.

Without an OpenAI key, explicit Markdown headings and metadata still work;
unformatted candidates remain unchanged and `classification_skipped` is true.
An attempted classifier failure raises `HeadingDetectionError.candidate_index`
with the original cause. Configured model instances use their own credentials.
Use `parse_document_structure` for the local-only pass.

The returned `DocumentStructure` contains source-aligned `blocks` and `candidates`;
candidates expose `is_heading` as true, false, or `None` when unclassified. Pass
`structure=structure` to either segmentation API to reuse these decisions. Without
precomputed structure, segmenters only perform the local pass: they make no hidden
heading-classification calls. A structure from different source text is rejected.

The first H1 is the document title. Additional H1s become H2s; existing H2–H6
headings retain their levels. If no H1 exists, a confirmed heading at the beginning
of the document becomes the H1 title. The title and adjacent publication metadata
are emitted before generated subtitles. Unformatted section headings become H2s
without changing their wording. Heading detection is heuristic plus model judgment;
review the candidate decisions shown in the notebook when accuracy matters.

Structural sections use `kind="title"`, `"metadata"`, or `"heading"`, with
`heading_level` and `heading_is_markdown` metadata. Their `text` and source offsets
remain unchanged; heading normalization happens only in `render_markdown`.
Structural blocks are excluded from embeddings and paragraph caps, and act as
mandatory section boundaries. Heading generation skips structural blocks and the
first content section following each existing heading, preventing duplicate
subtitles. Later subdivisions can receive generated H2 headings. An article's
existing title is never given a generated heading above it.

### Automatic embedding selection

With `embedding_model=None`, semantic segmentation uses
`openai:text-embedding-3-small` when `OPENAI_API_KEY` is nonblank; otherwise it tries
`sentence-transformers:intfloat/multilingual-e5-small` locally. Explicit model
names or configured Pydantic AI `EmbeddingModel` instances override selection.
`resolve_embedding_model()` reports the selection without loading a model;
actual embedding selection is also logged at INFO level.

The library reads process credentials. The example notebook loads `.env` first.
Authentication, quota, timeout, or other OpenAI errors do not cause a silent local
fallback. `EmbeddingGenerationError` reports the zero-based batch/sentence index
and preserves the cause. No partial segmentation is returned.

Install the offline alternatives with:

```sh
uv sync --extra offline
```

Local E5 runs on CPU with normalized vectors and batches of 32. Its weights are
cached and may download on first use; loaded models are reused. For prepared
weights only, use:

```python
sections = await segment_article_semantic(
    markdown_doc,
    embedding_model="sentence-transformers:intfloat/multilingual-e5-small",
    local_files_only=True,
    # local_model_path="/path/to/prepared/e5-model",  # optional
)
```

The local model uses the E5 `query: ` prefix for symmetric similarity. Token budgets
include that prefix and special tokens. Missing dependencies or weights produce
an actionable error. Local embedding fallback does **not** provide a local heading
model: the default heading generator still requires an OpenAI key. The notebook
skips heading generation without that key and can still display/save the sections.

### Offline lexical alternatives

```python
from data_platform.digest.segmenter.texttiling import segment_article

sections = segment_article(markdown_doc)  # existing count-based TextTiling
sections = segment_article(markdown_doc, vectorizer="tfidf")
```

Counts remain the compatibility default. TF-IDF uses scikit-learn word unigrams
and bigrams, sublinear term frequency, smoothed IDF, and L2 normalization. Weights
are fitted across article prose blocks and reused for adjacent comparison windows.
Diacritics are preserved, with no English-only stopword list. TF-IDF is a lexical
alternative, not a learned semantic representation.

### Boundaries and source preservation

Both segmentation APIs return immutable `Section` records with `text`, `start`,
`end`, `kind` (`prose`, `quote`, `title`, `metadata`, or `heading`), and optional `heading`. Offsets are Python
character indices, with an exclusive end. Concatenating section text reconstructs
nonblank input exactly. Blank input returns `[]`.

Blank lines separate paragraphs. Every independently attributed quotation gets its
own `quote` section, even when the same person speaks again or multiple quotations
occur in one paragraph. Adjacent reporting clauses, including speaker names and
titles, stay with their quotation. An utterance interrupted only by attribution,
such as `“We will invest,” she said, “next year.”`, remains one quotation. Separate
completed utterances never merge just because the speaker or topic is the same.

The shared deterministic detector supports common English/Vietnamese reporting
clauses, straight/curly quotation marks, guillemets, nested quotations, standalone
quotes, and Markdown blockquotes. Multi-paragraph quotations remain intact.
Ordinary inline quoted terms, code, and unmatched delimiters remain prose;
unrelated surrounding narration is kept outside the quote sections. Detection is
heuristic rather than a full grammatical or speaker-identity parser.

Attributed quotations are excluded from heading candidates, embeddings, and TF-IDF
vocabulary fitting. Comparison windows never cross quotes. Titles and existing
headings remain higher-priority structural boundaries. Prose fragments retain their
original paragraph boundaries for the four-paragraph cap. Source slices and offsets
remain exact; rendering adds blank lines where necessary for generated H2 headings
between sections originally occurring within one paragraph.
Paragraphs and quote sections have no size cap. Prose sections contain at most
four paragraphs, after topic boundaries have been selected.

Lexical TextTiling uses 20-token blocks and up to four blocks per neighboring
window. Semantic TextTiling uses sentence embeddings and windows of up to three
sentences on each side. Both use cosine similarity, three-gap smoothing, depth
peaks, and paragraph-aligned boundaries. Sentence splitting is heuristic, protecting
common English/Vietnamese abbreviations, initials, decimals, and URLs; it is not a
full language parser. Neither method guarantees ideal topic boundaries.

OpenAI embedding batches contain at most 64 inputs and 64,000 tokens. Oversized
sentences are split at Unicode-safe character boundaries within model token
budgets, then combined by token-weighted vector averaging. Text is never silently
truncated. Blank input, quote-only input, and single-paragraph prose runs require
no embeddings or local model loading.

### Heading rules and parallel requests

`generate_headings` accepts existing sections without rerunning segmentation.
Pass `article_text` to provide complete context, especially when passing a subset
of sections; by default it reconstructs context from the supplied sections.
The default heading model is `openai:gpt-6-luna`; `model` also accepts configured
Pydantic AI models.

Headings are instructed to use the article's language and a complete sentence with
an explicit subject, verb, and object. Ordinary headings summarize the first
substantive claim and preserve completed/planned/expected status. Quote headings
name the explicitly attributed speaker and paraphrase the first claim in neutral
indirect speech. Entity names replace pronouns; named people retain honorifics and
full titles supplied in the article. Missing details are never invented. Unnamed
speakers are described as unnamed. Structural metadata sections do not receive
generated headings. Context resolves references rather than replacing the target claim.

Structured output validates nonempty, single-line headings with two output retries.
Grammar and factual rules are prompt requirements and need review; they are not
mechanically guaranteed. `max_concurrency=4` bounds simultaneous section requests;
use one for sequential generation. Results preserve input order and inputs are
not mutated. On failure, remaining workers are cancelled and awaited;
`HeadingGenerationError.section_index` identifies the failed section and its cause.
Caller cancellation propagates. Empty inputs make no model requests.

`render_markdown` inserts escaped `##` headings without model requests; unheaded
sections remain unchanged. `segment_with_headings` and `texttiling_segement` remain
convenience wrappers, but now perform semantic embedding requests before heading
requests. Both detect document structure first, accept `decision_model`, and forward
`embedding_model`, `local_files_only`, `local_model_path`, and `max_concurrency`,
while supplying full article context. The `model` argument still
selects the heading model.

### Verification

The default test suite uses deterministic embedding and heading fixtures. Optional
live checks on both article fixtures:

```sh
RUN_LIVE_HEADINGS=1 uv run python -m unittest discover -s tests -p 'test_texttiling_segmenter.py' -v
RUN_LOCAL_EMBEDDINGS=1 uv run --extra offline python -m unittest discover -s tests -p 'test_semantic_segmenter.py' -v
```

Configure credentials in the process environment for hosted checks. Set
`HEADING_MODEL` to override the live heading model. Set `LOCAL_FILES_ONLY=1` for
the local smoke test to require cached weights. Live tests report boundaries and
headings for review rather than asserting a fixed section count. No crawler
integration is added.
