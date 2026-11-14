# Data platform

Install the project and its dependencies in editable mode from the repository root:

```sh
uv sync
uv run jupyter notebook
```

Open `notebooks/article_segmentation.ipynb` and select the project's `.venv`
Python interpreter/kernel. Imports work from `notebooks/` or any other working
directory in that environment, without changing `sys.path`:

```python
from data_platform.digest.segmenter.texttiling import segment_article

sections = segment_article("Đoạn văn đầu tiên.\n\nĐoạn văn thứ hai.")
```

If your notebook uses a different Python environment, install the project into
that kernel from a notebook under `notebooks/`, then restart the kernel:

```python
%pip install -e ..
```

That environment must use Python 3.14 or later. The distribution name is
`data-platform`; the Python import name is `data_platform`. Editable installs
pick up source edits; restart the kernel to reload modules already imported.

```text
src/data_platform/
    digest/          # segmentation, claim extraction, tagging
    ingest/          # article crawling
notebooks/           # notebooks consuming the installed package
tests/               # package and crawler tests
```

The previous top-level `digest` and `ingest` imports now live under
`data_platform.digest` and `data_platform.ingest`.

## Vingroup news crawler

Run from the repository root with Python 3.14 and uv:

```sh
uv sync
uv run python -m data_platform.ingest.crawler.vingroup_news discover --output .data/vingroup_links.jsonl
uv run python -m data_platform.ingest.crawler.vingroup_news fetch --input .data/vingroup_links.jsonl --output .data/vingroup_articles.jsonl
```

`discover` follows all pagination links in the Vietnamese “all news” listing. It saves one UTF-8 JSON object per line, with `url`, `title`, `published_date` (ISO date), `category`, `thumbnail_url`, and `source_listing_url`. Metadata is extracted only from `.itemNews`; unavailable fields are `null`. It does not request article bodies.

`fetch` reads the saved file and extracts `.newsWrap.stagger-up` from each article. Output preserves the metadata and adds `content_html` (the complete wrapper), `content_text` (with paragraph breaks), and `fetched_at` (UTC). URLs with successfully saved content are skipped on subsequent runs. Failed requests are logged to `.data/vingroup_articles.errors.jsonl` by default; override with `--errors PATH`. Re-running against the same input retries failures. The error log is historical and may include URLs that subsequently succeed.

Both commands append records and deduplicate by URL. Discovery starts from page one each time and skips URLs already in its output. Keep the manifest if you want to fetch content later without repeating discovery. Use separate output paths for a fresh snapshot.

For a small run, add `--max-pages 1` to discovery and `--max-articles 2` to fetch. Both support `--timeout 30`, `--delay 1`, and `--retries 3`. The article limit counts attempted, previously unsaved URLs. Transient failures are retried; an extraction failure is reported instead of saving empty content. Commands return a nonzero status on failure and retain already saved records.

## HTTP fetching

Both commands use `requests.Session` to reuse connections and cookies and follow redirects. No browser installation is required. Browser flags (`--transport`, `--browser-channel`, and `--headed`) have been removed.

Connection errors, timeouts, HTTP 429, and HTTP 5xx responses receive bounded retries. HTTP 401/403 responses fail immediately with an access error; a Cloudflare challenge is identified when the response headers indicate one. Changing the HTTP client does not guarantee access through a site challenge. Blocked responses are never saved as article content.

## Tests

```sh
uv run python -m unittest discover -s tests -v
```

## Article sections and headings

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

Open `notebooks/article_segmentation.ipynb` for article loading, section inspection,
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

## End-to-end LLM segmentation (Python API)

TextTiling lives in `data_platform.digest.segmenter.texttiling`. Existing public
imports from `data_platform.digest.segmenter` remain supported through lazy
exports. The independent `llm_e2e` pipeline asks one model to choose section
boundaries and write headings together:

```python
from data_platform.digest.segmenter.llm_e2e import llm_segment

segmented_doc = await llm_segment(
    markdown_doc,
    model="openai:gpt-6-astra",
    retries=2,
)
```

Like `texttiling_segement`, this async function accepts Markdown and returns
Markdown. Use `await` directly in Jupyter, or `asyncio.run` in a synchronous
script. Model arguments also accept configured Pydantic AI models.
The library uses caller-provided credentials and performs no file operations.

For inspection, import `split_blocks`, `generate_block_headings`, and
`reconstruct_document` from the same package. Split the document, await headings
for all blocks, then reconstruct. Complete quotations and their attribution stay
together, including multi-paragraph quotations; independent utterances remain
separate even for the same speaker.

The structured output contains one string per block: an empty string continues a
section, while a nonempty string supplies a level-two heading. Validation checks
count, empty-block handling, the `## ` prefix, non-whitespace heading text, absence
of CR/LF, and exclusion of the reserved `<SECTION_BOUNDARY>` marker. Literal
markers in source text are preserved. Failed validation sends corrective feedback
through Pydantic AI, with two retries by default. Provider errors and exhausted
retries propagate to the caller.

Section placement and wording are model-driven. The prompt enforces quotation
sections, title placement, existing-heading recognition, coherent prose groups of
at most four paragraphs, and the article-grounded sentence rules described above.
These semantic rules are not mechanically validated.

Reconstruction only prepends headings: it never asks the model to rewrite the
article. Text within blocks is preserved, but separators are normalized to blank
LF lines. Empty blocks from consecutive or trailing blank-line pairs are retained.
Unlike TextTiling's renderer, this pipeline preserves unformatted source headings
without converting them into Markdown headings.

See [the LLM notebook](notebooks/llm_e2e_segmentation.ipynb) for the one-call API,
intermediate results, credential loading, and optional UTF-8 saving.
