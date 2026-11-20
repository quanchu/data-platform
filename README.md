# Data platform

Python tools for article ingestion, segmentation, extraction, entity tagging, and
human-reviewed claim storage. The workflows remain independent Python APIs;
notebooks demonstrate how to use them.

## Setup

From the repository root:

```sh
uv sync
uv run jupyter notebook
```

Use Python 3.14 or later and select the project's `.venv` kernel. The editable
installation supports `data_platform` imports from notebooks or other working
directories. Restart the kernel after changing imported modules. For a different
kernel, install the project with `%pip install -e ..` from `notebooks/`.

```python
from data_platform.digest.segmenter.texttiling import segment_article

sections = segment_article("Đoạn văn đầu tiên.\n\nĐoạn văn thứ hai.")
```

Notebooks use pandas DataFrames for sections, claims, entities, and comparisons.
Shared setup, DataFrame conversion, sentence rows, and JSON export helpers live in
[`notebooks/utils.py`](notebooks/utils.py). Open the notebooks in Jupyter so their
kernels run from the notebook directory and can import the adjacent `utils` module.
Each follows setup, configuration, processing, review, and optional export. Model
calls are opt-in (`run_live=True`, or the explicit segmentation model settings);
saved outputs are cleared so a fresh kernel starts cleanly.

Libraries read process credentials; notebooks load `.env` without overriding the
existing environment. See `.env.example`. Model-based workflows accept configured
model objects for injection. Provider errors do not silently switch backends.

Optional TF-IDF and local sentence embeddings require:

```sh
uv sync --extra offline
```

## Processing methods

| Method | Guide | Notebook |
| --- | --- | --- |
| Discover and fetch news into resumable JSONL | [Vingroup crawler](src/data_platform/ingest/crawler/README.md) | — |
| Lexical or semantic segmentation and headings | [TextTiling](src/data_platform/digest/segmenter/texttiling/README.md) | [Article segmentation](notebooks/article_segmentation.ipynb) |
| Model-selected boundaries and headings | [LLM segmentation](src/data_platform/digest/segmenter/llm_e2e/README.md) | [LLM segmentation](notebooks/llm_e2e_segmentation.ipynb) |
| Verifiable factual claim extraction with checks | [Claimify](src/data_platform/digest/claimify/README.md) | [Claimify](notebooks/claimify.ipynb) |
| Reference resolution and relation tuples | [OpenIE](src/data_platform/digest/openie/README.md) | [OpenIE](notebooks/openie.ipynb) |
| Source-aligned entity tagging | [GLiNER](src/data_platform/digest/gliner/README.md) | [Entity tagging](notebooks/gliner_entity_tagging.ipynb) |
| Manual approval, verified retrieval, and comparison | [Claim store](src/data_platform/digest/claim_store/README.md) | [Claim store](notebooks/claim_store.ipynb) |

Crawling produces raw article content. Claim extraction requires prepared Markdown
with one H1 title and H2 sections; segmentation output must satisfy that contract
before extraction. Claimify filters subjective assertions; OpenIE preserves
attributed statements and opinions. Extracted assertions are not independently
verified facts. Claim-store approval comes only from manually placing JSON files
in `verified/`.

## Repository layout

```text
src/data_platform/
    digest/          # text processing, extraction, tagging, and claim storage
    ingest/          # crawling and persistence
notebooks/           # workflows using the installed package
tests/               # offline contracts and opt-in model smoke tests
.data/               # tracked article examples and local generated artifacts
```

Private text utilities and async workers support the separate digest workflows.
Existing public imports, including lazy `data_platform.digest.segmenter` exports
and `texttiling_segement`, remain supported. The old top-level `digest` and `ingest`
packages live under `data_platform`.

## Verification

```sh
uv run python -m unittest discover -s tests -v
```

The default suite uses injected models and makes no provider calls or model
downloads. Component guides describe opt-in live checks. Offline tests cover
structure, provenance, persistence, concurrency, and compatibility; semantic model
quality still requires review. Existing JSON claims and SQLite caches need no
migration after the internal refactor.

## Live processing demo

```sh
uv sync --extra demo --extra offline
uv run python -m data_platform.demo
```

Open `http://127.0.0.1:8765`. The local application shows real processing events,
source highlights, similarity/depth charts, and task results. Start with the
**list-aware** example and lexical segmentation, which needs no provider calls.
See the [demo walkthrough](src/data_platform/demo/README.md) for model setup,
extraction handoffs, and manual claim review.

## Internal organization

Shared Markdown tokenization and source-mapped block trees live in `digest/_text`.
Each processing component owns its orchestration and `prompts.py`, including the
questions attached to typed decision fields. Private event observers provide demo
progress without changing public library signatures. Ordinary library calls have
no observer and do not depend on the web application.

Lists are now structural segmentation units. Nested items and continuation
paragraphs stay together; their contents do not affect surrounding prose scores.
The added `list` section kind is the intentional compatibility extension in this
readability pass. Existing public imports and stored claim formats are preserved.
