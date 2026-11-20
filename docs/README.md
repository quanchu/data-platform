# Documentation

Data platform provides independent Python workflows for ingesting articles,
segmenting Markdown, extracting claims and relations, tagging entities, and
maintaining a human-reviewed claim store.

## Start here

- [Project setup and repository layout](../README.md)
- [Live demo quickstart](demo.md)
- [Notebook workflows](../notebooks/) — pandas-based review and optional export;
  shared helpers live in [utils.py](../notebooks/utils.py).

## Workflow guides

| Workflow | Guide |
| --- | --- |
| Article ingestion | [Vingroup crawler](../src/data_platform/ingest/crawler/README.md) |
| Lexical and semantic segmentation | [TextTiling](../src/data_platform/digest/segmenter/texttiling/README.md) |
| End-to-end model segmentation | [LLM segmentation](../src/data_platform/digest/segmenter/llm_e2e/README.md) |
| Factual claim extraction | [Claimify](../src/data_platform/digest/claimify/README.md) |
| Reference resolution and relation extraction | [OpenIE](../src/data_platform/digest/openie/README.md) |
| Entity tagging | [GLiNER](../src/data_platform/digest/gliner/README.md) |
| Claim approval, retrieval, and comparison | [Claim store](../src/data_platform/digest/claim_store/README.md) |
| Demo controls, APIs, and provenance | [Live demo guide](../src/data_platform/demo/README.md) |

Component guides stay beside their implementations and document the Python APIs.
This folder is the entry point for navigating them.

## Data flow

Articles in `.data/` can be segmented, then passed to Claimify or OpenIE. Extraction
requires one H1 title and nonempty H2 subtitles. Recognized publication-date and
author metadata can remain between the title and first section. Entity tagging
can operate on the source directly.

Claims retain source sentence offsets. Successful extraction means the assertions
passed checks against the source; it does not establish their real-world truth.
Claim-store records become verified only through manual approval.

## Checks

From the repository root:

```sh
uv run python -m unittest discover -s tests -v
node tests/test_demo_frontend.cjs
node tests/test_demo_lifecycle.cjs
```

The Python test suite uses injected models. Live provider calls and local model
inference need separate checks using the configuration in each component guide.
