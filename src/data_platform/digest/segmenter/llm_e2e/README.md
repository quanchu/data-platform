# End-to-end LLM segmentation

Ask one model to select section boundaries and write headings for article blocks,
then prepend those headings locally. This is independent of
[TextTiling](../texttiling/README.md) and uses no embedding stage.

## Setup and usage

Run `uv sync` from the repository root and configure `OPENAI_API_KEY` for the
default model. Configured Pydantic AI model instances are also accepted.

```python
from data_platform.digest.segmenter.llm_e2e import llm_segment

segmented_doc = await llm_segment(
    "# Report\n\nAcme builds cars.\n\nAcme sells batteries.",
    model="openai:gpt-6-astra",
    retries=2,
)
```

Input and output are Markdown strings. Use `await` in Jupyter or an async function;
use `asyncio.run` in scripts. Credentials and file operations belong to the caller.

## Processing and intermediate API

1. `split_blocks(markdown_doc)` splits on blank lines while keeping complete
   quotations and their attribution together.
2. `await generate_block_headings(blocks, model=..., retries=2)` requests all headings
   together. An empty string continues a section; a nonempty `## ` heading starts one.
3. `reconstruct_document(blocks, headings)` validates and prepends headings locally.

The model is instructed to preserve existing headings/title placement, keep coherent
prose groups within four paragraphs, and give separate quotations their own sections.
Generated headings preserve source language, entities, attribution, and modality.
These semantic requirements are prompt-driven, not mechanically guaranteed.

## Preservation, validation, and failures

Source text inside blocks is preserved; separators are normalized to blank LF lines.
This API does not return source offsets or rewrite article prose. Existing
unformatted headings are preserved as text, rather than converted to Markdown.
It does not guarantee the strict H1/H2 contract required by claim extractors; validate
or prepare that structure before downstream extraction.

Validation checks heading count, empty-block handling, the `## ` prefix, nonempty
heading text, no embedded line breaks, and no reserved `<SECTION_BOUNDARY>` marker
in generated headings. Literal source markers are preserved. Failed model output
receives bounded corrective retries; exhausted retries and provider errors propagate.
`retries` must be a nonnegative integer. Blank blocks require no model request.
The full block set goes to the model, so large articles remain subject to its context
limit; there is no automatic document chunking or backend fallback.

## Verification

From the repository root:

```sh
uv run python -m unittest discover -s tests -p 'test_llm_e2e.py' -v
```

Tests use deterministic model fixtures. See
[the notebook](../../../../../notebooks/llm_e2e_segmentation.ipynb) for intermediate
inspection and optional saving.
