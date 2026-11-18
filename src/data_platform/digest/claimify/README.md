# Claimify

Extract standalone, verifiable factual claims from segmented Markdown. This method
uses separate decision and generation models, with semantic checks and bounded repairs.

## Setup and usage

From the repository root, run `uv sync`. The defaults require `TYPESAFE_API_KEY`
and `OPENAI_API_KEY` in the process environment. Configured Pydantic AI model
instances can supply their own provider configuration.

```python
from data_platform.digest.claimify import extract_claims

result = await extract_claims(
    "# Report\n\n## Factory\n\nAcme plans two factories.",
    question=None,
    decision_model="typesafe:jev-latest",
    generation_model="openai:gpt-6-luna",
    max_concurrency=4,
)
for claim in result.claims:
    print(claim.text, claim.start, claim.end)
```

Use `await` in Jupyter or an async function; use `asyncio.run` in synchronous scripts.
The library does not load `.env` files or persist results.

## Input and processing

Input requires exactly one nonempty H1 title, followed by nonempty H2 sections.
All body content belongs beneath an H2. Blank input returns an empty result without
model calls. Code, HTML blocks, and tables are excluded as extraction targets but
remain in the original section context. H3–H6 remain within their enclosing H2.

Each sentence receives the title, subtitle, full original section, and optional
question. Context does not cross section boundaries. The pipeline:

1. Select factual content and filter mixed factual/subjective sentences.
2. Assess ambiguity; reject references that cannot be resolved from context.
3. Clarify references and expand names supported by the source.
4. Generate standalone claims and check each claim plus set-level coverage.

The decision model handles semantic judgments; the generation model writes text.
Each generation stage permits one semantic repair; typed outputs permit two
validation retries. Subjective opinions and feelings are excluded even when
attributed. Verifiability is assessed, not truth against external evidence.

## Output and failures

`ExtractionResult` contains the parsed `document`, ordered `sentences`, flattened
`claims`, and `requested_models`. Each sentence records selected/clarified text,
claims, stage outputs, actual model identity, available confidence, and usage.
Statuses are `extracted`, `no_verifiable_claims`, `unresolved_ambiguity`, and
`validation_failed`.

Claim offsets are half-open Python character indices into the original Markdown
sentence, not the generated claim. Input/configuration errors raise `ValueError`.
Operational failures raise `ClaimifyError` with section, sentence, stage, and original
cause; remaining workers are cancelled and awaited. Context is never silently
truncated. Semantic rejection is distinct from an operational failure.

## Verification

Run from the repository root:

```sh
uv run python -m unittest discover -s tests -p 'test_claimify.py' -v
uv run python -m data_platform.digest.claimify.evaluate tests/fixtures/claimify/cases.json --live --output /tmp/claimify-review.json
```

The first command is offline. The second makes paid provider calls and writes a
report for human semantic review; use a new output path. Offline tests do not prove
claim accuracy. See [implementation notes](IMPLEMENTATION_PLAN.md),
[the notebook](../../../../notebooks/claimify.ipynb), and
[OpenIE](../openie/README.md) for the tuple-based alternative.
