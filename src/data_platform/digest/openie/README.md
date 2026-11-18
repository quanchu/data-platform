# Reference resolution → OpenIE

Extract structured propositions and standalone claim text through two stages:
section-scoped reference resolution, then sentence-level OpenIE. This is an
independent alternative to [Claimify](../claimify/README.md).

## Setup and usage

Run `uv sync` from the repository root. Default LLM backends use process
`OPENAI_API_KEY`; model arguments also accept configured Pydantic AI models.

```python
from data_platform.digest.openie import extract_claims

result = await extract_claims(
    "# Report\n\n## Factory\n\nAcme builds cars. It plans buses in 2027.",
    resolution_model="openai:gpt-6-luna",
    extraction_model="openai:gpt-6-luna",
    max_concurrency=4,
)
for claim in result.claims:
    print(claim.subject, claim.relation, claim.object, claim.qualifiers)
    print(claim.text, claim.start, claim.end)
```

Use `await` in Jupyter/async functions or `asyncio.run` in scripts. The library
neither loads credentials from files nor saves output automatically.

## Input and stages

Input requires one nonempty H1 title and nonempty H2 section subtitles, with all
body content under H2s. Blank input makes no calls. Headings, code, HTML blocks,
and tables are not extraction targets. Resolution sees the title, subtitle,
complete original section body, and indexed target sentences; context never
crosses sections and is never silently truncated.

1. Resolve references once per section. Return every target sentence exactly once,
   with `resolved`, `unchanged`, or `unresolved` status. Replacements include source
   evidence pointing to a sentence in the same section or the title/subtitle.
2. Extract propositions from each original/resolved sentence pair. Unresolved
   sentences skip this stage. No separate semantic verification stage runs.

Prompts preserve English/Vietnamese source language, attribution, negation, dates,
conditions, and modality. Opinions and reported statements retain attribution.
Propositions are not externally verified facts; model output needs human review.

## Results and adapter interface

`ExtractionResult` provides ordered `sentences`, flattened `claims`, the parsed
`document`, and section-level `resolution_stages`. Sentence records contain their
resolution, claims, and extraction stage trace. Each claim contains `subject`,
`relation`, optional `object`, `qualifiers`, standalone `text`, source sentence and
section IDs, and original half-open Python character offsets. Qualifier kinds are
`time`, `location`, `condition`, `negation`, `modality`, `attribution`, and `other`.
Offsets identify the supporting original sentence, not the rewritten text.

Sentence outcomes are `extracted`, `no_claims`, and `unresolved_ambiguity`. Stage
traces include backend/model identity and available usage. Resolution usage is
recorded once per section rather than duplicated across sentences.

Dedicated model integrations implement either exported async protocol:

```python
# ReferenceResolver
async def resolve(*, title, section) -> ResolutionResponse: ...

# PropositionExtractor
async def extract(*, sentence, resolved) -> ExtractionResponse: ...
```

Pass adapter instances as `resolver=` and/or `extractor=`. Responses contain typed
records and `BackendMetadata(backend=..., model=..., usage=...)`; these types are
exported from `data_platform.digest.openie`. Mixed dedicated/LLM stages work.
Adapters own their configuration, so the corresponding `*_model` option applies
only when the default LLM backend is used. V1 bundles no dedicated NLP models.
Use async inference or offload blocking calls to a thread. Use `max_concurrency=1`
when a backend cannot safely handle simultaneous calls.

LLM output contracts receive two validation retries. Invalid adapter output and
provider failures raise `OpenIEError` with section, optional sentence, stage, and
original cause. Pending work is cancelled and awaited; caller cancellation
propagates. Invalid Markdown/configuration raises `ValueError`. No automatic
backend fallback or cross-sentence deduplication is performed.

## Verification

From the repository root:

```sh
uv run python -m unittest discover -s tests -p 'test_openie.py' -v
uv run python -m data_platform.digest.openie.evaluate tests/fixtures/openie/cases.json --live --output /tmp/openie-review.json
```

Live evaluation makes paid calls and requires a new output path. Override models
with `--resolution-model` and `--extraction-model`. Review reference correctness,
entailment, qualifier preservation, and coverage; status matches are diagnostic.
See [the notebook](../../../../notebooks/openie.ipynb) for intermediate inspection
and an adapter example.
