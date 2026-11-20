> Historical design notes. For current behavior and usage, see [README.md](README.md).

# Claimify implementation

## Contract

The async `extract_claims(markdown_doc, *, question=None, decision_model="typesafe:jev-latest", generation_model="openai:gpt-6-luna", max_concurrency=20)` API consumes already segmented Markdown. Exactly one H1 title and nonempty H2 subtitles are required. Body content belongs beneath an H2. H3–H6 remain within their H2 section. Blank input produces no requests.

Each sentence receives the document title, enclosing subtitle, complete original section body, original target sentence/index, and optional question. Context never crosses sections. Headings are context, not extraction targets. Original half-open Python character offsets are retained. Code, HTML blocks and tables are excluded from extraction and reported as excluded spans, but remain in source context.

## Pipeline

1. Parse structure and split paragraphs/list items into source-aligned sentences.
2. Jev selects fully factual, mixed or unverifiable sentences.
3. Luna filters mixed sentences; Jev checks support, coverage and removal of opinions.
4. Jev assesses referential/structural ambiguity and supplied name expansions. Unresolvable ambiguity rejects the sentence.
5. Luna clarifies when needed; Jev checks support, meaning and ambiguity resolution.
6. Luna generates standalone claims. Jev checks each claim and set-level coverage.
7. Return ordered sentence outcomes, claims, provenance, model versions, confidence and usage.

All semantic decisions use Jev; all text generation uses Luna. Python owns structural validation and orchestration. Each generation stage permits one semantic repair; structured-output validation permits two retries. Operational errors raise `ClaimifyError` with section, sentence and stage, cancel pending workers, and never become semantic rejections. Oversized contexts are never silently truncated. Single decisions replace repeated voting; provider-default confidence thresholds are retained without an uncalibrated application cutoff.

Statuses: `extracted`, `no_verifiable_claims`, `unresolved_ambiguity`, `validation_failed`.

This is a section-context adaptation, not a benchmark reproduction, of [Claimify §3](https://arxiv.org/pdf/2502.10855). [TypeSafe integration](https://pydantic.dev/docs/ai/models/typesafe/) supplies typed decisions. No retrieval, external fact-checking, automatic segmentation or persistence is part of the extraction API.

## Usage

Install with `uv sync` (or `uv sync --extra offline` to retain optional local segmentation dependencies). Set `TYPESAFE_API_KEY` and `OPENAI_API_KEY` in the process environment. Importing and parsing require no credentials. Injected model objects use their own configuration.

```python
from data_platform.digest.claimify import extract_claims

result = await extract_claims("# Report\n\n## Acme\n\nAcme plans two factories.")
for claim in result.claims:
    print(claim.text, claim.start, claim.end)
```

## Validation and rollout

Run `uv run python -m unittest discover -s tests -p 'test_claimify.py'` for offline parsing, context, routing, repair, cancellation, concurrency and real TypeSafe schema translation tests. No provider calls occur in these tests.

The English/Vietnamese cases in `tests/fixtures/claimify/cases.json` contain expected outcomes and propositions for manual review. Live evaluation is deliberately opt-in:

```sh
uv run python -m data_platform.digest.claimify.evaluate tests/fixtures/claimify/cases.json --live --output /tmp/claimify-evaluation.json
```

The report records latency, sentence outcome rates, model versions, per-stage usage and generated claims. Expected status matches are diagnostic only. A human must fill the entailment, coverage and contextual-completeness review fields and adjudicate the fixtures; semantic quality is not proven by offline test doubles. Live evaluation and human approval of its quality remain prerequisites for downstream use. No live quality results are claimed by this implementation.
