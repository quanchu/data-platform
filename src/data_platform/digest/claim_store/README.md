# Claim storage and conflict checking

See [the walkthrough notebook](../../../../notebooks/claim_store.ipynb) for extraction,
manual approval, retrieval, comparison tables, and report export using fictional examples
or your own Markdown files.

`ClaimStore` keeps reviewable JSON in `unverified/` and `verified/`, with a disposable
`index.sqlite` embedding cache. Verified means manually approved by you, not
externally fact-checked. Moving a JSON file into `verified/` approves it; moving it
out or deleting it withdraws approval at the next synchronization.

```python
from data_platform.digest.claimify import extract_claims
from data_platform.digest.claim_store import ClaimStore

store = ClaimStore('.data/claims')
result = await extract_claims(markdown)
records = store.store_unverified(result, source='article.md')

# Manually move approved <id>.json files from unverified/ to verified/.
matches = await store.retrieve('Acme revenue was $10m in 2025', top_k=20)
report = await store.check(records, top_k=20)
for pair in report.review_required:
    print(pair.decision.verdict, pair.claim.text, pair.verified_claim.text)
    print(pair.claim.source, pair.verified_claim.source, pair.similarity)
```

`list_claims(status)` and `get_claim(id)` read the JSON files; `get_claim` returns
`None` for a missing ID and prefers the verified copy if both folders contain it.
Status is derived only from the folder, never a field in JSON. Use a move rather
than a copy when approving. Repeated imports preserve existing records, including
manual edits and creation timestamps. IDs hash the parsed document content,
source reference, claim location, and original claim text. Claimify does not retain
the complete original Markdown, so document identity uses its parsed source record.
Keep IDs and filenames unchanged when editing records; new imports of changed
source content receive new IDs and need review.

`await sync_index()` returns verified, embedded, and removed counts. `retrieve()`
and `check()` synchronize automatically; a check uses one verified snapshot for
the entire document. Text edits and embedding model changes regenerate embeddings;
metadata edits do not. Delete `index.sqlite` to rebuild it from verified JSON.
Invalid JSON, mismatched filenames, invalid vectors, and model failures raise errors.
There is no automatic approval, repair of source files, or conflict resolution.

Embeddings default to `openai:text-embedding-3-small`, and comparisons to
`typesafe:jev-latest`. Configure provider credentials in the calling environment.
Pass `embedding_model` (a string or Pydantic AI `EmbeddingModel`) and
`comparison_model` (a string or Pydantic AI `Model`) to inject alternatives.
Injected embedding models must expose distinct provider/model identifiers when
vector semantics change. Comparisons run with `max_concurrency=4` by default.

Results include full source records and cosine similarities. The typed verdicts
are `conflict`, `equivalent`, `compatible`, `unrelated`, and `uncertain`.
`report.conflicts`, `report.uncertain`, and `report.review_required` support review.
The default top 20 candidates are a proof of concept: **no conflict found means
no conflict found among retrieved candidates**, not exhaustive consistency or
approval. Similarity itself is not a conflict decision. This local store targets
one user and roughly 500–10,000 claims, with cached vectors loaded into NumPy.

Run offline contracts with:

```sh
.venv/bin/python -m unittest discover -s tests -p test_claim_store.py -v
```

The semantic evaluation cases in `tests/fixtures/claim_store/comparisons.json`
include manually reviewed expected verdicts. They can be sent to the comparison
agent to evaluate a configured model; deterministic unit tests validate plumbing,
not live model accuracy.
