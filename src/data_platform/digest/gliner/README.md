# GLiNER entity tagging

Extract source-aligned entity mentions with a caller-loaded GLiNER model. The
package contains the chunking and selection logic used by
[the notebook](../../../../notebooks/gliner_entity_tagging.ipynb).

```python
from gliner import GLiNER
from data_platform.digest.gliner import extract_entities

model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1", local_files_only=True)
model.to("cpu")
model.eval()
entities = extract_entities(
    markdown_doc,
    model=model,
    labels=["company name", "date", "email", "person name"],
    threshold=0.5,
    chunk_words=200,
    chunk_overlap=40,
)
```

The example requires cached weights. The caller controls loading, device, and
whether downloads are allowed. Importing this package loads neither GLiNER nor
PyTorch and never downloads a model. Inference is synchronous and runs under
`torch.inference_mode()`; offload it yourself when using an async application.

Chunks use the supplied model's word splitter and maximum length. Exact duplicate
predictions retain their highest score. Overlapping predictions retain the
higher-scoring span; ties use source order. Output is an ordered list of dictionaries
with `text`, `label`, `score`, `start`, and `end`. Offsets are half-open Python
character indices into the unchanged source. Empty input returns an empty list.
Scores and selected mentions require human review.

The notebook owns model loading, DataFrame previews, and JSON/CSV exports. The library
neither reads input files nor writes exports. `light_finetune.py` remains a
placeholder; no training workflow is implemented.

```sh
uv run python -m unittest discover -s tests -p 'test_gliner.py' -v
```

Tests use a fake model and cover chunk boundaries, Unicode offsets, deduplication,
overlap selection, empty inputs, and notebook exports without downloading weights.

The live demo loads this model lazily and reuses it between runs. Chunk inference
and final entity selection emit source-aligned observations; ordinary library
calls need no observer. Overlap selection is a separate deterministic helper.
