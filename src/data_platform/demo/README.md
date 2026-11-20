# Live digest demonstration

This local application runs the actual Python APIs and streams their processing
steps to an HTML presentation. It does not simulate model results.

## Start

From the repository root:

```sh
uv sync --extra demo --extra offline
uv run python -m data_platform.demo
```

Open **http://127.0.0.1:8765**. Use `--port 8766` for another port or
`--store .data/claims-demo-other` for a separate review directory. The server binds
to loopback only. It loads `.env` without replacing existing process settings.
Credentials are never sent to the browser; the header shows configured providers.

The optional `offline` extra supplies TF-IDF and sentence embeddings. Lexical
counts work with just the `demo` extra. Model weights are not bundled. Prepare
local weights before presenting, or clear **Use cached local weights only** to
allow their first download. Entity inference uses the existing multilingual
GLiNER default on CPU; semantic segmentation accepts a local embedding directory.
Model loading can take time, so run a warm-up before the presentation.

## Suggested walkthrough

1. Choose an article from **.data Markdown** and run segmentation. The default
   uses semantic embeddings, heading classification, and heading generation with
   real configured models. Select a processing step to highlight its source.
   Use **Refresh files** to discover newly added Markdown, including `segmented/`.
   Lexical counts and disabled heading options remain available for offline runs.
2. Pass the segmented result to Claimify, or choose a prepared article from
   `.data/segmented/`. Inspect selection, ambiguity, generation, claim checks,
   and coverage with live decision and generation models.
3. Run OpenIE on the same input. Inspect section-level reference resolution and
   sentence-level relation extraction. Each result links to its source sentence.
4. Run entity tagging to see overlapping inference chunks, then final mention
   selection. Cache the model before presenting if internet access is uncertain.
5. From a completed Claimify run, choose **Send to claim store**, then save. This
   creates unverified records only. Manually inspect and move selected JSON files
   from `.data/claims-demo/unverified/` to `verified/`, then refresh. Run another
   Claimify extraction and compare against that approved set, or retrieve using
   source text as a query. An empty verified store produces no matches.

Segmentation results can be passed explicitly to Claimify or OpenIE. Extraction
requires one H1 title and nonempty H2 sections. Recognized date and author
metadata may appear between the title and first section. The server validates this before
starting extraction; it never silently inserts headings. You may enable heading
generation, edit the Markdown, or select a prepared segmented article.

## Controls and provenance

**Processing steps** shows a fixed set of logical stages. Toggle a stage to
select its evidence; repeated calls and sentence checks stay within that stage.
The arrows navigate stages. **Follow live** follows arriving events. Turn it off, click a step, or use the
arrows to inspect earlier work. This pauses the presentation view only; computation
continues. **Stop task** stops async work and waits for cleanup. A blocking local inference
call finishes before its run slot is released.

Only one run executes at a time. The latest 12 runs and their event histories stay
in server memory until eviction or restart. The **Final result** panel stays visible when reviewing earlier steps and shows
rendered Markdown or tables alongside complete JSON. Segmentation has **Rendered preview**, **Markdown source**, and **Download .md**
controls in **Final segmented Markdown**; **View final result** jumps to it, and **Rendered Markdown preview** also previews the input.
Previews disable embedded HTML, scripts, and remote images. Claimify results
show sentence outcome counts so validation failures remain visible. **Save result** explicitly
writes `.data/demo-results/<run-id>/result.json` by default. Edit **Save folder**
to use another local directory (absolute, `~`, or relative to the server working
directory); leaving it blank uses the default. The chosen directory contains
`<run-id>/result.json` and `run.json` (source, settings,
output, and events), plus `result.md` for segmentation. Repeated saves reuse the
same directory. Input files are never overwritten, and saved artifacts are excluded
from the input selector. **Download result** also exports JSON through the browser.
There is no automatic saving. Explicitly saved runs in the default result
directory can be reopened after a server restart; custom save directories remain
export-only. Use `--data PATH` to choose another
input directory and result destination.
A handoff to storage requires its completed Claimify run to remain in history.

Highlights use Python character offsets, including emoji and Vietnamese text.
Extraction after segmentation uses the generated Markdown as a distinct source.
Retrieved claims show their own source context rather than highlighting unrelated
positions in the current article. Source and model text are displayed as text,
never executed as HTML.

## Implementation and checks

`app.py` serves the API and static assets; `tasks.py` adapts existing workflows;
`runs.py` owns lifecycle, observation, and cancellation. The frontend is buildless
HTML/CSS/JavaScript. No frontend dependency installation is needed.

API routes: `/api/documents`, `/api/documents/content?path=...`,
`POST /api/runs/{id}/save`, `/api/capabilities`, `/api/examples`, `/api/claims`, `POST /api/runs`,
`GET /api/runs/{id}`, `GET /api/runs/{id}/events`, and
`POST /api/runs/{id}/cancel`. Events include sequence, stage, status, run ID,
document ID, source spans, and stage data. SSE reconnection resumes after the
last event ID. The task adapters never approve claims.

```sh
uv run python -m unittest discover -s tests -v
```

Demo tests inject deterministic models, exercise streaming and handoffs, and check
cancellation without provider requests. Live semantic quality remains a separate
manual check. The server reports missing credentials or uncached weights rather
than silently switching to fabricated outputs.

A run can be reopened at `/?run=<run-id>` while retained in memory or explicitly
saved under the default result directory. New runs update this URL.
