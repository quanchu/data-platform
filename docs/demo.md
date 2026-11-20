# Live demo quickstart

From the repository root, with Python 3.14 or later:

```sh
uv sync --extra demo --extra offline
uv run python -m data_platform.demo
```

Open <http://127.0.0.1:8765>. Use `--port 8767` if that port is occupied.
The server loads `.env` without replacing existing environment variables; use
[.env.example](../.env.example) as the configuration reference. Provider readiness
means a credential is configured, not that a live request has succeeded.

## Run and inspect

1. Select a Markdown article from `.data/`, upload one, or edit the source.
2. Choose a workflow and inspect its settings. Segmentation defaults in the UI
   use semantic embeddings, heading classification, and heading generation.
   Model-backed operations make live requests. Local entity tagging needs model
   weights; cached-only mode prevents downloading missing weights.
3. Run the task. The source panel switches between **Raw Markdown** and
   **Rendered preview**. Selecting evidence switches to raw text to show offsets.
4. Inspect **What changed** and expand the logical **Processing steps** below it.
   **Follow live** tracks arriving work; turning it off pauses the view only.
5. Inspect **Final result**. Segmentation provides rendered Markdown, raw Markdown,
   and a `.md` download. Its output can be passed to Claimify or OpenIE.

**Stop task** requests cancellation and waits for active work to finish cleanup.
Only one task runs at a time on each server. A local inference call already running
in a thread finishes before the run slot is released.

## Save results

Saving is optional. **Save folder** defaults to `.data/demo-results` and accepts
another local directory. Relative paths start at the server's working directory;
each run gets its own subfolder. Exports include `run.json` with source, settings,
and events, plus `result.json` and, for segmentation, `result.md`.

Saved runs in the default result directory can be reopened using `/?run=<run-id>`
after a server restart. Custom destinations are export-only. Unsaved history is
limited to the latest 12 in-memory runs. Browser downloads are also available.

## Understand Claimify outcomes

| Sentence status | Meaning |
| --- | --- |
| `extracted` | Claims passed the required source-based checks. |
| `no_verifiable_claims` | No verifiable factual assertions were returned. |
| `unresolved_ambiguity` | Available context could not resolve the sentence's meaning or references. |
| `validation_failed` | Filtering, clarification, claim, or coverage checks failed after retries. |

Only `extracted` sentences contribute final claims. A completed run can contain
failed sentence validations; those are different from a task-level provider or
connection failure. Review omitted sentences before treating the result as complete.

For connection problems, distinguish a page that cannot reach the local server
from a task error contacting a model provider. Check the displayed error and the
server terminal. Restart the server after Python changes and refresh the page
after interface changes.

See the [complete demo guide](../src/data_platform/demo/README.md) for API routes
and the claim-store walkthrough.
