# Vingroup news ingestion

Discover Vietnamese Vingroup news links and fetch article bodies using HTTP
requests. Discovery and fetching are separate resumable commands.

## Setup and usage

Run from the repository root with Python 3.14 or later:

```sh
uv sync
uv run python -m data_platform.ingest.crawler.vingroup_news discover --output .data/vingroup_links.jsonl
uv run python -m data_platform.ingest.crawler.vingroup_news fetch --input .data/vingroup_links.jsonl --output .data/vingroup_articles.jsonl
```

For a small run, use `--max-pages 1` during discovery and `--max-articles 2` during
fetching. Both support `--timeout 30`, `--delay 1`, and `--retries 3`. Run either
subcommand with `--help` for its options. No browser installation is needed.

## Output and resuming

Discovery follows the Vietnamese all-news pagination and extracts `.itemNews`
metadata into UTF-8 JSONL: `url`, `title`, `published_date` (ISO date), `category`,
`thumbnail_url`, and `source_listing_url`. Missing values are `null`.

Fetching reads that manifest and extracts `.newsWrap.stagger-up`, adding
`content_html`, `content_text`, and `fetched_at` (UTC). Both commands append and
deduplicate by URL. Discovery starts at page one on each run; successful article
fetches are skipped. Re-running retries failures. The article limit counts attempted,
previously unsaved articles. Use separate output paths for fresh snapshots.

Failed fetches are logged to `.data/vingroup_articles.errors.jsonl` by default;
set `--errors PATH` to change the path. The log is historical and may include
articles that succeeded on a later run. Output is raw article content, not segmented
Markdown ready for claim extraction.

## Errors and verification

Connection errors, timeouts, HTTP 429, and HTTP 5xx receive bounded retries.
HTTP 401/403 and Cloudflare challenges produce access errors; challenge pages are
not saved as articles. Failed extraction is reported instead of saving empty content.
Commands preserve completed records and return nonzero on failure. Site markup and
access controls may change and require maintenance.

Run offline fixtures from the repository root:

```sh
uv run python -m unittest discover -s tests -p 'test_vingroup_news.py' -v
```
