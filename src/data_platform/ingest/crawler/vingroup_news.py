"""Discover Vingroup news metadata, then fetch content from the saved manifest."""

from __future__ import annotations

import argparse
import sys
import time as time  # Compatibility for callers patching the transport clock here.
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

from ._errors import CrawlError
from ._jsonl import append_record, read_records
from ._parsing import (
    DATE_RE as DATE_RE,
)
from ._parsing import (
    PAGE_RE as PAGE_RE,
)
from ._parsing import (
    START_URL,
    parse_article,
    parse_listing,
)
from ._parsing import (
    USER_AGENT as USER_AGENT,
)
from ._parsing import (
    absolute_url as absolute_url,
)
from ._parsing import (
    card_url as card_url,
)
from ._parsing import (
    clean_text as clean_text,
)
from ._parsing import (
    same_site as same_site,
)
from ._transport import Transport

# Preserve historical import paths for class reprs and pickles.
CrawlError.__module__ = __name__
Transport.__module__ = __name__


def _saved_urls(path: Path, *, require_content: bool = False) -> set[str]:
    """Only completed article records count as fetched on a resumed crawl."""
    if not path.exists():
        return set()
    return {
        record["url"]
        for record in read_records(path)
        if not require_content
        or (record.get("content_text") and record.get("content_html"))
    }


def discover(args, transport) -> int:
    known = _saved_urls(args.output)
    pending = deque([START_URL])
    visited = set()
    fingerprints = set()
    pages = saved = 0
    while pending and (args.max_pages is None or pages < args.max_pages):
        url = pending.popleft()
        if url in visited:
            continue
        visited.add(url)
        records, links = parse_listing(transport.get(url), url)
        fingerprint = frozenset(record["url"] for record in records)
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        pages += 1
        for record in records:
            if record["url"] not in known:
                append_record(args.output, record)
                known.add(record["url"])
                saved += 1
        pending.extend(
            link for link in links if link not in visited and link not in pending
        )
        print(
            f"Discovered page {pages}: {len(records)} articles ({saved} newly saved)",
            file=sys.stderr,
        )
    print(f"Saved {saved} new article links to {args.output}", file=sys.stderr)
    return 0


def fetch(args, transport) -> int:
    known = _saved_urls(args.output, require_content=True)
    attempted = failed = saved = 0
    seen = set()
    for record in read_records(args.input):
        url = record["url"]
        if url in known or url in seen:
            continue
        if args.max_articles is not None and attempted >= args.max_articles:
            break
        seen.add(url)
        attempted += 1
        try:
            content = parse_article(transport.get(url), url)
        except CrawlError as exc:
            append_record(
                args.errors,
                {
                    "url": url,
                    "error": str(exc),
                    "failed_at": datetime.now(UTC).isoformat(),
                },
            )
            print(str(exc), file=sys.stderr)
            failed += 1
            continue
        append_record(args.output, {**record, **content})
        known.add(url)
        saved += 1
        print(f"Saved article {saved}: {url}", file=sys.stderr)
    print(
        f"Saved {saved} articles; {failed} failed. Output: {args.output}",
        file=sys.stderr,
    )
    return 1 if failed else 0


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("discover", "fetch"):
        command = commands.add_parser(name)
        command.add_argument(
            "--output",
            type=Path,
            default=Path(
                f".data/vingroup_{'links' if name == 'discover' else 'articles'}.jsonl"
            ),
        )
        command.add_argument(
            "--timeout", type=positive_int, default=30, help="timeout in seconds"
        )
        command.add_argument(
            "--delay", type=float, default=1.0, help="minimum seconds between requests"
        )
        command.add_argument(
            "--retries", type=int, default=3, help="retries after transient failures"
        )
        if name == "discover":
            command.add_argument("--max-pages", type=positive_int)
        else:
            command.add_argument(
                "--input", type=Path, default=Path(".data/vingroup_links.jsonl")
            )
            command.add_argument("--errors", type=Path)
            command.add_argument("--max-articles", type=positive_int)
    args = parser.parse_args(argv)
    if args.delay < 0 or args.retries < 0:
        parser.error("--delay and --retries must be nonnegative")
    if args.command == "fetch":
        args.errors = args.errors or args.output.with_suffix(".errors.jsonl")
        if len({p.resolve() for p in (args.input, args.output, args.errors)}) != 3:
            parser.error("input, output and error files must be different")
    try:
        with Transport(args) as transport:
            return (
                discover(args, transport)
                if args.command == "discover"
                else fetch(args, transport)
            )
    except (CrawlError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(
            "Interrupted; saved records are retained for the next run.", file=sys.stderr
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
