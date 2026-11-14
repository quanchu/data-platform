"""Discover Vingroup news metadata, then fetch content from the saved manifest."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup

START_URL = "https://vingroup.net/tin-tuc-su-kien"
DATE_RE = re.compile(r"\b(\d{2})[-/.](\d{2})[-/.](\d{4})\b")
PAGE_RE = re.compile(r"^/tin-tuc-su-kien(?:/tat-ca(?:/\d+)?)?/?$")
USER_AGENT = "Mozilla/5.0 (compatible; VingroupNewsCrawler/0.1)"


class CrawlError(Exception):
    """A fetch or extraction did not produce usable news data."""


def clean_text(value: str) -> str:
    return " ".join(value.split())


def absolute_url(value: str | None, base: str) -> str | None:
    if not value or value.strip().startswith(("#", "javascript:", "mailto:", "tel:")):
        return None
    parts = urlsplit(urljoin(base, value.strip()))
    if parts.scheme not in {"http", "https"}:
        return None
    return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))


def same_site(url: str) -> bool:
    return (urlsplit(url).hostname or "").lower() in {
        "vingroup.net",
        "www.vingroup.net",
    }


def card_url(card, base: str) -> str | None:
    for node in [card, *card.select("a, [data-href], [data-url], [onclick]")]:
        values = [node.get(key) for key in ("href", "data-href", "data-url")]
        # Some versions of the site navigate via an onclick on the news card.
        values.extend(re.findall(r"['\"]([^'\"]+)['\"]", node.get("onclick", "")))
        for value in values:
            url = absolute_url(value, base)
            if (
                url
                and same_site(url)
                and not PAGE_RE.fullmatch(urlsplit(url).path)
                and "/" in (value or "")
                and urlsplit(url).path != "/"
            ):
                return url
    return None


def parse_listing(html: str, source_url: str) -> tuple[list[dict], list[str]]:
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.select(".itemNews")
    if not cards:
        raise CrawlError(
            f"No .itemNews elements at {source_url}; page may be blocked or markup changed"
        )
    records = []
    seen = set()
    for card in cards:
        url = card_url(card, source_url)
        if not url:
            raise CrawlError(
                f"A .itemNews card has no usable article URL at {source_url}"
            )
        if url in seen:
            continue
        seen.add(url)
        title = card.select_one("h3, h2, .title, .name")
        category = card.select_one("h4, .category, .cate")
        image = card.select_one("img")
        date_match = DATE_RE.search(card.get_text(" ", strip=True))
        published_date = None
        if date_match:
            try:
                published_date = (
                    datetime.strptime("-".join(date_match.groups()), "%d-%m-%Y")
                    .date()
                    .isoformat()
                )
            except ValueError:
                pass
        records.append(
            {
                "url": url,
                "title": clean_text(title.get_text(" ", strip=True)) if title else None,
                "published_date": published_date,
                "category": clean_text(category.get_text(" ", strip=True))
                if category
                else None,
                "thumbnail_url": absolute_url(
                    image.get("data-src") or image.get("src"), source_url
                )
                if image
                else None,
                "source_listing_url": source_url,
            }
        )
    pages = set()
    for anchor in soup.select("a[href]"):
        url = absolute_url(anchor.get("href"), source_url)
        if url and same_site(url) and PAGE_RE.fullmatch(urlsplit(url).path):
            pages.add(url)
    return records, sorted(
        pages,
        key=lambda u: (
            int(re.search(r"/(\d+)/?$", u).group(1))
            if re.search(r"/(\d+)/?$", u)
            else 1
        ),
    )


def parse_article(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    wrapper = soup.select_one(".newsWrap.stagger-up")
    if wrapper is None:
        raise CrawlError(f"Missing .newsWrap.stagger-up at {url}")
    content_html = str(wrapper)
    for node in wrapper.select("script, style, noscript"):
        node.decompose()
    # Newlines preserve block boundaries without splitting inline formatting.
    for node in wrapper.select("p, div, h1, h2, h3, h4, h5, h6, li, blockquote, tr"):
        node.insert_before("\n")
        node.insert_after("\n")
    for node in wrapper.select("br"):
        node.replace_with("\n")
    text = "\n\n".join(
        clean_text(line) for line in wrapper.get_text().splitlines() if clean_text(line)
    )
    if not text:
        raise CrawlError(f"Empty article wrapper at {url}")
    return {
        "content_html": content_html,
        "content_text": text,
        "fetched_at": datetime.now(UTC).isoformat(),
    }


class Transport:
    def __init__(self, args):
        self.args = args
        self.client = None
        self.last_request = 0.0

    def __enter__(self):
        self.client = requests.Session()
        self.client.headers.update({"User-Agent": USER_AGENT})
        return self

    def __exit__(self, *_):
        if self.client:
            self.client.close()

    def get(self, url: str) -> str:
        for attempt in range(self.args.retries + 1):
            time.sleep(max(0, self.args.delay - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            try:
                response = self.client.get(
                    url, timeout=self.args.timeout, allow_redirects=True
                )
                status = response.status_code
                if status in {401, 403}:
                    reason = (
                        "Cloudflare challenge blocked the request."
                        if response.headers.get("cf-mitigated", "").lower()
                        == "challenge"
                        else "The server denied access to the request."
                    )
                    raise CrawlError(f"HTTP {status} at {url}. {reason}")
                if status != 429 and status < 500:
                    if status < 200 or status >= 300:
                        raise CrawlError(f"HTTP {status} at {url}")
                    if (
                        response.encoding is None
                        or response.encoding.lower() == "iso-8859-1"
                    ):
                        response.encoding = response.apparent_encoding
                    return response.text
                if attempt == self.args.retries:
                    raise CrawlError(f"HTTP {status} at {url}")
            except requests.RequestException as exc:
                if attempt == self.args.retries:
                    raise CrawlError(f"Failed fetching {url}: {exc}") from exc
            time.sleep(min(2**attempt, 30))
        raise AssertionError("unreachable")


def read_records(path: Path):
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict) or not isinstance(
                    record.get("url"), str
                ):
                    raise TypeError("expected an object with a string url")
                if not same_site(record["url"]) or not absolute_url(
                    record["url"], START_URL
                ):
                    raise ValueError("expected a Vingroup HTTP(S) URL")
            except (ValueError, TypeError) as exc:
                raise CrawlError(
                    f"Invalid JSONL record at {path}:{number}: {exc}"
                ) from exc
            yield record


def append_record(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def discover(args, transport) -> int:
    known = (
        {record["url"] for record in read_records(args.output)}
        if args.output.exists()
        else set()
    )
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
    known = (
        {
            record["url"]
            for record in read_records(args.output)
            if record.get("content_text") and record.get("content_html")
        }
        if args.output.exists()
        else set()
    )
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
