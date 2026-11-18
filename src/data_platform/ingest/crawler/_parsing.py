"""Pure URL, listing, and article parsing for Vingroup news."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from urllib.parse import urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup

from ._errors import CrawlError

START_URL = "https://vingroup.net/tin-tuc-su-kien"
DATE_RE = re.compile(r"\b(\d{2})[-/.](\d{2})[-/.](\d{4})\b")
PAGE_RE = re.compile(r"^/tin-tuc-su-kien(?:/tat-ca(?:/\d+)?)?/?$")
USER_AGENT = "Mozilla/5.0 (compatible; VingroupNewsCrawler/0.1)"


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
