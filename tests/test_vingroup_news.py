import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from data_platform.ingest.crawler.vingroup_news import (
    START_URL,
    CrawlError,
    Transport,
    append_record,
    discover,
    fetch,
    main,
    parse_article,
    parse_listing,
    read_records,
)


def listing(article="/tin-tuc-su-kien/bai-viet-1", pages=""):
    return f'''<div class="itemNews"><a href="{article}#top"><img data-src="/photo.jpg"></a>
    <h4>Tin Vingroup</h4><h3>Tin tức Việt Nam</h3><span>17-11-2026</span></div>{pages}'''


ARTICLE = """<nav>Exclude navigation</nav><div class="stagger-up other newsWrap">
<h1>Tiêu đề</h1><p>Một <strong>đoạn</strong> văn.</p><p>Đoạn hai.<br>Dòng mới.</p>
<script>exclude script</script></div><footer>Exclude footer</footer>"""


class FakeTransport:
    def __init__(self, pages):
        self.pages = pages
        self.requests = []

    def get(self, url):
        self.requests.append(url)
        value = self.pages[url]
        if isinstance(value, Exception):
            raise value
        return value


class CrawlerTests(unittest.TestCase):
    def test_listing_metadata_and_pagination(self):
        html = listing(
            pages='<a href="/tin-tuc-su-kien/tat-ca/2">2</a><a href="/tin-tuc-su-kien/vingroup">category</a>'
        )
        records, pages = parse_listing(html + listing(), START_URL)
        self.assertEqual(len(records), 1)
        self.assertEqual(
            records[0],
            {
                "url": "https://vingroup.net/tin-tuc-su-kien/bai-viet-1",
                "title": "Tin tức Việt Nam",
                "category": "Tin Vingroup",
                "published_date": "2026-11-17",
                "thumbnail_url": "https://vingroup.net/photo.jpg",
                "source_listing_url": START_URL,
            },
        )
        self.assertEqual(pages, [START_URL + "/tat-ca/2"])

    def test_onclick_and_missing_metadata(self):
        records, _ = parse_listing(
            """<div class="itemNews" onclick="location.href='/bai-viet/123'"><span>News</span></div>""",
            START_URL,
        )
        self.assertEqual(records[0]["url"], "https://vingroup.net/bai-viet/123")
        self.assertIsNone(records[0]["published_date"])

    def test_invalid_listing(self):
        for html in ("<h1>Access denied</h1>", '<div class="itemNews">No link</div>'):
            with self.assertRaises(CrawlError):
                parse_listing(html, START_URL)

    def test_article_wrapper_and_paragraphs(self):
        record = parse_article(ARTICLE, START_URL)
        self.assertEqual(
            record["content_text"], "Tiêu đề\n\nMột đoạn văn.\n\nĐoạn hai.\n\nDòng mới."
        )
        self.assertIn("<strong>đoạn</strong>", record["content_html"])
        self.assertNotIn("Exclude", record["content_html"])
        for html in ("<p>No wrapper</p>", '<div class="newsWrap stagger-up"> </div>'):
            with self.assertRaises(CrawlError):
                parse_article(html, START_URL)

    def test_discovery_follows_pages_and_stops_repeated_content(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "links.jsonl"
            page2, page3 = START_URL + "/tat-ca/2", START_URL + "/tat-ca/3"
            transport = FakeTransport(
                {
                    START_URL: listing(pages=f'<a href="{page2}">2</a>'),
                    page2: listing("/article/2", f'<a href="{page3}">3</a>'),
                    page3: listing(
                        "/article/2", '<a href="/tin-tuc-su-kien/tat-ca/4">4</a>'
                    ),
                }
            )
            args = argparse.Namespace(output=output, max_pages=None)
            self.assertEqual(discover(args, transport), 0)
            self.assertEqual(len(list(read_records(output))), 2)
            self.assertEqual(len(transport.requests), 3)
            discover(args, transport)
            self.assertEqual(len(list(read_records(output))), 2)

    def test_saved_manifest_handoff_failures_resume_and_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, output, errors = [
                root / name
                for name in ("links.jsonl", "articles.jsonl", "errors.jsonl")
            ]
            urls = ["https://vingroup.net/article/1", "https://vingroup.net/article/2"]
            for url in [*urls, urls[0]]:
                append_record(manifest, {"url": url, "title": "Tiêu đề"})
            args = argparse.Namespace(
                input=manifest, output=output, errors=errors, max_articles=None
            )
            transport = FakeTransport({urls[0]: CrawlError("failed"), urls[1]: ARTICLE})
            self.assertEqual(fetch(args, transport), 1)
            self.assertEqual([r["url"] for r in read_records(output)], [urls[1]])
            self.assertEqual(next(iter(read_records(errors)))["error"], "failed")
            transport = FakeTransport({urls[0]: ARTICLE})
            args.max_articles = 1
            self.assertEqual(fetch(args, transport), 0)
            self.assertEqual(transport.requests, [urls[0]])
            self.assertEqual(len(list(read_records(output))), 2)
            self.assertEqual(next(iter(read_records(output)))["title"], "Tiêu đề")

    def test_discovery_page_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "links.jsonl"
            transport = FakeTransport(
                {START_URL: listing(pages='<a href="/tin-tuc-su-kien/tat-ca/2">2</a>')}
            )
            self.assertEqual(
                discover(argparse.Namespace(output=output, max_pages=1), transport), 0
            )
            self.assertEqual(transport.requests, [START_URL])

    def test_http_transient_retry_and_blocked_page(self):
        args = argparse.Namespace(timeout=1, retries=1, delay=0)
        responses = [requests.Response() for _ in range(3)]
        for response, status in zip(responses, [503, 200, 403]):
            response.status_code = status
            response._content = ARTICLE.encode("utf-8")
            response.encoding = "utf-8"
        responses[-1].headers["cf-mitigated"] = "challenge"
        with (
            Transport(args) as transport,
            patch.object(transport.client, "get", side_effect=responses) as get,
            patch("data_platform.ingest.crawler.vingroup_news.time.sleep"),
        ):
            self.assertEqual(transport.get(START_URL), ARTICLE)
            with self.assertRaisesRegex(CrawlError, "403.*Cloudflare challenge"):
                transport.get(START_URL)
            self.assertEqual(get.call_count, 3)
            get.assert_called_with(START_URL, timeout=1, allow_redirects=True)

    def test_requests_timeout_retry_exhaustion(self):
        args = argparse.Namespace(timeout=1, retries=2, delay=0)
        with (
            Transport(args) as transport,
            patch.object(
                transport.client, "get", side_effect=requests.Timeout("timed out")
            ) as get,
            patch("data_platform.ingest.crawler.vingroup_news.time.sleep"),
        ):
            with self.assertRaisesRegex(CrawlError, "Failed fetching.*timed out"):
                transport.get(START_URL)
            self.assertEqual(get.call_count, 3)

    def test_http_errors_fail_without_retry(self):
        args = argparse.Namespace(timeout=1, retries=2, delay=0)
        for status in (401, 403, 404):
            response = requests.Response()
            response.status_code = status
            with (
                Transport(args) as transport,
                patch.object(transport.client, "get", return_value=response) as get,
            ):
                with self.assertRaisesRegex(CrawlError, str(status)):
                    transport.get(START_URL)
                self.assertEqual(get.call_count, 1)

    def test_malformed_manifest_and_conflicting_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text('{"title":"no URL"}\n', encoding="utf-8")
            with self.assertRaisesRegex(CrawlError, ":1"):
                list(read_records(path))
            with self.assertRaises(SystemExit):
                main(["fetch", "--input", str(path), "--output", str(path)])


if __name__ == "__main__":
    unittest.main()
