"""HTTP session lifetime, pacing, and transient-error retries."""

from __future__ import annotations

import time

import requests

from ._errors import CrawlError
from ._parsing import USER_AGENT


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
