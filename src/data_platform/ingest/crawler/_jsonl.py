"""UTF-8 JSONL persistence and manifest validation."""

from __future__ import annotations

import json
from pathlib import Path

from ._errors import CrawlError
from ._parsing import START_URL, absolute_url, same_site


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
