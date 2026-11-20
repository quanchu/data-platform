"""Shared setup, pandas conversion, and JSON exports for the notebooks."""

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel


def setup_notebook() -> Path:
    """Load this checkout's environment and configure readable DataFrame previews."""
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env", override=False)
    pd.set_option("display.max_colwidth", 120)
    return project_root


def to_dataframe(
    records: Iterable[Any],
    *,
    columns: Sequence[str],
    source: str | None = None,
) -> pd.DataFrame:
    """Convert models, dataclasses, or dictionaries; retain columns for empty input.

    Pass the original Markdown to add a source slice from each record's start/end
    offsets. Nested Pydantic values are converted to JSON-compatible dictionaries.
    """
    rows = []
    for record in records:
        if isinstance(record, BaseModel):
            row = record.model_dump(mode="json")
        elif is_dataclass(record) and not isinstance(record, type):
            row = asdict(record)
        else:
            row = dict(record)
        if source is not None:
            row["source"] = source[row["start"] : row["end"]]
        rows.append(row)
    return pd.DataFrame(rows, columns=list(columns))


def sentence_rows(document) -> list[dict]:
    """Flatten source sentences while keeping their section labels."""
    return [
        {"section": section.subtitle, **sentence.model_dump(mode="json")}
        for section in document.sections
        for sentence in section.sentences
    ]


def write_json(path: str | Path, data: BaseModel | Mapping[str, Any]) -> Path:
    """Write a UTF-8 JSON export, creating its output directory when needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = data.model_dump(mode="json") if isinstance(data, BaseModel) else data
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
