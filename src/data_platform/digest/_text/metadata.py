"""Recognize article date and author lines without treating prose as metadata."""

import re

_METADATA = re.compile(
    r"^(?:published(?:\s+date|\s+on)?|publication date|date|ngày đăng(?: bài)?|"
    r"ngày xuất bản|đăng ngày|updated|cập nhật|author|by|tác giả)\b",
    re.IGNORECASE,
)
_DATE = re.compile(r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}(?:\s+.*)?$")


def is_metadata_line(line: str) -> bool:
    line = line.strip()
    if _DATE.fullmatch(line):
        return True
    if re.match(r"^(?i:by|author:|tác giả:)\s+[A-ZÀ-Ỹ][^.!?]{0,100}$", line):
        return True
    return bool(_METADATA.match(line) and re.search(r"\d", line))
