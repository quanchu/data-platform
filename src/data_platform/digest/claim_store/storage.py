"""Human-reviewed JSON files are the source of truth for claim approval."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from ..claimify.models import ExtractionResult
from .models import ClaimRecord, Status


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class ClaimFiles:
    def __init__(self, root: Path):
        self.root = root

    def _read(self, path: Path) -> ClaimRecord:
        try:
            record = ClaimRecord.model_validate_json(path.read_text(encoding="utf-8"))
            if path.stem != record.id:
                raise ValueError("filename must match claim ID")
            return record
        except (ValueError, OSError) as exc:
            raise ValueError(f"Invalid claim file {path}: {exc}") from exc

    def list_claims(self, status: Status) -> list[ClaimRecord]:
        if status not in ("verified", "unverified"):
            raise ValueError("status must be verified or unverified")
        return [
            self._read(path) for path in sorted((self.root / status).glob("*.json"))
        ]

    def get_claim(self, id: str) -> ClaimRecord | None:
        # Validate IDs before constructing a path from user input.
        if len(id) != 64 or any(c not in "0123456789abcdef" for c in id):
            raise ValueError("Invalid claim ID")
        for status in ("verified", "unverified"):
            path = self.root / status / f"{id}.json"
            if path.exists():
                return self._read(path)
        return None

    def store_unverified(
        self, result: ExtractionResult, *, source: str
    ) -> list[ClaimRecord]:
        # Claimify retains parsed source content, not the original Markdown string.
        document_hash = _hash(result.document.model_dump_json())
        sections = {s.index: s for s in result.document.sections}
        sentences = {
            (s.section_index, s.index): s
            for section in result.document.sections
            for s in section.sentences
        }
        records = []
        for claim in result.claims:
            section = sections[claim.section_index]
            sentence = sentences[(claim.section_index, claim.sentence_index)]
            identity = [
                document_hash,
                source,
                claim.section_index,
                claim.sentence_index,
                claim.start,
                claim.end,
                claim.text,
            ]
            id = _hash(json.dumps(identity, ensure_ascii=False))
            record = ClaimRecord(
                id=id,
                text=claim.text,
                source=source,
                document_hash=document_hash,
                document_title=result.document.title,
                original_sentence=sentence.text,
                section_title=section.subtitle,
                section_index=claim.section_index,
                sentence_index=claim.sentence_index,
                start=claim.start,
                end=claim.end,
                created_at=datetime.now(timezone.utc),
            )
            existing = self.get_claim(id)
            if existing is None:
                path = self.root / "unverified" / f"{id}.json"
                with path.open("x", encoding="utf-8") as stream:
                    stream.write(record.model_dump_json(indent=2) + "\n")
            records.append(existing or record)
        return records
