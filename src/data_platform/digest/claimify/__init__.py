"""Claimify extraction from Markdown with one H1 and H2 sections."""

from .document import parse_document
from .extractor import ClaimifyError, extract_claims
from .models import Claim, Document, ExtractionResult, Section, SentenceResult

__all__ = ["Claim", "ClaimifyError", "Document", "ExtractionResult", "Section", "SentenceResult", "extract_claims", "parse_document"]
