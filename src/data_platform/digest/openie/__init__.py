"""Reference resolution followed by OpenIE, with injectable async backends."""
from ..source import Document, Section, Sentence, parse_document
from .backends import LLMPropositionExtractor, LLMReferenceResolver, PropositionExtractor, ReferenceResolver
from .extractor import OpenIEError, extract_claims
from .models import (
    BackendMetadata, Claim, Evidence, ExtractionResponse, ExtractionResult,
    Proposition, Qualifier, Replacement, ResolutionResponse, ResolvedSentence,
    SentenceResult, StageRecord,
)

__all__ = [
    'extract_claims', 'parse_document', 'OpenIEError', 'ReferenceResolver',
    'PropositionExtractor', 'LLMReferenceResolver', 'LLMPropositionExtractor',
    'Document', 'Section', 'Sentence', 'BackendMetadata', 'Claim', 'Evidence',
    'ExtractionResponse', 'ExtractionResult', 'Proposition', 'Qualifier',
    'Replacement', 'ResolutionResponse', 'ResolvedSentence', 'SentenceResult', 'StageRecord',
]
