"""Common utilities, settings, types, and errors."""

from urban_rag.common.errors import (
    DocumentNotFoundError,
    GenerationError,
    IngestError,
    RateLimitError,
    RetrievalError,
    ServiceUnavailableError,
    UrbanRagError,
    ValidationError,
)
from urban_rag.common.types import (
    Answer,
    AnswerDiagnostics,
    AnswerResponse,
    ChunkRecord,
    Citation,
    DocumentRecord,
    LayoutBlock,
    PageMetadata,
    PageRecord,
    QueryRequest,
    RetrievalCandidate,
    RetrievalResult,
    SectionRecord,
    Trace,
    TraceSpan,
)

__all__ = [
    # errors
    "DocumentNotFoundError",
    "GenerationError",
    "IngestError",
    "RateLimitError",
    "RetrievalError",
    "ServiceUnavailableError",
    "UrbanRagError",
    "ValidationError",
    # types
    "Answer",
    "AnswerDiagnostics",
    "AnswerResponse",
    "ChunkRecord",
    "Citation",
    "DocumentRecord",
    "LayoutBlock",
    "PageMetadata",
    "PageRecord",
    "QueryRequest",
    "RetrievalCandidate",
    "RetrievalResult",
    "SectionRecord",
    "Trace",
    "TraceSpan",
]
