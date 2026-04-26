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
from urban_rag.common.logging import configure_logging, get_logger
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
    # types
    "Answer",
    "AnswerDiagnostics",
    "AnswerResponse",
    "ChunkRecord",
    "Citation",
    # errors
    "DocumentNotFoundError",
    "DocumentRecord",
    "GenerationError",
    "IngestError",
    "LayoutBlock",
    "PageMetadata",
    "PageRecord",
    "QueryRequest",
    "RateLimitError",
    "RetrievalCandidate",
    "RetrievalError",
    "RetrievalResult",
    "SectionRecord",
    "ServiceUnavailableError",
    "Trace",
    "TraceSpan",
    "UrbanRagError",
    "ValidationError",
    # logging
    "configure_logging",
    "get_logger",
]
