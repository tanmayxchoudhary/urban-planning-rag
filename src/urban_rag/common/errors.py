"""Typed error hierarchy for Urban RAG."""

from __future__ import annotations


class UrbanRagError(Exception):
    """Base exception for all Urban RAG errors."""

    code: str = "internal"
    message: str = "An unexpected error occurred"

    def __init__(self, message: str | None = None, trace_id: str | None = None) -> None:
        self.message = message or self.message
        self.trace_id = trace_id
        super().__init__(self.message)

    def to_dict(self) -> dict[str, str]:
        """Return a dict suitable for structured error responses."""
        out: dict[str, str] = {"code": self.code, "message": self.message}
        if self.trace_id:
            out["trace_id"] = self.trace_id
        return out


class DocumentNotFoundError(UrbanRagError):
    """Raised when a requested document is not in the corpus."""

    code = "document_not_found"
    message = "Document not found in corpus"


class ValidationError(UrbanRagError):
    """Raised when input validation fails."""

    code = "validation_error"
    message = "Input validation failed"


class IngestError(UrbanRagError):
    """Raised when ingestion fails."""

    code = "ingest_error"
    message = "Document ingestion failed"


class EmbeddingError(UrbanRagError):
    """Raised when embedding fails."""

    code = "embedding_error"
    message = "Embedding generation failed"


class RetrievalError(UrbanRagError):
    """Raised when retrieval fails."""

    code = "retrieval_error"
    message = "Retrieval failed"


class GenerationError(UrbanRagError):
    """Raised when generation fails."""

    code = "generation_error"
    message = "Answer generation failed"


class ServiceUnavailableError(UrbanRagError):
    """Raised when a required service is unavailable."""

    code = "service_unavailable"
    message = "Service unavailable"


class RateLimitError(UrbanRagError):
    """Raised when rate limit is exceeded."""

    code = "rate_limited"
    message = "Rate limit exceeded"


class ParseError(UrbanRagError):
    """Raised when document parsing fails (Docling, Marker, etc.)."""

    code = "parse_error"
    message = "Document parsing failed"
