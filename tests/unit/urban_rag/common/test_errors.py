"""Unit tests for common/errors.py — error hierarchy."""

from __future__ import annotations

import pytest

from urban_rag.common.errors import (
    DocumentNotFoundError,
    EmbeddingError,
    GenerationError,
    IngestError,
    RateLimitError,
    RetrievalError,
    ServiceUnavailableError,
    UrbanRagError,
    ValidationError,
)


class TestUrbanRagError:
    """Tests for UrbanRagError base class."""

    def test_instantiation_with_default_message(self) -> None:
        """Base error uses default code and message."""
        err = UrbanRagError()
        assert err.code == "internal"
        assert err.message == "An unexpected error occurred"
        assert err.trace_id is None

    def test_instantiation_with_custom_message(self) -> None:
        """Custom message overrides default."""
        err = UrbanRagError(message="Something went wrong")
        assert err.message == "Something went wrong"
        assert err.code == "internal"

    def test_instantiation_with_trace_id(self) -> None:
        """trace_id is stored and accessible."""
        err = UrbanRagError(trace_id="abc-123")
        assert err.trace_id == "abc-123"

    def test_instantiation_with_both_custom_message_and_trace_id(self) -> None:
        """Both message and trace_id can be set."""
        err = UrbanRagError(message="Custom error", trace_id="trace-456")
        assert err.message == "Custom error"
        assert err.trace_id == "trace-456"

    def test_to_dict_without_trace_id(self) -> None:
        """to_dict() returns code and message when no trace_id."""
        err = UrbanRagError(message="Boom")
        result = err.to_dict()
        assert result == {"code": "internal", "message": "Boom"}

    def test_to_dict_with_trace_id(self) -> None:
        """to_dict() includes trace_id when present."""
        err = UrbanRagError(message="Boom", trace_id="trace-789")
        result = err.to_dict()
        assert result == {"code": "internal", "message": "Boom", "trace_id": "trace-789"}

    def test_is_instance_of_exception(self) -> None:
        """UrbanRagError is a subclass of Exception."""
        err = UrbanRagError()
        assert isinstance(err, Exception)


class TestDocumentNotFoundError:
    """Tests for DocumentNotFoundError."""

    def test_default_code_and_message(self) -> None:
        """Uses document_not_found code and appropriate message."""
        err = DocumentNotFoundError()
        assert err.code == "document_not_found"
        assert err.message == "Document not found in corpus"

    def test_custom_message_override(self) -> None:
        """Custom message can override the default."""
        err = DocumentNotFoundError(message="Doc abc not in corpus")
        assert err.message == "Doc abc not in corpus"
        assert err.code == "document_not_found"

    def test_to_dict(self) -> None:
        """to_dict() returns correct structure."""
        err = DocumentNotFoundError(message="Not found", trace_id="t1")
        result = err.to_dict()
        assert result == {
            "code": "document_not_found",
            "message": "Not found",
            "trace_id": "t1",
        }


class TestValidationError:
    """Tests for ValidationError."""

    def test_default_code_and_message(self) -> None:
        """Uses validation_error code."""
        err = ValidationError()
        assert err.code == "validation_error"
        assert err.message == "Input validation failed"

    def test_custom_message(self) -> None:
        """Custom message overrides default."""
        err = ValidationError(message="Field x is required")
        assert err.message == "Field x is required"

    def test_to_dict(self) -> None:
        """to_dict() returns correct structure."""
        err = ValidationError(trace_id="t2")
        result = err.to_dict()
        assert result == {
            "code": "validation_error",
            "message": "Input validation failed",
            "trace_id": "t2",
        }


class TestIngestError:
    """Tests for IngestError."""

    def test_default_code_and_message(self) -> None:
        """Uses ingest_error code."""
        err = IngestError()
        assert err.code == "ingest_error"
        assert err.message == "Document ingestion failed"

    def test_custom_message(self) -> None:
        """Custom message overrides default."""
        err = IngestError(message="PDF parse failed at page 3")
        assert err.message == "PDF parse failed at page 3"

    def test_to_dict(self) -> None:
        """to_dict() returns correct structure."""
        err = IngestError(message="Ingest failed", trace_id="t3")
        result = err.to_dict()
        assert result == {
            "code": "ingest_error",
            "message": "Ingest failed",
            "trace_id": "t3",
        }


class TestEmbeddingError:
    """Tests for EmbeddingError."""

    def test_default_code_and_message(self) -> None:
        """Uses embedding_error code."""
        err = EmbeddingError()
        assert err.code == "embedding_error"
        assert err.message == "Embedding generation failed"

    def test_custom_message(self) -> None:
        """Custom message overrides default."""
        err = EmbeddingError(message="Model timeout on batch 2")
        assert err.message == "Model timeout on batch 2"

    def test_to_dict(self) -> None:
        """to_dict() returns correct structure."""
        err = EmbeddingError(trace_id="t4")
        result = err.to_dict()
        assert result == {
            "code": "embedding_error",
            "message": "Embedding generation failed",
            "trace_id": "t4",
        }


class TestRetrievalError:
    """Tests for RetrievalError."""

    def test_default_code_and_message(self) -> None:
        """Uses retrieval_error code."""
        err = RetrievalError()
        assert err.code == "retrieval_error"
        assert err.message == "Retrieval failed"

    def test_custom_message(self) -> None:
        """Custom message overrides default."""
        err = RetrievalError(message="Qdrant connection refused")
        assert err.message == "Qdrant connection refused"

    def test_to_dict(self) -> None:
        """to_dict() returns correct structure."""
        err = RetrievalError(message="Ret failed", trace_id="t5")
        result = err.to_dict()
        assert result == {
            "code": "retrieval_error",
            "message": "Ret failed",
            "trace_id": "t5",
        }


class TestGenerationError:
    """Tests for GenerationError."""

    def test_default_code_and_message(self) -> None:
        """Uses generation_error code."""
        err = GenerationError()
        assert err.code == "generation_error"
        assert err.message == "Answer generation failed"

    def test_custom_message(self) -> None:
        """Custom message overrides default."""
        err = GenerationError(message="Gemini API key invalid")
        assert err.message == "Gemini API key invalid"

    def test_to_dict(self) -> None:
        """to_dict() returns correct structure."""
        err = GenerationError(trace_id="t6")
        result = err.to_dict()
        assert result == {
            "code": "generation_error",
            "message": "Answer generation failed",
            "trace_id": "t6",
        }


class TestServiceUnavailableError:
    """Tests for ServiceUnavailableError."""

    def test_default_code_and_message(self) -> None:
        """Uses service_unavailable code."""
        err = ServiceUnavailableError()
        assert err.code == "service_unavailable"
        assert err.message == "Service unavailable"

    def test_custom_message(self) -> None:
        """Custom message overrides default."""
        err = ServiceUnavailableError(message="Qdrant is down")
        assert err.message == "Qdrant is down"

    def test_to_dict(self) -> None:
        """to_dict() returns correct structure."""
        err = ServiceUnavailableError(message="Service down", trace_id="t7")
        result = err.to_dict()
        assert result == {
            "code": "service_unavailable",
            "message": "Service down",
            "trace_id": "t7",
        }


class TestRateLimitError:
    """Tests for RateLimitError."""

    def test_default_code_and_message(self) -> None:
        """Uses rate_limited code."""
        err = RateLimitError()
        assert err.code == "rate_limited"
        assert err.message == "Rate limit exceeded"

    def test_custom_message(self) -> None:
        """Custom message overrides default."""
        err = RateLimitError(message="API quota exceeded for today")
        assert err.message == "API quota exceeded for today"

    def test_to_dict(self) -> None:
        """to_dict() returns correct structure."""
        err = RateLimitError(trace_id="t8")
        result = err.to_dict()
        assert result == {
            "code": "rate_limited",
            "message": "Rate limit exceeded",
            "trace_id": "t8",
        }


class TestErrorInheritance:
    """Tests verifying all errors inherit from UrbanRagError."""

    @pytest.mark.parametrize(
        "error_class",
        [
            DocumentNotFoundError,
            ValidationError,
            IngestError,
            EmbeddingError,
            RetrievalError,
            GenerationError,
            ServiceUnavailableError,
            RateLimitError,
        ],
    )
    def test_all_subclasses_inherit_from_urban_rag_error(
        self, error_class: type[UrbanRagError]
    ) -> None:
        """Every error subclass is a subclass of UrbanRagError."""
        assert issubclass(error_class, UrbanRagError)

    @pytest.mark.parametrize(
        "error_class",
        [
            DocumentNotFoundError,
            ValidationError,
            IngestError,
            EmbeddingError,
            RetrievalError,
            GenerationError,
            ServiceUnavailableError,
            RateLimitError,
        ],
    )
    def test_all_subclasses_are_exceptions(self, error_class: type[UrbanRagError]) -> None:
        """Every error subclass is a subclass of Exception."""
        assert issubclass(error_class, Exception)

    @pytest.mark.parametrize(
        "error_class",
        [
            DocumentNotFoundError,
            ValidationError,
            IngestError,
            EmbeddingError,
            RetrievalError,
            GenerationError,
            ServiceUnavailableError,
            RateLimitError,
        ],
    )
    def test_all_subclasses_have_distinct_code(self, error_class: type[UrbanRagError]) -> None:
        """Each subclass has its own distinct code."""
        err = error_class()
        assert err.code != "internal"
