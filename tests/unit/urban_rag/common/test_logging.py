"""Unit tests for common/logging.py — JSON structured logging with correlation IDs."""

from __future__ import annotations

import io
import json
import os
import sys

# Set required environment variable before importing the module under test
os.environ.setdefault("GEMINI_API_KEY", "test-api-key-for-unit-tests")

from urban_rag.common import configure_logging, get_logger


class TestConfigureLogging:
    """Tests for configure_logging()."""

    def test_configure_logging_binds_context_variables(self) -> None:
        """configure_logging binds corpus_version and service to all log events."""
        # Capture stdout to inspect log output
        old_stdout = sys.stdout
        captured = io.StringIO()
        sys.stdout = captured

        try:
            configure_logging(corpus_version="v1.2.3", service="test-svc")
            logger = get_logger()
            logger.info("test event")

            output = captured.getvalue()
            event = json.loads(output.strip())

            assert "corpus_version" in event
            assert event["corpus_version"] == "v1.2.3"
            assert "service" in event
            assert event["service"] == "test-svc"
            assert "query_id" in event
        finally:
            sys.stdout = old_stdout

    def test_configure_logging_sets_log_level(self) -> None:
        """configure_logging accepts corpus_version and service without error."""
        # Should not raise
        configure_logging(corpus_version="test", service="cli")
        logger = get_logger()
        assert logger is not None


class TestGetLogger:
    """Tests for get_logger()."""

    def test_get_logger_returns_structlog_logger(self) -> None:
        """get_logger returns a structlog logger (BoundLoggerLazyProxy)."""
        logger = get_logger()
        assert logger is not None
        # structlog.get_logger() returns a BoundLoggerLazyProxy, not directly a PrintLogger
        assert hasattr(logger, "info")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")

    def test_get_logger_binds_query_id(self) -> None:
        """get_logger accepts query_id and binds it to log events."""
        old_stdout = sys.stdout
        captured = io.StringIO()
        sys.stdout = captured

        try:
            configure_logging(corpus_version="v1.0", service="test")
            logger = get_logger(query_id="q-abc-123")
            logger.info("test event")

            output = captured.getvalue()
            event = json.loads(output.strip())

            assert "query_id" in event
            assert event["query_id"] == "q-abc-123"
        finally:
            sys.stdout = old_stdout

    def test_get_logger_binds_corpus_version(self) -> None:
        """get_logger accepts corpus_version override and binds it to log events."""
        old_stdout = sys.stdout
        captured = io.StringIO()
        sys.stdout = captured

        try:
            configure_logging(corpus_version="default", service="test")
            logger = get_logger(corpus_version="v2.0.0")
            logger.info("test event")

            output = captured.getvalue()
            event = json.loads(output.strip())

            assert "corpus_version" in event
            assert event["corpus_version"] == "v2.0.0"
        finally:
            sys.stdout = old_stdout

    def test_get_logger_binds_service(self) -> None:
        """get_logger accepts service override and binds it to log events."""
        old_stdout = sys.stdout
        captured = io.StringIO()
        sys.stdout = captured

        try:
            configure_logging(corpus_version="v1.0", service="default-svc")
            logger = get_logger(service="api-svc")
            logger.info("test event")

            output = captured.getvalue()
            event = json.loads(output.strip())

            assert "service" in event
            assert event["service"] == "api-svc"
        finally:
            sys.stdout = old_stdout

    def test_get_logger_binds_extra_fields(self) -> None:
        """get_logger accepts **extra fields and binds them to log events."""
        old_stdout = sys.stdout
        captured = io.StringIO()
        sys.stdout = captured

        try:
            configure_logging(corpus_version="v1.0", service="test")
            logger = get_logger(extra_field="custom_value", count=42)
            logger.info("test event")

            output = captured.getvalue()
            event = json.loads(output.strip())

            assert "extra_field" in event
            assert event["extra_field"] == "custom_value"
            assert "count" in event
            assert event["count"] == 42
        finally:
            sys.stdout = old_stdout

    def test_json_output_contains_required_fields_together(self) -> None:
        """Log event JSON contains query_id, corpus_version, and service together."""
        old_stdout = sys.stdout
        captured = io.StringIO()
        sys.stdout = captured

        try:
            configure_logging(corpus_version="v1.0.0", service="cli")
            logger = get_logger(query_id="test-q-id")
            logger.info("correlation test")

            output = captured.getvalue()
            event = json.loads(output.strip())

            # All three required fields must be present
            assert "query_id" in event, "query_id missing from JSON log"
            assert "corpus_version" in event, "corpus_version missing from JSON log"
            assert "service" in event, "service missing from JSON log"

            # Verify expected values
            assert event["query_id"] == "test-q-id"
            assert event["corpus_version"] == "v1.0.0"
            assert event["service"] == "cli"
        finally:
            sys.stdout = old_stdout
