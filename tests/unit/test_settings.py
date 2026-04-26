"""Unit tests for Settings."""

from __future__ import annotations

import os

import pytest
from pydantic_core._pydantic_core import ValidationError

from urban_rag.common.settings import Settings, get_settings


def test_settings_loads_with_valid_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings loads successfully with valid GEMINI_API_KEY."""
    monkeypatch.setenv("GEMINI_API_KEY", "valid_test_key_12345")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")

    # Force reload by resetting singleton
    import urban_rag.common.settings as settings_module

    settings_module._settings = None
    s = Settings()

    assert s.gemini_api_key == "valid_test_key_12345"
    assert s.qdrant_url == "http://localhost:6333"


def test_settings_qdrant_url_has_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """QDRANT_URL has a default value."""
    monkeypatch.setenv("GEMINI_API_KEY", "valid_test_key_12345")
    monkeypatch.delenv("QDRANT_URL", raising=False)

    import urban_rag.common.settings as settings_module

    settings_module._settings = None
    s = Settings()

    assert s.qdrant_url == "http://localhost:6333"


def test_settings_gemini_api_key_validation_rejects_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GEMINI_API_KEY raises ValidationError if missing or placeholder."""
    monkeypatch.setenv("GEMINI_API_KEY", "your_gemini_api_key_here")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")

    import urban_rag.common.settings as settings_module

    settings_module._settings = None

    with pytest.raises(ValidationError, match="GEMINI_API_KEY is required"):
        Settings()


def test_settings_gemini_api_key_validation_rejects_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GEMINI_API_KEY raises ValidationError if empty."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")

    import urban_rag.common.settings as settings_module

    settings_module._settings = None

    with pytest.raises(ValidationError, match="GEMINI_API_KEY is required"):
        Settings()


def test_get_settings_returns_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_settings() returns the same singleton instance."""
    monkeypatch.setenv("GEMINI_API_KEY", "valid_test_key_12345")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")

    import urban_rag.common.settings as settings_module

    settings_module._settings = None

    s1 = get_settings()
    s2 = get_settings()

    assert s1 is s2
