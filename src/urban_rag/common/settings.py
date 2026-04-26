"""Pydantic settings for the Urban RAG application."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized settings for Urban RAG.

    All configuration flows through this class. No ad-hoc os.environ elsewhere.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "urban-rag"
    app_version: str = "0.1.0"
    debug: bool = False

    # API server
    api_host: str = "0.0.0.0"  # noqa: S104
    api_port: int = 3100

    # Corpus paths
    docs_dir: str = "data/docs"
    manifest_path: str = "data/manifest.parquet"

    # Embedding
    embed_model: str = "vidore/colqwen2.5-v0.2"
    text_embed_model: str = "lightonai/GTE-ModernColBERT-v1"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_visual: str = "pages_visual"
    qdrant_collection_text: str = "pages_text"

    # VLM Generation
    gemini_model: str = "gemini-2.5-flash"
    gemini_api_key: str | None = None

    # Logging
    log_level: str = "INFO"

    # Telemetry
    otel_endpoint: str | None = None

    def log_config(self) -> dict:
        """Return logging configuration dict."""
        return {
            "log_level": self.log_level,
            "version": self.app_version,
        }


# Global singleton — created once at import time.
_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the global Settings singleton, instantiating it on first call."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
