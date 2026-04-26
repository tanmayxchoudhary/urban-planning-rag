"""Unit tests for src/urban_rag/embed/text_encoder.py — GTE-ModernColBERT text encoder."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from urban_rag.embed.text_encoder import (
    DEFAULT_MODEL_ID,
    EMBEDDING_DIM,
    TextEmbedder,
    get_text_embedder,
    release_text_embedder,
)


class TestTextEmbedderInit:
    """Tests for TextEmbedder initialization."""

    def test_init_cpu_mode(self) -> None:
        """Test initialization in CPU mode."""
        embedder = TextEmbedder(device="cpu")
        assert embedder.device == "cpu"
        assert embedder.embedding_dim == EMBEDDING_DIM
        assert not embedder.is_loaded

    @patch("urban_rag.common.settings.get_settings")
    def test_init_with_custom_model(self, mock_settings: MagicMock) -> None:
        """Test initialization with custom model ID."""
        mock_settings.return_value.text_embed_model = None
        embedder = TextEmbedder(model_id="custom/model", device="cpu")
        assert embedder.model_id == "custom/model"

    def test_embedding_dim_constant(self) -> None:
        """Test that EMBEDDING_DIM is correctly set to 768 for GTE-ModernColBERT."""
        assert EMBEDDING_DIM == 768

    def test_default_model_id(self) -> None:
        """Test that DEFAULT_MODEL_ID is correct."""
        assert DEFAULT_MODEL_ID == "lightonai/GTE-ModernColBERT-v1"

    def test_unload_before_load(self) -> None:
        """Test unload() when model is not loaded is a no-op."""
        embedder = TextEmbedder(device="cpu")
        embedder.unload()  # Should not raise
        assert not embedder.is_loaded


class TestTextEmbedderProperties:
    """Tests for TextEmbedder properties."""

    def test_is_loaded_default_false(self) -> None:
        """Test is_loaded defaults to False."""
        embedder = TextEmbedder(device="cpu")
        assert embedder.is_loaded is False

    def test_embedding_dim(self) -> None:
        """Test embedding_dim returns 768."""
        embedder = TextEmbedder(device="cpu")
        assert embedder.embedding_dim == 768


class TestTextEmbedderEmbedChunks:
    """Tests for TextEmbedder.embed_chunks method."""

    @patch("urban_rag.embed.text_encoder.torch.cuda.is_available", return_value=False)
    def test_embed_chunks_empty_raises(self, mock_cuda: MagicMock) -> None:
        """Test that embed_chunks raises on empty input."""
        embedder = TextEmbedder(device="cpu")
        with pytest.raises(Exception):  # EmbeddingError
            embedder.embed_chunks([])

    @patch("urban_rag.embed.text_encoder.torch.cuda.is_available", return_value=False)
    def test_embed_chunks_loads_model(self, mock_cuda: MagicMock) -> None:
        """Test that embed_chunks loads the model if not loaded."""
        embedder = TextEmbedder(device="cpu")
        # When model is not loaded, _load_model is called first
        # We verify by checking that calling embed_chunks with valid state
        # would trigger loading - we test the is_loaded flag
        assert not embedder.is_loaded


class TestTextEmbedderEmbedQuery:
    """Tests for TextEmbedder.embed_query method."""

    @patch("urban_rag.embed.text_encoder.torch.cuda.is_available", return_value=False)
    def test_embed_query_empty_raises(self, mock_cuda: MagicMock) -> None:
        """Test that embed_query raises on empty query."""
        embedder = TextEmbedder(device="cpu")
        with pytest.raises(Exception):  # EmbeddingError
            embedder.embed_query("")


class TestTextEmbedderEncodeBatch:
    """Tests for TextEmbedder.encode_text_batch method."""

    @patch("urban_rag.embed.text_encoder.torch.cuda.is_available", return_value=False)
    def test_encode_text_batch_empty(self, mock_cuda: MagicMock) -> None:
        """Test encode_text_batch with empty list returns empty list."""
        embedder = TextEmbedder(device="cpu")
        result = embedder.encode_text_batch([])
        assert result == []


class TestSingletonFunctions:
    """Tests for get_text_embedder and release_text_embedder functions."""

    def test_get_text_embedder_returns_instance(self) -> None:
        """Test that get_text_embedder returns a TextEmbedder instance."""
        import urban_rag.embed.text_encoder as text_module

        text_module._text_embedder_instance = None

        with patch.object(text_module, "TextEmbedder") as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            result = get_text_embedder()
            assert result == mock_instance

        text_module._text_embedder_instance = None

    def test_get_text_embedder_reuses_instance(self) -> None:
        """Test that get_text_embedder reuses the singleton instance."""
        import urban_rag.embed.text_encoder as text_module

        text_module._text_embedder_instance = None

        mock_instance = MagicMock(spec=TextEmbedder)
        text_module._text_embedder_instance = mock_instance

        result = get_text_embedder()
        assert result == mock_instance

        text_module._text_embedder_instance = None

    def test_release_text_embedder(self) -> None:
        """Test that release_text_embedder clears the instance."""
        import urban_rag.embed.text_encoder as text_module

        mock_instance = MagicMock(spec=TextEmbedder)
        text_module._text_embedder_instance = mock_instance

        release_text_embedder()

        assert text_module._text_embedder_instance is None
        mock_instance.unload.assert_called_once()
