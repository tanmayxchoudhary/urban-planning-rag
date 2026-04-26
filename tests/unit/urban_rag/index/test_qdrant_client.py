"""Unit tests for index/qdrant_client.py — collection bootstrap and schema."""

from __future__ import annotations

import os
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

# Ensure test environment has required settings
os.environ.setdefault("GEMINI_API_KEY", "test-api-key-for-unit-tests")


class TestQdrantCollectionBootstrap:
    """Tests for Qdrant collection creation and idempotency."""

    @pytest.fixture(autouse=True)
    def reset_settings(self) -> None:
        """Reset the settings singleton before each test."""
        import urban_rag.common.settings as settings_module

        settings_module._settings = None
        yield
        settings_module._settings = None

    @pytest.fixture(autouse=True)
    def reset_client(self) -> None:
        """Reset the qdrant_client singleton before each test."""
        import urban_rag.index.qdrant_client as qc_module

        qc_module._client = None
        yield
        qc_module._client = None

    @pytest.fixture
    def mock_qdrant_instance(self) -> MagicMock:
        """Create a mock Qdrant client instance."""
        mock = MagicMock()
        mock.get_collection.return_value = None
        mock.collection_exists.return_value = False
        return mock

    def test_pages_visual_collection_has_multivector_config(
        self, mock_qdrant_instance: MagicMock
    ) -> None:
        """pages_visual collection uses multivector config with MAX_SIM comparator."""
        from qdrant_client import models

        with ExitStack() as stack:
            stack.enter_context(
                patch.dict(os.environ, {"GEMINI_API_KEY": "test-api-key-for-unit-tests"})
            )
            stack.enter_context(
                patch(
                    "urban_rag.index.qdrant_client.QdrantClient",
                    return_value=mock_qdrant_instance,
                )
            )
            from urban_rag.index.qdrant_client import (
                COLLECTION_PAGES_VISUAL,
                create_collections,
                get_qdrant_client,
            )

            client = get_qdrant_client()
            create_collections(client)

            # Verify create_collection was called for pages_visual
            calls = mock_qdrant_instance.create_collection.call_args_list
            visual_call = None
            for call in calls:
                if call.kwargs.get("collection_name") == COLLECTION_PAGES_VISUAL:
                    visual_call = call
                    break

            assert visual_call is not None, (
                f"pages_visual collection was not created. Calls: {calls}"
            )

            # Check multivector config
            vectors_config = visual_call.kwargs["vectors_config"]
            assert "patches" in vectors_config
            patches_cfg = vectors_config["patches"]
            assert isinstance(patches_cfg, models.VectorParams)
            assert patches_cfg.multivector_config is not None
            assert (
                patches_cfg.multivector_config.comparator
                == models.MultiVectorComparator.MAX_SIM
            )

    def test_pages_text_collection_has_text_embeddings(
        self, mock_qdrant_instance: MagicMock
    ) -> None:
        """pages_text collection has text embeddings (768-dim for GTE-ModernColBERT)."""
        from qdrant_client import models

        with ExitStack() as stack:
            stack.enter_context(
                patch.dict(os.environ, {"GEMINI_API_KEY": "test-api-key-for-unit-tests"})
            )
            stack.enter_context(
                patch(
                    "urban_rag.index.qdrant_client.QdrantClient",
                    return_value=mock_qdrant_instance,
                )
            )
            from urban_rag.index.qdrant_client import (
                COLLECTION_PAGES_TEXT,
                create_collections,
                get_qdrant_client,
            )

            client = get_qdrant_client()
            create_collections(client)

            calls = mock_qdrant_instance.create_collection.call_args_list
            text_call = None
            for call in calls:
                if call.kwargs.get("collection_name") == COLLECTION_PAGES_TEXT:
                    text_call = call
                    break

            assert text_call is not None, "pages_text collection was not created"
            vectors_config = text_call.kwargs["vectors_config"]
            assert "text" in vectors_config
            text_cfg = vectors_config["text"]
            assert isinstance(text_cfg, models.VectorParams)
            assert text_cfg.size == 768

    def test_scalar_quantization_configured(
        self, mock_qdrant_instance: MagicMock
    ) -> None:
        """pages_visual has scalar quantization (INT8, 0.99 quantile, always_ram)."""
        from qdrant_client import models

        with ExitStack() as stack:
            stack.enter_context(
                patch.dict(os.environ, {"GEMINI_API_KEY": "test-api-key-for-unit-tests"})
            )
            stack.enter_context(
                patch(
                    "urban_rag.index.qdrant_client.QdrantClient",
                    return_value=mock_qdrant_instance,
                )
            )
            from urban_rag.index.qdrant_client import (
                COLLECTION_PAGES_VISUAL,
                create_collections,
                get_qdrant_client,
            )

            client = get_qdrant_client()
            create_collections(client)

            calls = mock_qdrant_instance.create_collection.call_args_list
            visual_call = None
            for call in calls:
                if call.kwargs.get("collection_name") == COLLECTION_PAGES_VISUAL:
                    visual_call = call
                    break

            assert visual_call is not None
            qc = visual_call.kwargs["quantization_config"]
            assert isinstance(qc, models.ScalarQuantization)
            assert isinstance(qc.scalar, models.ScalarQuantizationConfig)
            assert qc.scalar.type == models.ScalarType.INT8
            assert qc.scalar.quantile == 0.99
            assert qc.scalar.always_ram is True

    def test_collection_creation_is_idempotent(
        self, mock_qdrant_instance: MagicMock
    ) -> None:
        """Re-running create_collections does not raise — collection exists is a no-op."""
        with ExitStack() as stack:
            stack.enter_context(
                patch.dict(os.environ, {"GEMINI_API_KEY": "test-api-key-for-unit-tests"})
            )
            stack.enter_context(
                patch(
                    "urban_rag.index.qdrant_client.QdrantClient",
                    return_value=mock_qdrant_instance,
                )
            )
            from urban_rag.index.qdrant_client import (
                create_collections,
                get_qdrant_client,
            )

            mock_qdrant_instance.collection_exists.return_value = True

            client = get_qdrant_client()
            create_collections(client)

            assert mock_qdrant_instance.create_collection.call_count == 0

    def test_idempotent_when_collection_exists(
        self, mock_qdrant_instance: MagicMock
    ) -> None:
        """If collection exists, create_collections skips it (no-op)."""
        with ExitStack() as stack:
            stack.enter_context(
                patch.dict(os.environ, {"GEMINI_API_KEY": "test-api-key-for-unit-tests"})
            )
            stack.enter_context(
                patch(
                    "urban_rag.index.qdrant_client.QdrantClient",
                    return_value=mock_qdrant_instance,
                )
            )
            from urban_rag.index.qdrant_client import (
                create_collections,
                get_qdrant_client,
            )

            mock_qdrant_instance.collection_exists.side_effect = lambda name, **_: True

            client = get_qdrant_client()
            create_collections(client)

            assert mock_qdrant_instance.create_collection.call_count == 0

    def test_visual_collection_uses_correct_embedding_dim(
        self, mock_qdrant_instance: MagicMock
    ) -> None:
        """pages_visual patches vector uses 128-dim (actual ColQwen2.5 output)."""
        with ExitStack() as stack:
            stack.enter_context(
                patch.dict(os.environ, {"GEMINI_API_KEY": "test-api-key-for-unit-tests"})
            )
            stack.enter_context(
                patch(
                    "urban_rag.index.qdrant_client.QdrantClient",
                    return_value=mock_qdrant_instance,
                )
            )
            from urban_rag.index.qdrant_client import (
                COLLECTION_PAGES_VISUAL,
                create_collections,
                get_qdrant_client,
            )

            client = get_qdrant_client()
            create_collections(client)

            calls = mock_qdrant_instance.create_collection.call_args_list
            visual_call = None
            for call in calls:
                if call.kwargs.get("collection_name") == COLLECTION_PAGES_VISUAL:
                    visual_call = call
                    break

            assert visual_call is not None
            vectors_config = visual_call.kwargs["vectors_config"]
            assert vectors_config["patches"].size == 128
            assert vectors_config["pooled"].size == 128

    def test_client_singleton(self) -> None:
        """get_qdrant_client returns the same instance on repeated calls."""
        with ExitStack() as stack:
            stack.enter_context(
                patch.dict(os.environ, {"GEMINI_API_KEY": "test-api-key-for-unit-tests"})
            )
            mock_cls = stack.enter_context(
                patch("urban_rag.index.qdrant_client.QdrantClient")
            )
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance

            from urban_rag.index.qdrant_client import get_qdrant_client

            client1 = get_qdrant_client()
            client2 = get_qdrant_client()

            assert client1 is client2
            assert mock_cls.call_count == 1
