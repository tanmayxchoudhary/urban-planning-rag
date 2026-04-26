"""GTE-ModernColBERT text encoder for parallel text retrieval channel.

This module implements the TextEmbedder class that encodes text chunks
using the GTE-ModernColBERT late-interaction model from lightonai.

Key features:
- Late-interaction (multi-vector) encoding per token
- CPU-friendly (no GPU required)
- Batch encoding of text chunks
- 768-dimensional embeddings per token

Reference: lightonai/GTE-ModernColBERT-v1 on HuggingFace
"""

from __future__ import annotations

from typing import Any

import structlog
import torch

from urban_rag.common.errors import EmbeddingError

logger = structlog.get_logger(__name__)

# Default model identifier
DEFAULT_MODEL_ID = "lightonai/GTE-ModernColBERT-v1"
# Embedding dimension for GTE-ModernColBERT (768 for ModernBERT base)
EMBEDDING_DIM = 768


class TextEmbedder:
    """GTE-ModernColBERT text embedding model loader and encoder.

    This class handles:
    - Lazy loading of the GTE-ModernColBERT model
    - CPU-based inference (GPU not needed for text encoding)
    - Batch encoding of text chunks with late-interaction per-token vectors

    Attributes:
        model_id: HuggingFace model identifier
        device: "cuda" or "cpu"
        is_loaded: Whether the model has been loaded
    """

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
    ) -> None:
        """Initialize the TextEmbedder.

        Args:
            model_id: HuggingFace model identifier.
                      Defaults to lightonai/GTE-ModernColBERT-v1.
            device: Device to run on. Defaults to "cuda" if available,
                    else "cpu".
        """
        from urban_rag.common.settings import get_settings

        settings = get_settings()
        self.model_id = model_id or settings.text_embed_model or DEFAULT_MODEL_ID

        if device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        self._model: Any = None  # Loaded lazily; type is transformers.AutoModel at runtime
        self._tokenizer: Any = None  # Loaded lazily; type is transformers.AutoTokenizer at runtime
        self._is_loaded = False

        logger.info(  # type: ignore
            "TextEmbedder initialized",
            model_id=self.model_id,  # type: ignore
            device=self._device,  # type: ignore
        )

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._is_loaded

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension (768 for GTE-ModernColBERT)."""
        return EMBEDDING_DIM

    @property
    def device(self) -> str:
        """Return the device the model is on."""
        return self._device

    def _load_model(self) -> None:
        """Load the GTE-ModernColBERT model and tokenizer.

        Raises:
            EmbeddingError: If model loading fails.
        """
        if self._is_loaded:
            return

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise EmbeddingError(
                f"Failed to import transformers: {e}. "
                "Please install with: pip install transformers"
            ) from e

        try:
            logger.info("Loading GTE-ModernColBERT model", model_id=self.model_id)  # type: ignore

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )

            # Load model
            self._model = AutoModel.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )

            # Move model to device
            self._model = self._model.to(self._device)  # type: ignore[reportCallIssue]
            self._model.eval()  # type: ignore[reportCallIssue]

            self._is_loaded = True
            logger.info("GTE-ModernColBERT model loaded successfully")

        except Exception as e:
            raise EmbeddingError(f"Failed to load GTE-ModernColBERT model: {e}") from e

    @torch.inference_mode()
    def embed_chunks(self, texts: list[str]) -> torch.Tensor:
        """Encode a batch of text chunks to multi-vector embeddings.

        Args:
            texts: List of text strings (text chunks).

        Returns:
            Tensor of shape (num_chunks, num_tokens, embedding_dim).
            The num_tokens dimension is variable per text due to different
            text lengths.

        Raises:
            EmbeddingError: If encoding fails or no texts provided.
        """
        if not texts:
            raise EmbeddingError("No text chunks provided for embedding")

        if not self._is_loaded:
            self._load_model()

        if self._model is None or self._tokenizer is None:
            raise EmbeddingError("Model or tokenizer not loaded")

        try:
            # Tokenize texts
            encoded = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            # Move inputs to device
            input_ids = encoded["input_ids"].to(self._device)
            attention_mask = encoded["attention_mask"].to(self._device)

            # Run model inference
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # Extract last hidden state (late-interaction multi-vector per token)
            # Shape: (batch_size, seq_len, hidden_dim)
            embeddings = outputs.last_hidden_state

            # Return on CPU for consistency with VisualEmbedder
            return embeddings.cpu()

        except Exception as e:
            raise EmbeddingError(f"Failed to encode text chunks: {e}") from e

    @torch.inference_mode()
    def embed_query(self, query: str) -> torch.Tensor:
        """Encode a text query to a multi-vector embedding.

        Args:
            query: The text query string.

        Returns:
            Tensor of shape (1, query_length, embedding_dim).
        """
        if not query:
            raise EmbeddingError("Empty query provided")

        if not self._is_loaded:
            self._load_model()

        try:
            if self._tokenizer is None or self._model is None:
                raise EmbeddingError("Model or tokenizer not loaded")

            # Tokenize query
            encoded = self._tokenizer(
                query,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            # Move to device
            input_ids = encoded["input_ids"].to(self._device)
            attention_mask = encoded["attention_mask"].to(self._device)

            # Run model
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            embeddings = outputs.last_hidden_state
            return embeddings.cpu()

        except Exception as e:
            raise EmbeddingError(f"Failed to encode query: {e}") from e

    def encode_text_batch(
        self, texts: list[str], batch_size: int = 32
    ) -> list[list[list[float]]]:
        """Encode a batch of texts and return as list-of-floats.

        This is a convenience method for the HTTP embed service.

        Args:
            texts: List of text strings.
            batch_size: Batch size for processing.

        Returns:
            List of multi-vector embeddings (list of list of list of floats).
        """
        all_embeddings: list[list[list[float]]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            tensor = self.embed_chunks(batch)
            # tensor shape: (batch_size, num_tokens, embedding_dim)
            for j in range(tensor.shape[0]):
                all_embeddings.append(tensor[j].tolist())

        return all_embeddings

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._is_loaded = False

        if self._device == "cuda":
            torch.cuda.empty_cache()

        logger.info("TextEmbedder model unloaded")


# Singleton instance for reuse across requests
_text_embedder_instance: TextEmbedder | None = None


def get_text_embedder() -> TextEmbedder:
    """Get the singleton TextEmbedder instance.

    Returns:
        The singleton TextEmbedder instance.
    """
    global _text_embedder_instance
    if _text_embedder_instance is None:
        _text_embedder_instance = TextEmbedder()
    return _text_embedder_instance


def release_text_embedder() -> None:
    """Release the singleton text embedder instance to free memory."""
    global _text_embedder_instance
    if _text_embedder_instance is not None:
        _text_embedder_instance.unload()
        _text_embedder_instance = None
