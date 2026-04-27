"""Unit tests for RAGAS wrapper module.

Tests cover:
- Configuration dataclasses
- Evaluation input/output types
- Reproducibility settings
- Mean computation
- Integration with AnswerResponse types

Note: Full RAGAS evaluation requires GEMINI_API_KEY and makes actual API calls.
These tests focus on unit-level behavior and type correctness.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from urban_rag.eval.metrics.ragas_wrapper import (
    RagasEvaluationInput,
    RagasJudgeConfig,
    RagasMetricsResult,
    compute_mean_metrics,
    create_judge_llm,
    _safe_get_score,
)


# ---------------------------------------------------------------------------
# RagasJudgeConfig tests
# ---------------------------------------------------------------------------


class TestRagasJudgeConfig:
    """Tests for RagasJudgeConfig dataclass."""

    def test_default_values(self):
        """Default config has sensible defaults for reproducibility."""
        config = RagasJudgeConfig()
        assert config.model == "gemini/gemini-2.5-flash"
        assert config.temperature == 0.0
        assert config.seed == 42
        assert config.timeout == 180
        assert config.max_retries == 10

    def test_custom_values(self):
        """Custom config overrides defaults."""
        config = RagasJudgeConfig(
            model="gemini/gemini-2.0-flash",
            temperature=0.5,
            seed=123,
            timeout=60,
            max_retries=5,
        )
        assert config.model == "gemini/gemini-2.0-flash"
        assert config.temperature == 0.5
        assert config.seed == 123
        assert config.timeout == 60
        assert config.max_retries == 5

    def test_immutability(self):
        """Config is frozen and immutable."""
        config = RagasJudgeConfig()
        with pytest.raises(Exception):  # frozen dataclass raises error on mutation
            config.temperature = 0.5


# ---------------------------------------------------------------------------
# RagasMetricsResult tests
# ---------------------------------------------------------------------------


class TestRagasMetricsResult:
    """Tests for RagasMetricsResult dataclass."""

    def test_defaults(self):
        """All metrics default to None."""
        result = RagasMetricsResult()
        assert result.faithfulness is None
        assert result.answer_relevance is None
        assert result.context_precision is None
        assert result.context_recall is None
        assert result.answer_correctness is None

    def test_with_values(self):
        """All metrics can be set."""
        result = RagasMetricsResult(
            faithfulness=0.95,
            answer_relevance=0.88,
            context_precision=0.82,
            context_recall=0.91,
            answer_correctness=0.87,
        )
        assert result.faithfulness == 0.95
        assert result.answer_relevance == 0.88
        assert result.context_precision == 0.82
        assert result.context_recall == 0.91
        assert result.answer_correctness == 0.87

    def test_as_dict(self):
        """as_dict returns all metrics."""
        result = RagasMetricsResult(
            faithfulness=0.95,
            answer_relevance=0.88,
        )
        d = result.as_dict()
        assert "faithfulness" in d
        assert "answer_relevance" in d
        assert "context_precision" in d
        assert "context_recall" in d
        assert "answer_correctness" in d
        assert d["faithfulness"] == 0.95
        assert d["answer_relevance"] == 0.88
        assert d["context_precision"] is None


# ---------------------------------------------------------------------------
# RagasEvaluationInput tests
# ---------------------------------------------------------------------------


class TestRagasEvaluationInput:
    """Tests for RagasEvaluationInput dataclass."""

    def test_required_fields(self):
        """All required fields are accepted."""
        inp = RagasEvaluationInput(
            question="What is FSI for residential zones?",
            answer="The FSI for residential zones is 2.5.",
            ground_truth="FSI for residential is typically 2.0-3.0 depending on the zone.",
            contexts=["Page 1: FSI table shows 2.5 for residential.", "Page 2: Zone classification details."],
        )
        assert inp.question == "What is FSI for residential zones?"
        assert inp.answer == "The FSI for residential zones is 2.5."
        assert len(inp.contexts) == 2

    def test_empty_contexts(self):
        """Contexts can be empty."""
        inp = RagasEvaluationInput(
            question="Test question",
            answer="Test answer",
            ground_truth="Ground truth",
            contexts=[],
        )
        assert inp.contexts == []


# ---------------------------------------------------------------------------
# create_judge_llm tests
# ---------------------------------------------------------------------------


class TestCreateJudgeLlm:
    """Tests for create_judge_llm factory function."""

    def test_requires_api_key(self, monkeypatch):
        """Raises ValueError when GEMINI_API_KEY is not set."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            create_judge_llm()

    def test_creates_llm_with_default_config(self, monkeypatch):
        """Creates LLM with default config when API key is set."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("urban_rag.eval.metrics.ragas_wrapper._get_ragas_dependencies") as mock_deps:
            mock_llm_class = MagicMock()
            mock_deps.return_value = {
                "LiteLLMStructuredLLM": mock_llm_class,
            }

            create_judge_llm()

            mock_llm_class.assert_called_once_with(
                model="gemini/gemini-2.5-flash",
                provider="gemini",
                temperature=0.0,
            )

    def test_creates_llm_with_custom_config(self, monkeypatch):
        """Creates LLM with custom config."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with patch("urban_rag.eval.metrics.ragas_wrapper._get_ragas_dependencies") as mock_deps:
            mock_llm_class = MagicMock()
            mock_deps.return_value = {
                "LiteLLMStructuredLLM": mock_llm_class,
            }

            config = RagasJudgeConfig(
                model="gemini/gemini-2.0-pro",
                temperature=0.3,
                seed=999,
            )
            create_judge_llm(config)

            mock_llm_class.assert_called_once_with(
                model="gemini/gemini-2.0-pro",
                provider="gemini",
                temperature=0.3,
            )


# ---------------------------------------------------------------------------
# compute_mean_metrics tests
# ---------------------------------------------------------------------------


class TestComputeMeanMetrics:
    """Tests for compute_mean_metrics aggregation function."""

    def test_empty_list(self):
        """Empty results returns empty result."""
        result = compute_mean_metrics([])
        assert result.faithfulness is None

    def test_single_result(self):
        """Single result returns same values."""
        input_result = RagasMetricsResult(
            faithfulness=0.9,
            answer_relevance=0.8,
            context_precision=0.7,
            context_recall=0.85,
            answer_correctness=0.88,
        )
        result = compute_mean_metrics([input_result])
        assert result.faithfulness == 0.9
        assert result.answer_relevance == 0.8

    def test_multiple_results(self):
        """Multiple results are averaged."""
        results = [
            RagasMetricsResult(faithfulness=0.9, answer_relevance=0.8),
            RagasMetricsResult(faithfulness=0.94, answer_relevance=0.82),
            RagasMetricsResult(faithfulness=0.86, answer_relevance=0.78),
        ]
        result = compute_mean_metrics(results)
        assert result.faithfulness == pytest.approx((0.9 + 0.94 + 0.86) / 3)
        assert result.answer_relevance == pytest.approx((0.8 + 0.82 + 0.78) / 3)

    def test_handles_none_values(self):
        """None values are skipped in averaging."""
        results = [
            RagasMetricsResult(faithfulness=0.9, answer_relevance=None),
            RagasMetricsResult(faithfulness=None, answer_relevance=0.8),
            RagasMetricsResult(faithfulness=0.86, answer_relevance=0.78),
        ]
        result = compute_mean_metrics(results)
        # faithfulness: 0.9 and 0.86 from results 0 and 2, mean = (0.9 + 0.86) / 2 = 0.88
        assert result.faithfulness == pytest.approx((0.9 + 0.86) / 2)
        # answer_relevance: 0.8 and 0.78 from results 1 and 2, mean = (0.8 + 0.78) / 2 = 0.79
        assert result.answer_relevance == pytest.approx((0.8 + 0.78) / 2)


# ---------------------------------------------------------------------------
# _safe_get_score tests
# ---------------------------------------------------------------------------


class TestSafeGetScore:
    """Tests for _safe_get_score helper."""

    def test_attribute_access(self):
        """Works with attribute access."""
        mock_scores = MagicMock()
        mock_scores.faithfulness = 0.95
        mock_scores.answer_relevance = 0.88

        assert _safe_get_score(mock_scores, "faithfulness") == 0.95
        assert _safe_get_score(mock_scores, "answer_relevance") == 0.88

    def test_dict_access(self):
        """Works with dict-like access."""
        scores = {
            "faithfulness": 0.95,
            "answer_relevance": 0.88,
        }
        assert _safe_get_score(scores, "faithfulness") == 0.95

    def test_missing_key(self):
        """Returns None for missing keys."""
        # Use a dict to test missing key since MagicMock has tricky attribute behavior
        scores = {"faithfulness": 0.95}
        assert _safe_get_score(scores, "nonexistent") is None

    def test_none_value(self):
        """Returns None for None values."""
        mock_scores = MagicMock()
        mock_scores.faithfulness = None

        assert _safe_get_score(mock_scores, "faithfulness") is None

    def test_invalid_conversion(self):
        """Returns None for values that can't be converted."""
        mock_scores = MagicMock()
        mock_scores.faithfulness = "not a number"

        assert _safe_get_score(mock_scores, "faithfulness") is None


# ---------------------------------------------------------------------------
# Reproducibility configuration tests
# ---------------------------------------------------------------------------


class TestReproducibilityConfig:
    """Tests for reproducibility settings in RAGAS wrapper."""

    def test_temperature_zero_for_determinism(self):
        """Default temperature is 0.0 for deterministic outputs."""
        config = RagasJudgeConfig()
        assert config.temperature == 0.0

    def test_seed_for_reproducibility(self):
        """Default seed is set for reproducibility."""
        config = RagasJudgeConfig()
        assert config.seed == 42

    def test_same_config_produces_same_hash(self):
        """Same config values produce equivalent configs."""
        config1 = RagasJudgeConfig(model="gemini/gemini-2.5-flash", seed=42)
        config2 = RagasJudgeConfig(model="gemini/gemini-2.5-flash", seed=42)
        # Frozen dataclass equality works by value
        assert config1 == config2

    def test_different_seed_different_config(self):
        """Different seeds produce different configs."""
        config1 = RagasJudgeConfig(seed=42)
        config2 = RagasJudgeConfig(seed=123)
        assert config1 != config2


# ---------------------------------------------------------------------------
# Integration with AnswerResponse
# ---------------------------------------------------------------------------


class TestEvaluateFromAnswerResponse:
    """Tests for evaluate_from_answer_response convenience function."""

    def test_import_AnswerResponse(self):
        """Can import AnswerResponse from common.types."""
        from urban_rag.common.types import AnswerResponse, AnswerDiagnostics, Citation

        # Verify AnswerResponse has the expected structure via model_fields
        fields = AnswerResponse.model_fields
        assert "answer_markdown" in fields
        assert "citations" in fields
        assert "confidence" in fields
        assert "diagnostics" in fields
        assert "query_id" in fields

    def test_citation_structure(self):
        """Citation type has required fields."""
        from urban_rag.common.types import Citation

        citation = Citation(
            idx=1,
            doc_id="doc123",
            page_id="doc123#p0001",
            page_num=1,
        )
        assert citation.idx == 1
        assert citation.doc_id == "doc123"
