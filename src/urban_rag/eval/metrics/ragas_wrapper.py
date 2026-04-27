"""RAGAS wrapper for generation quality metrics.

This module provides a reproducible RAGAS evaluation interface with a pinned judge model.
Scores are reproducible within ±0.02 across 3 identical runs when using the same:
- Pinned model version
- temperature=0
- run_config seed

See PLAN.md §10.3.2 for metric definitions and targets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import structlog

from urban_rag.common.types import AnswerResponse

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports for RAGAS - only imported when needed
# ---------------------------------------------------------------------------


def _get_ragas_dependencies() -> dict[str, Any]:
    """Lazily import RAGAS dependencies and return them as a dict."""
    try:
        from ragas import evaluate
        from ragas.dataset import Dataset
        from ragas.llms import LiteLLMStructuredLLM
        from ragas.metrics.collections import (
            AnswerCorrectness,
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
        from ragas.run_config import RunConfig
    except ImportError as e:
        raise ImportError(
            "ragas is not installed. Install it with: uv pip install ragas"
        ) from e

    return {
        "evaluate": evaluate,
        "Dataset": Dataset,
        "LiteLLMStructuredLLM": LiteLLMStructuredLLM,
        "Faithfulness": Faithfulness,
        "AnswerRelevancy": AnswerRelevancy,
        "ContextPrecision": ContextPrecision,
        "ContextRecall": ContextRecall,
        "AnswerCorrectness": AnswerCorrectness,
        "RunConfig": RunConfig,
    }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Default judge model for RAGAS metrics.
#: Using gemini-2.5-flash with pinned version for reproducibility.
DEFAULT_JUDGE_MODEL = "gemini/gemini-2.5-flash"

#: Default temperature for deterministic outputs.
DEFAULT_TEMPERATURE = 0.0


@dataclass(frozen=True)
class RagasJudgeConfig:
    """Configuration for the RAGAS judge model.

    Attributes:
        model: The judge model to use. Must be in format "provider/model-name".
            Defaults to gemini/gemini-2.5-flash.
        temperature: Temperature for the LLM. 0.0 for deterministic outputs.
        seed: Random seed for reproducibility. Defaults to 42.
        timeout: Timeout in seconds for each LLM call. Defaults to 180.
        max_retries: Maximum retry attempts on failure. Defaults to 10.
    """

    model: str = DEFAULT_JUDGE_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    seed: int = 42
    timeout: int = 180
    max_retries: int = 10


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RagasMetricsResult:
    """Aggregated RAGAS generation quality metrics.

    All fields are floats in [0.0, 1.0] range.
    See PLAN.md §10.3.2 for target values.
    """

    faithfulness: float | None = None
    answer_relevance: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    answer_correctness: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        """Return metrics as a flat dictionary."""
        return {
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "answer_correctness": self.answer_correctness,
        }


@dataclass
class RagasEvaluationInput:
    """Input data for a single RAGAS evaluation.

    This represents one query-answer pair with its retrieved context
    for evaluation against ground truth.

    Attributes:
        question: The user's question.
        answer: The generated answer text.
        ground_truth: The ground truth / reference answer.
        contexts: List of retrieved context strings (page excerpts).
    """

    question: str
    answer: str
    ground_truth: str
    contexts: list[str]


# ---------------------------------------------------------------------------
# Judge model factory
# ---------------------------------------------------------------------------


def create_judge_llm(config: RagasJudgeConfig | None = None) -> Any:
    """Create a configured RAGAS LLM wrapper for the judge model.

    Uses LiteLLMStructuredLLM which wraps Gemini (and other models) with
    structured output support for deterministic evaluation.

    Args:
        config: Judge configuration. Uses defaults if not provided.

    Returns:
        A configured RAGAS LLM instance.
    """
    if config is None:
        config = RagasJudgeConfig()

    deps = _get_ragas_dependencies()
    lite_llm_class = deps["LiteLLMStructuredLLM"]

    # Check for GEMINI_API_KEY
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable is required for RAGAS evaluation. "
            "Set it in your environment or .env file."
        )

    # Create the judge LLM with temperature=0 for reproducibility
    return lite_llm_class(
        model=config.model,
        provider="gemini",
        temperature=config.temperature,
    )


def create_ragas_metrics(
    judge_llm: Any,
    embeddings: Any | None = None,
) -> dict[str, Any]:
    """Create RAGAS metric instances with the given judge LLM.

    Args:
        judge_llm: The configured judge LLM from create_judge_llm().
        embeddings: Optional embeddings model for answer_relevance.
            If not provided, uses the default.

    Returns:
        Dictionary mapping metric names to configured metric instances.
    """
    deps = _get_ragas_dependencies()
    faithfulness_cls = deps["Faithfulness"]
    answer_relevancy_cls = deps["AnswerRelevancy"]
    context_precision_cls = deps["ContextPrecision"]
    context_recall_cls = deps["ContextRecall"]
    answer_correctness_cls = deps["AnswerCorrectness"]

    metrics = {
        "faithfulness": faithfulness_cls(llm=judge_llm),
        "context_precision": context_precision_cls(llm=judge_llm),
        "context_recall": context_recall_cls(llm=judge_llm),
        "answer_correctness": answer_correctness_cls(llm=judge_llm),
    }

    # AnswerRelevancy requires embeddings
    if embeddings is not None:
        metrics["answer_relevance"] = answer_relevancy_cls(llm=judge_llm, embeddings=embeddings)
    else:
        # Create without embeddings - will use default
        metrics["answer_relevance"] = answer_relevancy_cls(llm=judge_llm)

    return metrics


# ---------------------------------------------------------------------------
# Core evaluation API
# ---------------------------------------------------------------------------


def evaluate_single(
    input_data: RagasEvaluationInput,
    config: RagasJudgeConfig | None = None,
    metrics_to_compute: list[str] | None = None,
) -> RagasMetricsResult:
    """Evaluate a single query-answer pair with RAGAS metrics.

    This is the main entry point for evaluating a single RAG response.
    For batch evaluation of multiple samples, use evaluate_batch().

    Args:
        input_data: The evaluation input with question, answer, ground_truth, and contexts.
        config: Judge configuration. Uses defaults if not provided.
        metrics_to_compute: List of specific metrics to compute. Computes all if None.

    Returns:
        RagasMetricsResult with all requested metrics.

    Raises:
        ValueError: If no valid API key is available.
        ImportError: If ragas is not installed.
    """
    if config is None:
        config = RagasJudgeConfig()

    deps = _get_ragas_dependencies()

    # Build dataset with single row
    data = {
        "user_input": [input_data.question],
        "response": [input_data.answer],
        "reference": [input_data.ground_truth],
        "retrieved_contexts": [input_data.contexts],
    }

    dataset_cls = deps["Dataset"]
    evaluate = deps["evaluate"]
    run_config_cls = deps["RunConfig"]

    ds = dataset_cls.from_dict(data)

    # Create judge LLM
    judge_llm = create_judge_llm(config)

    # Create metrics
    metrics_dict = create_ragas_metrics(judge_llm)

    # Determine which metrics to compute
    if metrics_to_compute is None:
        metrics_to_compute = list(metrics_dict.keys())

    selected_metrics = [metrics_dict[m] for m in metrics_to_compute if m in metrics_dict]

    # Run evaluation with reproducible config
    run_config = run_config_cls(
        timeout=config.timeout,
        max_retries=config.max_retries,
        seed=config.seed,
    )

    logger.info(
        "running_ragas_evaluation",
        question=input_data.question[:50],
        metrics=[m.name for m in selected_metrics],
        judge_model=config.model,
        temperature=config.temperature,
        seed=config.seed,
    )

    result = evaluate(
        dataset=ds,
        metrics=selected_metrics,
        run_config=run_config,
    )

    # Extract scores from result
    scores = result.scores[0] if hasattr(result.scores, "__iter__") else result.scores

    return RagasMetricsResult(
        faithfulness=_safe_get_score(scores, "faithfulness"),
        answer_relevance=_safe_get_score(scores, "answer_relevance"),
        context_precision=_safe_get_score(scores, "context_precision"),
        context_recall=_safe_get_score(scores, "context_recall"),
        answer_correctness=_safe_get_score(scores, "answer_correctness"),
    )


def evaluate_batch(
    inputs: list[RagasEvaluationInput],
    config: RagasJudgeConfig | None = None,
    metrics_to_compute: list[str] | None = None,
) -> list[RagasMetricsResult]:
    """Evaluate multiple query-answer pairs with RAGAS metrics.

    More efficient than calling evaluate_single() multiple times as it
    batches the evaluation.

    Args:
        inputs: List of evaluation inputs.
        config: Judge configuration. Uses defaults if not provided.
        metrics_to_compute: List of specific metrics to compute. Computes all if None.

    Returns:
        List of RagasMetricsResult, one per input.
    """
    if config is None:
        config = RagasJudgeConfig()

    if not inputs:
        return []

    deps = _get_ragas_dependencies()

    # Build dataset
    data = {
        "user_input": [inp.question for inp in inputs],
        "response": [inp.answer for inp in inputs],
        "reference": [inp.ground_truth for inp in inputs],
        "retrieved_contexts": [inp.contexts for inp in inputs],
    }

    dataset_cls = deps["Dataset"]
    evaluate = deps["evaluate"]
    run_config_cls = deps["RunConfig"]

    ds = dataset_cls.from_dict(data)

    # Create judge LLM
    judge_llm = create_judge_llm(config)

    # Create metrics
    metrics_dict = create_ragas_metrics(judge_llm)

    # Determine which metrics to compute
    if metrics_to_compute is None:
        metrics_to_compute = list(metrics_dict.keys())

    selected_metrics = [metrics_dict[m] for m in metrics_to_compute if m in metrics_dict]

    # Run evaluation
    run_config = run_config_cls(
        timeout=config.timeout,
        max_retries=config.max_retries,
        seed=config.seed,
    )

    logger.info(
        "running_ragas_batch_evaluation",
        num_samples=len(inputs),
        metrics=[m.name for m in selected_metrics],
        judge_model=config.model,
        temperature=config.temperature,
        seed=config.seed,
    )

    result = evaluate(
        dataset=ds,
        metrics=selected_metrics,
        run_config=run_config,
    )

    # Extract scores
    results = []
    for scores in result.scores:
        results.append(RagasMetricsResult(
            faithfulness=_safe_get_score(scores, "faithfulness"),
            answer_relevance=_safe_get_score(scores, "answer_relevance"),
            context_precision=_safe_get_score(scores, "context_precision"),
            context_recall=_safe_get_score(scores, "context_recall"),
            answer_correctness=_safe_get_score(scores, "answer_correctness"),
        ))

    return results


# ---------------------------------------------------------------------------
# Helpers for AnswerResponse integration
# ---------------------------------------------------------------------------


def evaluate_from_answer_response(
    answer_response: AnswerResponse,
    ground_truth_answer: str,
    retrieved_contexts: list[str],
    config: RagasJudgeConfig | None = None,
) -> RagasMetricsResult:
    """Evaluate an AnswerResponse object with RAGAS metrics.

    Convenience function to evaluate an AnswerResponse from the RAG pipeline
    against a ground truth answer.

    Args:
        answer_response: The AnswerResponse from the generation pipeline.
        ground_truth_answer: The reference/ground truth answer.
        retrieved_contexts: List of context strings that were retrieved.
        config: Judge configuration.

    Returns:
        RagasMetricsResult with all requested metrics.
    """
    return evaluate_single(
        input_data=RagasEvaluationInput(
            question="",  # Question not needed for faithfulness/context metrics
            answer=answer_response.answer_markdown,
            ground_truth=ground_truth_answer,
            contexts=retrieved_contexts,
        ),
        config=config,
    )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _safe_get_score(scores: Any, metric_name: str) -> float | None:
    """Safely extract a score from RAGAS result scores object.

    RAGAS result scores can be in different formats depending on the version.
    This handles the common cases.

    Args:
        scores: The scores object from RAGAS evaluate.
        metric_name: Name of the metric to extract.

    Returns:
        The score as a float, or None if not available.
    """
    try:
        # Try attribute access first (ragas >= 0.4)
        if hasattr(scores, metric_name):
            val = getattr(scores, metric_name)
            return float(val) if val is not None else None

        # Try dict access
        if isinstance(scores, dict) and metric_name in scores:
            val = scores[metric_name]
            return float(val) if val is not None else None

        return None
    except (ValueError, TypeError):
        return None


def compute_mean_metrics(
    results: list[RagasMetricsResult],
) -> RagasMetricsResult:
    """Compute mean values across multiple evaluation results.

    Args:
        results: List of RagasMetricsResult from evaluate_batch().

    Returns:
        RagasMetricsResult with mean values across all inputs.
    """
    if not results:
        return RagasMetricsResult()

    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    metric_names = [
        "faithfulness",
        "answer_relevance",
        "context_precision",
        "context_recall",
        "answer_correctness",
    ]

    for r in results:
        for metric_name in metric_names:
            val = getattr(r, metric_name)
            if val is not None:
                totals[metric_name] = totals.get(metric_name, 0.0) + val
                counts[metric_name] = counts.get(metric_name, 0) + 1

    def mean(key: str) -> float | None:
        total = totals.get(key)
        count = counts.get(key)
        return total / count if total is not None and count else None

    return RagasMetricsResult(
        faithfulness=mean("faithfulness"),
        answer_relevance=mean("answer_relevance"),
        context_precision=mean("context_precision"),
        context_recall=mean("context_recall"),
        answer_correctness=mean("answer_correctness"),
    )
