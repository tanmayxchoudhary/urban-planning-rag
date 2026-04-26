"""Generation orchestrator - orchestrates answer synthesis from retrieval results.

This module implements the generation orchestration per PLAN.md Part VIII §8.1-§8.7:
    - Takes a query and RetrievalResult from the retrieve orchestrator
    - Builds answer prompts from candidates
    - Streams tokens from Gemini with image grounding
    - Validates citations against candidate set
    - Determines confidence (high/medium/low)
    - Emits Answer events in strict order per API contract

Event order per VAL-API-007:
    [1] retrieval_started    → caller already has this, we just validate input
    [2] retrieval_completed  → validate candidates are present
    [3] generation_started  → before first token
    [4] token (repeated)    → streaming tokens from Gemini
    [5] generation_completed → final answer with citations, confidence, diagnostics
    [6] done                → terminal event

Returns:
    AsyncIterator[AnswerEvent] — all events are typed dataclasses
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import AsyncGenerator, Literal

import structlog

from urban_rag.common.types import (
    AnswerDiagnostics,
    Citation,
    RetrievalCandidate,
    RetrievalResult,
)
from urban_rag.generate.gemini import generate_stream
from urban_rag.generate.prompts import (
    AnswerMode,
    build_answer_prompt,
    build_citations_list,
    extract_citations_from_answer,
    validate_citations,
)

logger = structlog.get_logger(__name__, service="generate-orchestrator")

# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnswerEvent:
    """Base event for answer generation stream."""

    query_id: str


@dataclass(frozen=True)
class RetrievalStartedEvent(AnswerEvent):
    """Event emitted when retrieval stage begins (validation only)."""



@dataclass(frozen=True)
class RetrievalCompletedEvent(AnswerEvent):
    """Event emitted when retrieval is validated and candidates are confirmed.

    Per VAL-API-008: carries candidates array and latency_ms.
    """

    candidates: list[RetrievalCandidate]
    latency_ms: int


@dataclass(frozen=True)
class GenerationStartedEvent(AnswerEvent):
    """Event emitted before first generation token."""

    mode: Literal["fast", "deep"]


@dataclass(frozen=True)
class TokenEvent(AnswerEvent):
    """A single token from the generation stream."""

    chunk: str


@dataclass(frozen=True)
class GenerationCompletedEvent(AnswerEvent):
    """Final generation event with complete answer object.

    Per VAL-API-009: includes answer object with answer_markdown,
    citations[], confidence, diagnostics, and query_id.
    """

    answer_markdown: str
    citations: list[Citation]
    confidence: Literal["high", "medium", "low"]
    diagnostics: AnswerDiagnostics


@dataclass(frozen=True)
class RefusedEvent(AnswerEvent):
    """Event emitted when query is refused."""

    reason: str
    message: str


@dataclass(frozen=True)
class ErrorEvent(AnswerEvent):
    """Event emitted on generation error."""

    code: str
    message: str
    stage: Literal["retrieval", "generation"] = "generation"


@dataclass(frozen=True)
class DoneEvent(AnswerEvent):
    """Terminal event — stream is complete."""



# Union type for the async iterator return
AnswerStreamEvent = (
    RetrievalStartedEvent
    | RetrievalCompletedEvent
    | GenerationStartedEvent
    | TokenEvent
    | GenerationCompletedEvent
    | RefusedEvent
    | ErrorEvent
    | DoneEvent
)


# ---------------------------------------------------------------------------
# Main answer orchestrator
# ---------------------------------------------------------------------------


async def answer(
    query: str,
    retrieval_result: RetrievalResult,
    mode: Literal["fast", "deep"] = "fast",
    max_output_tokens: int = 8192,
    temperature: float = 0.3,
) -> AsyncGenerator[AnswerStreamEvent, None]:
    """Generate a grounded answer from retrieval results with streaming events.

    This is the single entrypoint for answer generation. It:
    1. Validates retrieval result has candidates
    2. Builds answer prompts from candidates
    3. Streams tokens from Gemini with image grounding
    4. Validates citations against candidate set
    5. Determines confidence level
    6. Emits events in strict order per VAL-API-007

    Per PART VIII §8.1-§8.7:
        - Strict grounding in retrieved pages
        - [N] citation markers validated against candidates
        - Confidence is one of high/medium/low
        - Deep mode: self-critique pass before final answer

    Args:
        query: The user's natural language question.
        retrieval_result: Result from the retrieve orchestrator.
        mode: Generation mode — "fast" (Gemini 2.5 Flash) or "deep" (Flash + self-critique).
        max_output_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.

    Yields:
        AnswerStreamEvent in strict order:
        - retrieval_started (validation)
        - retrieval_completed (candidates confirmed)
        - generation_started
        - token (repeated)
        - generation_completed (with full answer)
        - done

    Citation contract per VAL-API-031:
        Each citation includes: page_id, doc_id, page_num, doc_title,
        section_title, score — all resolving to page_ids in retrieval_result.
    """
    # Generate a query_id for this answer session
    query_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    logger.info(
        "answer_generation_start",
        query_id=query_id,
        query=query[:100],
        mode=mode,
        candidates_count=len(retrieval_result.candidates),
    )

    # ── Emit: retrieval_started ──────────────────────────────────────────
    yield RetrievalStartedEvent(query_id=query_id)

    # ── Validate candidates ───────────────────────────────────────────────
    if not retrieval_result.candidates:
        logger.warning("no_candidates_for_answer", query_id=query_id)
        yield RefusedEvent(
            query_id=query_id,
            reason="insufficient_context",
            message=(
                "The retrieved pages do not contain sufficient information "
                "to answer this query. Please try a more specific question."
            ),
        )
        yield DoneEvent(query_id=query_id)
        return

    retrieval_latency_ms = retrieval_result.latency_ms

    # ── Emit: retrieval_completed ─────────────────────────────────────────
    yield RetrievalCompletedEvent(
        query_id=query_id,
        candidates=retrieval_result.candidates,
        latency_ms=retrieval_latency_ms,
    )

    # ── Build answer prompts ───────────────────────────────────────────────
    answer_mode = AnswerMode.DEEP if mode == "deep" else AnswerMode.FAST

    try:
        system_prompt, user_prompt, _template_version = build_answer_prompt(
            question=query,
            candidates=retrieval_result.candidates,
            mode=answer_mode,
        )
    except ValueError as e:
        logger.error("prompt_build_failed", query_id=query_id, error=str(e))
        yield ErrorEvent(
            query_id=query_id,
            code="generation_error",
            message=f"Failed to build answer prompt: {e}",
            stage="generation",
        )
        yield DoneEvent(query_id=query_id)
        return

    # ── Emit: generation_started ──────────────────────────────────────────
    yield GenerationStartedEvent(query_id=query_id, mode=mode)

    # ── Stream tokens from Gemini ──────────────────────────────────────────
    generation_start = time.perf_counter()
    full_text = ""

    # Collect image URIs from candidates for visual grounding
    image_uris = [
        c.page_image_uri for c in retrieval_result.candidates if c.page_image_uri
    ]

    token_count = 0
    error_occurred = False
    error_message = ""

    async for event in generate_stream(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_uris=image_uris if image_uris else None,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    ):
        if event.is_error:
            error_occurred = True
            error_message = event.error or "Unknown generation error"
            break

        if event.chunk:
            token_count += 1
            yield TokenEvent(query_id=query_id, chunk=event.chunk)
            full_text += event.chunk

    generation_latency_ms = int((time.perf_counter() - generation_start) * 1000)

    if error_occurred:
        logger.error(
            "generation_stream_error",
            query_id=query_id,
            error=error_message,
        )
        yield ErrorEvent(
            query_id=query_id,
            code="generation_failed",
            message=error_message,
            stage="generation",
        )
        yield DoneEvent(query_id=query_id)
        return

    # ── Validate and fix citations ────────────────────────────────────────
    corrected_text, invalid_indices = validate_citations(
        full_text, retrieval_result.candidates
    )

    # Extract cited indices from the (possibly corrected) text
    cited_indices = extract_citations_from_answer(corrected_text)

    # Build citations list
    citations_list = build_citations_list(retrieval_result.candidates, cited_indices)

    # Build Citation objects per VAL-API-031
    citations: list[Citation] = []
    for c in citations_list:
        citations.append(Citation(
            idx=c["idx"],
            doc_id=c["doc_id"],
            page_id=c["page_id"],
            page_num=c["page_num"],
            doc_title=c["doc_title"],
            section_title=c["section_title"],
            score=c["score"],
        ))

    # ── Determine confidence ───────────────────────────────────────────────
    # Use the same logic as gemini._determine_confidence but also consider
    # invalid (hallucinated) citations as a negative signal
    confidence = _determine_confidence(
        answer_text=corrected_text,
        cited_indices=cited_indices,
        total_candidates=len(retrieval_result.candidates),
        has_invalid_citations=len(invalid_indices) > 0,
    )

    # ── Build diagnostics ──────────────────────────────────────────────────
    total_latency_ms = int((time.perf_counter() - start_time) * 1000)

    diagnostics = AnswerDiagnostics(
        latency_ms={
            "retrieve": retrieval_latency_ms,
            "generate": generation_latency_ms,
            "total": total_latency_ms,
        },
        backends={
            "generate": "gemini-2.5-flash",
        },
        candidate_count={
            "fused": len(retrieval_result.candidates),
            "cited": len(cited_indices),
        },
        flags=dict(retrieval_result.flags),
    )

    # ── Emit: generation_completed ─────────────────────────────────────────
    logger.info(
        "answer_generation_complete",
        query_id=query_id,
        answer_length=len(corrected_text),
        citations=len(citations),
        confidence=confidence,
        invalid_citations=len(invalid_indices),
        total_latency_ms=total_latency_ms,
    )

    yield GenerationCompletedEvent(
        query_id=query_id,
        answer_markdown=corrected_text,
        citations=citations,
        confidence=confidence,
        diagnostics=diagnostics,
    )

    # ── Emit: done ─────────────────────────────────────────────────────────
    yield DoneEvent(query_id=query_id)


# ---------------------------------------------------------------------------
# Confidence determination
# ---------------------------------------------------------------------------


def _determine_confidence(
    answer_text: str,
    cited_indices: list[int],
    total_candidates: int,
    has_invalid_citations: bool = False,
) -> Literal["high", "medium", "low"]:
    """Determine confidence level based on answer characteristics.

    Per VAL-API-029: confidence is strictly one of high|medium|low.

    Confidence rules:
        - high: substantial answer (>200 chars) with good citation coverage (>=30%)
        - medium: answer exists and is not clearly low confidence
        - low: refusal phrases, short text, or hallucinated citations

    Args:
        answer_text: The generated answer text.
        cited_indices: List of cited candidate indices.
        total_candidates: Total number of candidates available.
        has_invalid_citations: True if any citations were hallucinated.

    Returns:
        Confidence level: "high", "medium", or "low".
    """
    # Hallucinated citations immediately reduce confidence
    if has_invalid_citations:
        return "low"

    # Refusal patterns
    refusal_phrases = [
        "cannot be answered",
        "not in the corpus",
        "i don't know",
        "no information",
        "outside the scope",
        "insufficient information",
        "cannot answer",
    ]

    answer_lower = answer_text.lower()
    for phrase in refusal_phrases:
        if phrase in answer_lower:
            return "low"

    # Check for minimal content
    if len(answer_text.strip()) < 50:
        return "low"

    # Check citation coverage
    citation_ratio = len(cited_indices) / max(total_candidates, 1)
    has_citations = len(cited_indices) > 0

    # High confidence: good citation coverage and substantial answer
    if has_citations and citation_ratio >= 0.3 and len(answer_text) > 200:
        return "high"

    # Medium: some citations or substantial answer without citations
    if has_citations or len(answer_text) > 150:
        return "medium"

    return "low"


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def smoke_test() -> dict:
    """Run a smoke test of the generation orchestrator.

    Verifies:
        - _determine_confidence handles all cases correctly
        - Event types are properly structured
        - Citation validation works end-to-end

    Returns:
        Dict with smoke test results.
    """
    from urban_rag.common.types import RetrievalCandidate

    # Test confidence determination
    assert _determine_confidence("", [], 5) == "low"
    assert _determine_confidence("I cannot answer.", [], 5) == "low"
    assert _determine_confidence("FSI is 2.5.", [], 5) == "low"
    assert _determine_confidence("FSI is 2.5.", [1], 5) == "low"  # short text
    assert _determine_confidence("A" * 300, [1, 2], 3) == "high"
    assert _determine_confidence("B" * 250, [], 5) == "medium"

    # Hallucinated citations → low
    assert _determine_confidence(
        "Answer with valid text.", [1, 2], 2, has_invalid_citations=True
    ) == "low"

    # Create synthetic candidates for citation validation test
    candidates = [
        RetrievalCandidate(
            page_id="doc1#p001",
            score=0.8,
            channel_scores={"visual": 10.0},
            channel_ranks={"visual": 1},
            page_image_uri="s3://doc1/p001.png",
            extracted_text_excerpt="FSI for residential zones is 2.5",
            section_title="Section 3.2",
        ),
        RetrievalCandidate(
            page_id="doc2#p010",
            score=0.75,
            channel_scores={"visual": 9.0},
            channel_ranks={"visual": 2},
            page_image_uri="s3://doc2/p010.png",
            extracted_text_excerpt="Parking requirements: 1 per 100 sqm",
            section_title="Section 5.1",
        ),
    ]

    # Test citation validation
    test_answer = "The FSI is 2.5 [1] and parking is 1 per 100 sqm [2]. Unknown [99]."
    corrected, invalid = validate_citations(test_answer, candidates)
    assert corrected == "The FSI is 2.5 [1] and parking is 1 per 100 sqm [2]. Unknown [?]."
    assert 99 in invalid

    # Test event types are frozen dataclasses
    e = TokenEvent(query_id="test-id", chunk="Hello")
    assert e.query_id == "test-id"
    assert e.chunk == "Hello"

    # Test DoneEvent
    done = DoneEvent(query_id="test-id")
    assert done.query_id == "test-id"

    logger.info("generate_orchestrator_smoke_test_passed")

    return {
        "passed": True,
        "confidence_logic_valid": True,
        "citation_validation_valid": True,
        "event_types_valid": True,
    }
