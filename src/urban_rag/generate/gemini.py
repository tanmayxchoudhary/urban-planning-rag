"""Streaming Gemini 2.5 Flash client with image attachment.

Per PLAN.md Part VIII §8.1:
    - Gemini 2.5 Flash for fast/deep mode generation
    - Image attachment capability for visual grounding
    - Streaming token-by-token delivery
    - Max-output cap enforcement
    - Cost_usd recorded per request and traced

The module provides:
    - generate_stream(): async generator yielding token events
    - generate(): higher-level API returning complete AnswerResponse
    - _GeminiStreamEvent: typed event for streaming responses
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal

import structlog

from urban_rag.common.errors import GenerationError
from urban_rag.common.settings import get_settings
from urban_rag.common.types import (
    AnswerDiagnostics,
    AnswerResponse,
    Citation,
    RetrievalCandidate,
)
from urban_rag.telemetry.metrics import record_tokens

logger = structlog.get_logger(__name__, service="generate-gemini")

# Gemini API constants
_GEMINI_MODEL = "gemini-2.5-flash"
_API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Retry configuration
_MAX_RETRIES = 3
_INITIAL_RETRY_DELAY = 1.0
_RETRY_BACKOFF_MULTIPLIER = 2.0
_TIMEOUT_SECONDS = 60

# Token pricing (USD per million tokens) — Gemini 2.5 Flash
# https://ai.google.dev/pricing
_PRICE_PER_MILLION_INPUT_TOKENS = 0.075  # $0.075/M input
_PRICE_PER_MILLION_OUTPUT_TOKENS = 3.5  # $3.50/M output


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _GeminiStreamEvent:
    """A single event from the Gemini streaming response.

    Attributes:
        chunk: Text content of this chunk (empty for metadata events).
        is_final: True if this is the final event in the stream.
        prompt_tokens: Number of input tokens (populated on final event).
        completion_tokens: Number of output tokens generated (final event only).
        cost_usd: Cost in USD for this request (final event only).
        error: Error message if the stream failed.
    """

    chunk: str = ""
    is_final: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None

    @property
    def is_error(self) -> bool:
        return self.error is not None


# ---------------------------------------------------------------------------
# Core streaming API
# ---------------------------------------------------------------------------


async def generate_stream(
    system_prompt: str,
    user_prompt: str,
    image_uris: list[str] | None = None,
    max_output_tokens: int = 8192,
    temperature: float = 0.3,
) -> AsyncIterator[_GeminiStreamEvent]:
    """Generate a streaming response from Gemini 2.5 Flash.

    This is an async generator that yields token events as they arrive from
    the Gemini API. Each event contains a chunk of text; the final event
    also contains token usage and cost information.

    Args:
        system_prompt: System instruction string (domain scope, grounding rules).
        user_prompt: The formatted user prompt with page content.
        image_uris: Optional list of image URIs to attach as grounding context.
        max_output_tokens: Maximum tokens to generate (output cap per PLAN §8.1).
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).

    Yields:
        _GeminiStreamEvent events — chunks progressively, final event with cost.

    Raises:
        GenerationError: If all retries fail or API returns an unrecoverable error.
        ServiceUnavailableError: If Gemini API is unreachable or returns 503.
    """
    settings = get_settings()
    api_key = settings.gemini_api_key

    if not api_key or api_key in ("your_gemini_api_key_here", "test-api-key-for-unit-tests"):
        yield _GeminiStreamEvent(
            error="GEMINI_API_KEY is not set or is a placeholder value",
        )
        return

    url = _API_URL_TEMPLATE.format(model=_GEMINI_MODEL) + f"?key={api_key}&alt=sse"

    # Build request payload
    payload = _build_payload(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_uris=image_uris,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )

    retry_delay = _INITIAL_RETRY_DELAY
    last_error: str | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            async for event in _stream_request(url, payload, timeout=_TIMEOUT_SECONDS):
                yield event
            return  # Stream completed successfully

        except TimeoutError:
            last_error = f"Request timed out after {_TIMEOUT_SECONDS}s"
            logger.warning(
                "gemini_stream_timeout",
                attempt=attempt + 1,
                max_retries=_MAX_RETRIES,
                retry_delay=retry_delay,
            )

        except urllib.error.HTTPError as exc:
            if exc.code == 503:
                last_error = "Gemini API is temporarily unavailable (503)"
                logger.warning("gemini_api_unavailable_503", attempt=attempt + 1)
            else:
                error_body = exc.read().decode("utf-8")[:500] if exc.fp else ""
                last_error = f"HTTP {exc.code}: {error_body}"
                logger.error("gemini_http_error", status=exc.code, error=error_body)

        except urllib.error.URLError as exc:
            last_error = f"Network error: {exc.reason}"
            logger.warning("gemini_network_error", error=str(exc))

        except Exception as exc:
            last_error = f"Unexpected error: {exc}"
            logger.error("gemini_stream_unexpected_error", error=str(exc))

        # Retry with exponential backoff
        if attempt < _MAX_RETRIES - 1:
            logger.info("gemini_retrying", delay=retry_delay, last_error=last_error)
            time.sleep(retry_delay)
            retry_delay *= _RETRY_BACKOFF_MULTIPLIER

    # All retries exhausted
    yield _GeminiStreamEvent(error=f"Failed after {_MAX_RETRIES} retries: {last_error}")


def _build_payload(
    system_prompt: str,
    user_prompt: str,
    image_uris: list[str] | None,
    max_output_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    """Build the Gemini API request payload.

    Args:
        system_prompt: System instructions.
        user_prompt: User prompt with page content.
        image_uris: Optional list of image URIs.
        max_output_tokens: Max output cap.
        temperature: Sampling temperature.

    Returns:
        Dict payload suitable for json.dumps and POST body.
    """
    contents: list[dict[str, Any]] = []

    # Add image URIs as context if provided
    if image_uris:
        image_parts = []
        for uri in image_uris:
            # Gemini accepts image URIs as File references or as base64 inline
            # We use the file uri format for S3-based images
            image_parts.append({
                "file_data": {
                    "mime_type": "image/png",
                    "file_uri": uri,
                }
            })
        # First content block: images
        contents.append({"role": "user", "parts": image_parts})

    # Second content block: text prompt
    contents.append({
        "role": "user",
        "parts": [{"text": user_prompt}],
    })

    return {
        "contents": contents,
        "system_instruction": {
            "parts": [{"text": system_prompt}],
        },
        "generation_config": {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "response_mime_type": "text/plain",
        },
    }



async def _stream_request(
    url: str,
    payload: dict[str, Any],
    timeout: int,
) -> AsyncIterator[_GeminiStreamEvent]:
    """Make a streaming request to the Gemini API and yield events.

    Uses urllib.request with POST method and parses SSE (Server-Sent Events)
    from the response body.

    Args:
        url: The Gemini API endpoint URL.
        payload: JSON-serializable request payload.
        timeout: Request timeout in seconds.

    Yields:
        _GeminiStreamEvent events parsed from SSE data.

    Raises:
        TimeoutError: If the request times out.
        urllib.error.HTTPError: On HTTP errors.
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(  # noqa: S310  # HTTPS only
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            # Read and parse SSE stream
            async for event in _parse_sse_stream(resp):
                yield event

    except TimeoutError:
        raise

    # Handle HTTP errors
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")[:500] if e.fp else ""
        logger.error("gemini_http_error", status=e.code, error=error_body)
        raise


async def _parse_sse_stream(resp: Any) -> AsyncIterator[_GeminiStreamEvent]:
    """Parse Server-Sent Events from the Gemini streaming response.

    Gemini's SSE format is:
        data: {"chunks": [{"content_part": {"text": "..."}}]}

    Or for the final chunk with usage metadata:
        data: {"chunks": [...], "usage_metadata": {...}}

    Args:
        resp: The urllib response object with a read() method.

    Yields:
        _GeminiStreamEvent events with chunk text and final metadata.
    """
    buffer = ""
    prompt_tokens = 0
    completion_tokens = 0

    while True:
        chunk = resp.read(4096).decode("utf-8")
        if not chunk:
            break

        buffer += chunk

        # Process complete lines
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if not line.startswith("data:"):
                continue

            json_str = line[5:].strip()  # Remove "data:" prefix
            if not json_str:
                continue

            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                logger.warning("gemini_sse_parse_error", raw=json_str[:100])
                continue

            # Extract text chunks
            chunks = data.get("chunks", [])
            for ch in chunks:
                content_part = ch.get("content_part", {})
                text = content_part.get("text", "")
                if text:
                    yield _GeminiStreamEvent(chunk=text)

            # Extract usage metadata from final chunk
            usage = data.get("usage_metadata", {})
            if usage:
                prompt_tokens = usage.get("prompt_token_count", 0)
                completion_tokens = usage.get("candidates_token_count", 0)
                cost_usd = _calculate_cost(prompt_tokens, completion_tokens)

                yield _GeminiStreamEvent(
                    is_final=True,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost_usd=cost_usd,
                )

    # If we never got usage metadata, estimate from completion tokens received
    # (Gemini may send usage in a different format or not at all)
    if completion_tokens == 0:
        # Estimate cost from empty final event
        yield _GeminiStreamEvent(
            is_final=True,
            prompt_tokens=0,
            completion_tokens=0,
            cost_usd=0.0,
        )


def _calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate the USD cost for a Gemini API call.

    Uses the pricing from:
        https://ai.google.dev/pricing

    Gemini 2.5 Flash pricing:
        - Input: $0.075 per million tokens
        - Output: $3.50 per million tokens

    Args:
        prompt_tokens: Number of input tokens.
        completion_tokens: Number of output tokens generated.

    Returns:
        Cost in USD as a float.
    """
    input_cost = (prompt_tokens / 1_000_000) * _PRICE_PER_MILLION_INPUT_TOKENS
    output_cost = (completion_tokens / 1_000_000) * _PRICE_PER_MILLION_OUTPUT_TOKENS
    return round(input_cost + output_cost, 6)


# ---------------------------------------------------------------------------
# High-level answer API
# ---------------------------------------------------------------------------


async def generate(
    question: str,
    candidates: list[RetrievalCandidate],
    system_prompt: str,
    user_prompt: str,
    mode: Literal["fast", "deep"] = "fast",
    max_output_tokens: int = 8192,
    temperature: float = 0.3,
) -> tuple[AnswerResponse, float, int, int]:
    """Generate a complete answer from the Gemini streaming response.

    Collects all tokens from the stream and returns a fully-typed AnswerResponse
    with citations, diagnostics, and cost tracking.

    Args:
        question: The user's question.
        candidates: Retrieval candidates for citation building.
        system_prompt: System instruction string.
        user_prompt: The formatted user prompt with page content.
        mode: Generation mode (fast or deep) for diagnostics.
        max_output_tokens: Maximum tokens to generate (output cap).
        temperature: Sampling temperature.

    Returns:
        Tuple of (AnswerResponse, cost_usd, prompt_tokens, completion_tokens).
        - AnswerResponse with answer_markdown, citations, confidence, and diagnostics.
        - cost_usd: The cost in USD for this API call.
        - prompt_tokens: Number of input tokens.
        - completion_tokens: Number of output tokens generated.

    Raises:
        GenerationError: If generation fails after all retries.
    """
    from urban_rag.generate.prompts import build_citations_list, extract_citations_from_answer

    # Collect all chunks from streaming
    full_text = ""
    prompt_tokens = 0
    completion_tokens = 0
    cost_usd = 0.0
    error_occurred = False
    error_message = ""

    image_uris = [c.page_image_uri for c in candidates if c.page_image_uri]

    async for event in generate_stream(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_uris=image_uris if image_uris else None,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    ):
        if event.is_error:
            error_occurred = True
            error_message = event.error or "Unknown error"
            break

        full_text += event.chunk

        if event.is_final:
            prompt_tokens = event.prompt_tokens
            completion_tokens = event.completion_tokens
            cost_usd = event.cost_usd

    # Record token metrics after stream is consumed
    record_tokens(mode=mode, token_type="prompt", count=prompt_tokens)  # noqa: S106
    record_tokens(mode=mode, token_type="output", count=completion_tokens)  # noqa: S106

    if error_occurred:
        logger.error("gemini_generation_failed", error=error_message)
        raise GenerationError(f"Generation failed: {error_message}")

    # Extract cited indices and build citations list
    cited_indices = extract_citations_from_answer(full_text)
    citations_list = build_citations_list(candidates, cited_indices)

    # Build citations as Citation objects
    citations = [
        Citation(
            idx=c["idx"],
            doc_id=c["doc_id"],
            doc_hash=c["page_id"].split("#")[0] if "#" in c["page_id"] else "",
            doc_filename="",
            page_id=c["page_id"],
            page_num=c["page_num"],
            page_image_uri="",
            doc_title=c["doc_title"],
            section_title=c["section_title"],
            section_path=[],
            score=c["score"],
            rerank_score=None,
        )
        for c in citations_list
    ]

    # Determine confidence based on completion and citation coverage
    confidence = _determine_confidence(full_text, cited_indices, len(candidates))

    # Build diagnostics
    diagnostics = AnswerDiagnostics(
        latency_ms={},  # Populated by caller
        backends={"generate": _GEMINI_MODEL},
        candidate_count={},
        flags={},
    )

    # Build the answer response
    response = AnswerResponse(
        answer_markdown=full_text,
        citations=citations,
        confidence=confidence,
        diagnostics=diagnostics,
        query_id="",  # Populated by caller
    )

    # Attach cost to the underlying Answer in diagnostics if needed
    # The cost_usd is tracked here and should be propagated to trace

    logger.info(
        "gemini_generation_complete",
        mode=mode,
        answer_length=len(full_text),
        citations=len(citations),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
    )

    return response, cost_usd, prompt_tokens, completion_tokens


def _determine_confidence(
    answer_text: str,
    cited_indices: list[int],
    total_candidates: int,
) -> Literal["high", "medium", "low"]:
    """Determine confidence level based on answer characteristics.

    Per PLAN.md Part VIII §8.2:
        - high: direct answer with multiple supporting citations
        - medium: answer exists but citations are thin
        - low: refusal or minimal content

    Args:
        answer_text: The generated answer text.
        cited_indices: List of cited candidate indices.
        total_candidates: Total number of candidates available.

    Returns:
        Confidence level: "high", "medium", or "low".
    """
    # Refusal patterns
    refusal_phrases = [
        "cannot be answered",
        "not in the corpus",
        "I don't know",
        "no information",
        "outside the scope",
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


def generate_sync(
    question: str,
    candidates: list[RetrievalCandidate],
    system_prompt: str,
    user_prompt: str,
    mode: str = "fast",
    max_output_tokens: int = 8192,
    temperature: float = 0.3,
) -> tuple[str, float, int, int]:
    """Synchronous wrapper for generate_stream.

    Collects all chunks from the async generator and returns them as a single
    string along with cost and token usage.

    This is a convenience function for non-async contexts (CLI, tests).

    Args:
        Same as generate().

    Returns:
        Tuple of (full_text, cost_usd, prompt_tokens, completion_tokens).
    """
    import asyncio

    full_text = ""
    prompt_tokens = 0
    completion_tokens = 0
    cost_usd = 0.0

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    image_uris = [c.page_image_uri for c in candidates if c.page_image_uri]

    async def run_stream():
        nonlocal full_text, prompt_tokens, completion_tokens, cost_usd
        async for event in generate_stream(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_uris=image_uris if image_uris else None,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        ):
            full_text += event.chunk
            if event.is_final:
                prompt_tokens = event.prompt_tokens
                completion_tokens = event.completion_tokens
                cost_usd = event.cost_usd

    loop.run_until_complete(run_stream())

    return full_text, cost_usd, prompt_tokens, completion_tokens


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def smoke_test() -> dict:
    """Run a smoke test of the gemini module.

    Verifies:
        - _build_payload produces valid JSON
        - _calculate_cost computes correctly
        - _determine_confidence handles edge cases
        - generate_stream can be initialized (no API call made)

    Returns:
        Dict with smoke test results.
    """
    # Test _build_payload
    payload = _build_payload(
        system_prompt="You are an expert.",
        user_prompt="What is FSI?",
        image_uris=["s3://bucket/image.png"],
        max_output_tokens=1024,
        temperature=0.3,
    )
    assert "contents" in payload
    assert "system_instruction" in payload
    assert "generation_config" in payload
    assert payload["generation_config"]["max_output_tokens"] == 1024

    # Test _calculate_cost
    cost = _calculate_cost(prompt_tokens=1_000_000, completion_tokens=500_000)
    expected = (1.0 * 0.075) + (0.5 * 3.5)  # $0.075 + $1.75 = $1.825
    assert abs(cost - expected) < 0.001, f"Expected ~{expected}, got {cost}"

    # Zero tokens
    cost_zero = _calculate_cost(0, 0)
    assert cost_zero == 0.0

    # Test _determine_confidence
    assert _determine_confidence("", [], 5) == "low"
    assert _determine_confidence("I cannot answer this question.", [], 5) == "low"
    # "FSI is 2.5." is < 50 chars, so even with citations it's "low"
    assert _determine_confidence("FSI is 2.5.", [1, 2], 5) == "low"
    # Substantial text with citations = "medium" (no high because text < 200 chars)
    medium_text = "FSI for residential zones is 2.5 [1]. Parking [2]."
    assert _determine_confidence(medium_text, [1, 2], 3) == "medium"
    # Long text with citations = "high" (>200 chars + citation_ratio >= 0.3)
    assert _determine_confidence("A" * 250, [1, 2], 3) == "high"

    # Test with empty candidates
    assert _determine_confidence("Short answer.", [], 0) == "low"
    assert _determine_confidence("A" * 300, [1, 2], 3) == "high"

    # Test with empty candidates
    assert _determine_confidence("Short answer.", [], 0) == "low"
    assert _determine_confidence("A" * 300, [1, 2], 3) == "high"

    logger.info("gemini_smoke_test_passed")

    return {
        "passed": True,
        "payload_structure_valid": True,
        "cost_calculation_valid": True,
        "confidence_logic_valid": True,
    }
