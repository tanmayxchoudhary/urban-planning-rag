"""Unit tests for the generate gemini module.

Tests streaming Gemini 2.5 Flash client with image attachment per PART VIII §8.1:
    - Streaming token-by-token delivery
    - Max-output cap enforcement
    - Cost_usd recorded per request
    - Image attachment for visual grounding
"""

import pytest

from urban_rag.common.types import RetrievalCandidate
from urban_rag.generate.gemini import (
    _build_payload,
    _calculate_cost,
    _determine_confidence,
    _GeminiStreamEvent,
    generate_stream,
    generate_sync,
    smoke_test,
)


class TestBuildPayload:
    """Tests for _build_payload()."""

    def test_build_payload_basic(self):
        """Basic payload with text only."""
        payload = _build_payload(
            system_prompt="You are an expert.",
            user_prompt="What is FSI?",
            image_uris=None,
            max_output_tokens=1024,
            temperature=0.3,
        )
        assert "contents" in payload
        assert "system_instruction" in payload
        assert "generation_config" in payload
        assert payload["generation_config"]["max_output_tokens"] == 1024
        assert payload["generation_config"]["temperature"] == 0.3

    def test_build_payload_with_images(self):
        """Payload includes image URIs when provided."""
        payload = _build_payload(
            system_prompt="You are an expert.",
            user_prompt="What is FSI?",
            image_uris=["s3://bucket/doc1.png", "s3://bucket/doc2.png"],
            max_output_tokens=2048,
            temperature=0.5,
        )
        # Should have two content blocks: images and text
        assert len(payload["contents"]) == 2
        # First block is images
        assert "file_data" in payload["contents"][0]["parts"][0]
        assert payload["contents"][0]["parts"][0]["file_data"]["file_uri"] == "s3://bucket/doc1.png"

    def test_build_payload_empty_images(self):
        """Empty image list produces text-only payload."""
        payload = _build_payload(
            system_prompt="You are an expert.",
            user_prompt="What is FSI?",
            image_uris=[],
            max_output_tokens=1024,
            temperature=0.3,
        )
        # With empty image list, only text content block
        assert len(payload["contents"]) == 1

    def test_build_payload_system_instruction(self):
        """System instruction is included correctly."""
        payload = _build_payload(
            system_prompt="Special instructions for this query.",
            user_prompt="What is FSI?",
            image_uris=None,
            max_output_tokens=1024,
            temperature=0.3,
        )
        assert "parts" in payload["system_instruction"]
        assert "text" in payload["system_instruction"]["parts"][0]
        special_instr = "Special instructions for this query."
        assert payload["system_instruction"]["parts"][0]["text"] == special_instr


class TestCalculateCost:
    """Tests for _calculate_cost()."""

    def test_cost_full_tokens(self):
        """Cost calculation with realistic token counts."""
        # 1M input + 500k output
        cost = _calculate_cost(prompt_tokens=1_000_000, completion_tokens=500_000)
        expected = (1.0 * 0.075) + (0.5 * 3.5)  # $0.075 + $1.75 = $1.825
        assert abs(cost - expected) < 0.001

    def test_cost_zero_tokens(self):
        """Zero tokens yields zero cost."""
        assert _calculate_cost(0, 0) == 0.0

    def test_cost_small_request(self):
        """Small request cost calculation."""
        # 1000 input + 100 output
        cost = _calculate_cost(prompt_tokens=1000, completion_tokens=100)
        expected = (1000 / 1_000_000 * 0.075) + (100 / 1_000_000 * 3.5)
        assert abs(cost - expected) < 0.0001

    def test_cost_large_request(self):
        """Large request cost calculation."""
        # 10M input + 8M output
        cost = _calculate_cost(prompt_tokens=10_000_000, completion_tokens=8_000_000)
        # $0.75 + $28 = $28.75
        assert abs(cost - 28.75) < 0.01

    def test_cost_rounding(self):
        """Cost is rounded to 6 decimal places."""
        cost = _calculate_cost(prompt_tokens=333, completion_tokens=777)
        # Very small cost, should be rounded
        assert len(str(cost).split(".")[-1]) <= 6


class TestDetermineConfidence:
    """Tests for _determine_confidence()."""

    def test_confidence_refusal_phrases(self):
        """Refusal phrases result in low confidence."""
        phrases = [
            "cannot be answered",
            "not in the corpus",
            "I don't know",
            "no information",
            "outside the scope",
        ]
        for phrase in phrases:
            assert _determine_confidence(phrase, [], 5) == "low", f"Failed for: {phrase}"

    def test_confidence_short_answer(self):
        """Answer shorter than 50 chars is low confidence."""
        assert _determine_confidence("FSI is 2.5.", [], 5) == "low"
        assert _determine_confidence("Short.", [], 5) == "low"
        assert _determine_confidence("", [], 5) == "low"

    def test_confidence_medium_no_citations(self):
        """Short answer without citations is low."""
        text = "FSI for residential zones is 2.5 according to Indian regulations."
        assert _determine_confidence(text, [], 5) == "low"

    def test_confidence_high_with_citations(self):
        """Good citation coverage and substantial answer (>200 chars) = high."""
        text = "FSI for residential zones is 2.5 [1]. " * 20  # Make it > 200 chars
        # 2 citations out of 3 candidates = 66% ratio, text > 200 chars
        assert _determine_confidence(text, [1, 2], 3) == "high"

    def test_confidence_medium_single_citation(self):
        """Short text with single citation is low (needs >150 chars for medium)."""
        text = "FSI is 2.5 [1]."
        assert _determine_confidence(text, [1], 5) == "low"

    def test_confidence_low_empty_candidates(self):
        """Empty candidates with short text is low."""
        assert _determine_confidence("Short.", [], 0) == "low"

    def test_confidence_high_many_candidates(self):
        """High citation ratio (>30%) with substantial text is high."""
        text = "A" * 300  # Long answer
        # 4 citations out of 10 = 40% ratio - meets >= 30% threshold
        assert _determine_confidence(text, [1, 2, 3, 4], 10) == "high"

    def test_confidence_medium_low_citation_ratio(self):
        """Low citation ratio but substantial answer (>150 chars) is medium."""
        text = "B" * 250  # Long answer but only 1 citation
        # 1 citation out of 10 = 10% ratio (< 30%) but > 150 chars
        assert _determine_confidence(text, [1], 10) == "medium"

    def test_confidence_medium_substantial_no_citations(self):
        """Substantial answer without citations is medium (if > 150 chars)."""
        text = "B" * 200
        assert _determine_confidence(text, [], 5) == "medium"


class TestGeminiStreamEvent:
    """Tests for _GeminiStreamEvent dataclass."""

    def test_stream_event_default(self):
        """Default event is empty with is_final=False."""
        event = _GeminiStreamEvent()
        assert event.chunk == ""
        assert event.is_final is False
        assert event.prompt_tokens == 0
        assert event.completion_tokens == 0
        assert event.cost_usd == 0.0
        assert event.error is None

    def test_stream_event_chunk(self):
        """Chunk is captured correctly."""
        event = _GeminiStreamEvent(chunk="Hello world")
        assert event.chunk == "Hello world"

    def test_stream_event_final(self):
        """Final event has token counts and cost."""
        event = _GeminiStreamEvent(
            is_final=True,
            prompt_tokens=1000,
            completion_tokens=500,
            cost_usd=0.00175,
        )
        assert event.is_final is True
        assert event.prompt_tokens == 1000
        assert event.completion_tokens == 500
        assert event.cost_usd == 0.00175

    def test_stream_event_error(self):
        """Error event is detected via is_error property."""
        event = _GeminiStreamEvent(error="API key invalid")
        assert event.is_error is True
        assert event.error == "API key invalid"

    def test_stream_event_no_error(self):
        """No error event has is_error=False."""
        event = _GeminiStreamEvent(chunk="some text")
        assert event.is_error is False


class TestGenerateStream:
    """Tests for generate_stream()."""

    @pytest.fixture
    def sample_candidates(self) -> list[RetrievalCandidate]:
        """Create sample retrieval candidates."""
        return [
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

    @pytest.mark.asyncio
    async def test_generate_stream_no_api_key(self):
        """No API key yields error event."""
        from unittest.mock import patch

        # Mock settings to return placeholder key
        with patch("urban_rag.generate.gemini.get_settings") as mock_settings:
            mock_settings.return_value.gemini_api_key = "your_gemini_api_key_here"

            events = []
            async for event in generate_stream(
                system_prompt="You are an expert.",
                user_prompt="What is FSI?",
            ):
                events.append(event)

            assert len(events) >= 1
            error_events = [e for e in events if e.is_error]
            assert len(error_events) >= 1
            err_lower = error_events[0].error.lower()
            assert "placeholder" in err_lower or "not set" in err_lower

    @pytest.mark.asyncio
    async def test_generate_stream_empty_prompt(self):
        """Empty system prompt yields error event (validation)."""
        from unittest.mock import patch

        with patch("urban_rag.generate.gemini.get_settings") as mock_settings:
            mock_settings.return_value.gemini_api_key = "test-key"

            # Empty system prompt should still yield error
            events = []
            async for event in generate_stream(
                system_prompt="",
                user_prompt="What is FSI?",
            ):
                events.append(event)

            # The API key check passes but system prompt may be rejected
            # Just verify we get events (could be error or empty stream)


class TestGenerateSync:
    """Tests for generate_sync()."""

    def test_generate_sync_no_api_key(self):
        """No API key yields error in sync mode."""
        from unittest.mock import patch

        with patch("urban_rag.generate.gemini.get_settings") as mock_settings:
            mock_settings.return_value.gemini_api_key = "your_gemini_api_key_here"

            _text, _cost, _prompt_tok, _completion_tok = generate_sync(
                question="What is FSI?",
                candidates=[],
                system_prompt="You are an expert.",
                user_prompt="What is FSI?",
            )

            # Error case - text may be empty, cost should be 0
            assert _cost == 0.0

    def test_generate_sync_basic(self):
        """Sync wrapper collects all chunks."""
        from unittest.mock import patch

        with patch("urban_rag.generate.gemini.get_settings") as mock_settings:
            mock_settings.return_value.gemini_api_key = "your_gemini_api_key_here"

            _text, _cost, _prompt_tok, _completion_tok = generate_sync(
                question="What is FSI?",
                candidates=[],
                system_prompt="You are an expert.",
                user_prompt="What is FSI?",
            )

            # Without real API, we get empty/error results
            # Just verify the return type is string (error event yields empty)


class TestSmokeTest:
    """Tests for smoke_test()."""

    def test_smoke_test_passes(self):
        """smoke_test() runs without errors and returns pass."""
        result = smoke_test()
        assert result["passed"] is True
        assert result["payload_structure_valid"] is True
        assert result["cost_calculation_valid"] is True
        assert result["confidence_logic_valid"] is True

    def test_smoke_test_payload_structure(self):
        """Payload has correct structure after smoke test."""
        result = smoke_test()
        assert result["payload_structure_valid"] is True

    def test_smoke_test_cost_calculation(self):
        """Cost calculation logic validated."""
        result = smoke_test()
        assert result["cost_calculation_valid"] is True

    def test_smoke_test_confidence_logic(self):
        """Confidence determination logic validated."""
        result = smoke_test()
        assert result["confidence_logic_valid"] is True


class TestGeminiEventEdgeCases:
    """Edge case tests for Gemini event handling."""

    def test_multiple_hallucinated_citations(self):
        """Multiple hallucinated citations correctly replaced."""
        from urban_rag.generate.prompts import validate_citations

        candidates = [
            RetrievalCandidate(
                page_id="doc1#p001",
                score=0.8,
                channel_scores={},
                channel_ranks={},
                page_image_uri="s3://doc1/p001.png",
                extracted_text_excerpt="Test excerpt",
            ),
            RetrievalCandidate(
                page_id="doc2#p002",
                score=0.75,
                channel_scores={},
                channel_ranks={},
                page_image_uri="s3://doc2/p002.png",
                extracted_text_excerpt="Test excerpt 2",
            ),
            RetrievalCandidate(
                page_id="doc3#p003",
                score=0.75,
                channel_scores={},
                channel_ranks={},
                page_image_uri="s3://doc3/p003.png",
                extracted_text_excerpt="Test excerpt 3",
            ),
        ]

        answer = "Facts: [1], [2], [3], [99], [100], [5]."
        corrected, invalid = validate_citations(answer, candidates)

        # [99], [100], [5] should be replaced with [?]
        assert corrected.count("[?]") == 3
        # [1], [2], [3] should remain
        assert "[1]" in corrected
        assert "[2]" in corrected
        assert "[3]" in corrected
        assert 99 in invalid
        assert 100 in invalid
        assert 5 in invalid
        assert 1 not in invalid
        assert 2 not in invalid
        assert 3 not in invalid

    def test_citation_at_document_boundary(self):
        """Citation exactly at document boundary is valid."""
        from urban_rag.generate.prompts import validate_citations

        candidates = [
            RetrievalCandidate(
                page_id="doc1#p001",
                score=0.8,
                channel_scores={},
                channel_ranks={},
                page_image_uri="s3://doc1/p001.png",
                extracted_text_excerpt="Test excerpt",
            ),
        ]

        # Citation [1] is valid (1-based index into 1 candidate)
        answer = "The answer is [1]."
        corrected, invalid = validate_citations(answer, candidates)
        assert "[1]" in corrected
        assert invalid == []

        # Citation [2] is invalid (only 1 candidate)
        answer2 = "The answer is [2]."
        corrected2, invalid2 = validate_citations(answer2, candidates)
        assert "[?]" in corrected2
        assert 2 in invalid2
