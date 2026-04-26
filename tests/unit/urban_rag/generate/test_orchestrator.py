"""Unit tests for the generate orchestrator module.

Tests the answer() function returning streamed Answer events per PART VIII §8.1–§8.7:
    - Returns streamed Answer events in correct order
    - Citations are valid and resolve to page_ids in retrieval result
    - confidence field is one of high/medium/low
    - Deep mode uses self-critique pass
"""

import pytest

from urban_rag.common.types import RetrievalCandidate, RetrievalResult
from urban_rag.generate.orchestrator import (
    AnswerEvent,
    DoneEvent,
    ErrorEvent,
    GenerationCompletedEvent,
    GenerationStartedEvent,
    RefusedEvent,
    RetrievalCompletedEvent,
    RetrievalStartedEvent,
    TokenEvent,
    _determine_confidence,
    answer,
    smoke_test,
)


class TestDetermineConfidence:
    """Tests for _determine_confidence()."""

    def test_confidence_refusal_phrases(self):
        """Refusal phrases result in low confidence."""
        phrases = [
            "cannot be answered",
            "I don't know",
            "not in the corpus",
            "no information",
            "outside the scope",
            "insufficient information",
            "cannot answer",
        ]
        for phrase in phrases:
            assert _determine_confidence(phrase, [], 5) == "low", f"Failed for: {phrase}"

    def test_confidence_short_answer(self):
        """Answer shorter than 50 chars is low confidence."""
        assert _determine_confidence("FSI is 2.5.", [], 5) == "low"
        assert _determine_confidence("Short.", [], 5) == "low"
        assert _determine_confidence("", [], 5) == "low"

    def test_confidence_hallucinated_citations(self):
        """Hallucinated citations result in low confidence regardless of text."""
        text = "A" * 300  # Long answer
        assert _determine_confidence(text, [1, 2], 3, has_invalid_citations=True) == "low"

    def test_confidence_high(self):
        """Good citation coverage and substantial answer (>200 chars) = high."""
        text = "FSI for residential zones is 2.5 [1]. " * 20  # > 200 chars
        assert _determine_confidence(text, [1, 2], 3) == "high"

    def test_confidence_medium_no_citations_long_text(self):
        """Substantial answer (>150 chars) without citations is medium."""
        text = "B" * 200
        assert _determine_confidence(text, [], 5) == "medium"

    def test_confidence_medium_single_citation(self):
        """Single citation but substantial answer (>150 chars) is medium."""
        # ~100 chars with 1 citation out of 5 candidates (20% ratio < 30% for high)
        # but > 150 chars threshold for medium
        text = "FSI is 2.5 [1]. " * 10  # ~100 chars
        assert _determine_confidence(text, [1], 5) == "medium"

    def test_confidence_low_empty_candidates(self):
        """Empty candidates with short text is low."""
        assert _determine_confidence("Short.", [], 0) == "low"


class TestEventTypes:
    """Tests for AnswerEvent types."""

    def test_retrieval_started_event(self):
        """RetrievalStartedEvent has correct structure."""
        from urban_rag.generate.orchestrator import RetrievalStartedEvent
        event = RetrievalStartedEvent(query_id="test-id")
        assert event.query_id == "test-id"

    def test_retrieval_completed_event(self):
        """RetrievalCompletedEvent carries candidates and latency_ms."""
        candidates = [
            RetrievalCandidate(
                page_id="doc1#p001",
                score=0.8,
                channel_scores={},
                channel_ranks={},
                page_image_uri="s3://doc1/p001.png",
                extracted_text_excerpt="Test",
            )
        ]
        event = RetrievalCompletedEvent(
            query_id="test-id",
            candidates=candidates,
            latency_ms=150,
        )
        assert event.query_id == "test-id"
        assert len(event.candidates) == 1
        assert event.latency_ms == 150

    def test_token_event(self):
        """TokenEvent carries a chunk."""
        event = TokenEvent(query_id="test-id", chunk="Hello ")
        assert event.query_id == "test-id"
        assert event.chunk == "Hello "

    def test_generation_started_event(self):
        """GenerationStartedEvent carries mode."""
        event = GenerationStartedEvent(query_id="test-id", mode="fast")
        assert event.query_id == "test-id"
        assert event.mode == "fast"

        event_deep = GenerationStartedEvent(query_id="test-id", mode="deep")
        assert event_deep.mode == "deep"

    def test_generation_completed_event(self):
        """GenerationCompletedEvent carries full answer data."""
        from urban_rag.common.types import AnswerDiagnostics, Citation
        diagnostics = AnswerDiagnostics()
        citations = [
            Citation(
                idx=1,
                doc_id="doc1",
                page_id="doc1#p001",
                page_num=1,
                doc_title="Test Doc",
                section_title="Section 1",
                score=0.8,
            )
        ]
        event = GenerationCompletedEvent(
            query_id="test-id",
            answer_markdown="The FSI is 2.5 [1].",
            citations=citations,
            confidence="high",
            diagnostics=diagnostics,
        )
        assert event.query_id == "test-id"
        assert event.confidence == "high"
        assert len(event.citations) == 1
        assert event.citations[0].idx == 1

    def test_refused_event(self):
        """RefusedEvent carries reason and message."""
        event = RefusedEvent(
            query_id="test-id",
            reason="out_of_corpus",
            message="The query is outside the corpus scope.",
        )
        assert event.query_id == "test-id"
        assert event.reason == "out_of_corpus"

    def test_error_event(self):
        """ErrorEvent carries code, message, and stage."""
        event = ErrorEvent(
            query_id="test-id",
            code="generation_failed",
            message="API key invalid",
            stage="generation",
        )
        assert event.code == "generation_failed"
        assert event.stage == "generation"

    def test_done_event(self):
        """DoneEvent is the terminal event."""
        event = DoneEvent(query_id="test-id")
        assert event.query_id == "test-id"


class TestAnswerFunction:
    """Tests for the answer() async generator."""

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

    @pytest.fixture
    def sample_retrieval_result(
        self, sample_candidates
    ) -> RetrievalResult:
        """Create a sample RetrievalResult."""
        return RetrievalResult(
            query="What is FSI for residential zones?",
            expanded_queries=["What is FSI for residential zones?"],
            candidates=sample_candidates,
            latency_ms=100,
            flags={},
            retrieval_strategy="hybrid",
        )

    @pytest.mark.asyncio
    async def test_answer_no_candidates_returns_refused(
        self, sample_retrieval_result: RetrievalResult
    ):
        """No candidates yields RefusedEvent."""
        sample_retrieval_result.candidates = []

        events = []
        async for event in answer(
            query="What is FSI?",
            retrieval_result=sample_retrieval_result,
        ):
            events.append(event)

        refused_events = [e for e in events if isinstance(e, RefusedEvent)]
        assert len(refused_events) == 1
        assert refused_events[0].reason == "insufficient_context"
        # Last event should be DoneEvent
        assert isinstance(events[-1], DoneEvent)

    @pytest.mark.asyncio
    async def test_answer_event_order(
        self,
        sample_retrieval_result: RetrievalResult,
        sample_candidates: list[RetrievalCandidate],
    ):
        """Events follow strict order per VAL-API-007."""
        from unittest.mock import patch, AsyncMock

        async def mock_stream(*args, **kwargs):
            # Yield a final event with no error
            yield type("E", (), {"is_error": False, "is_final": True, "chunk": "", "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0})()

        with patch(
            "urban_rag.generate.orchestrator.generate_stream",
            side_effect=mock_stream,
        ):
            events = []
            async for event in answer(
                query="What is FSI?",
                retrieval_result=sample_retrieval_result,
            ):
                events.append(event)

        # Event order check
        assert isinstance(events[0], RetrievalStartedEvent)
        assert isinstance(events[1], RetrievalCompletedEvent)
        assert isinstance(events[2], GenerationStartedEvent)
        # Tokens may follow
        final_event = events[-1]
        assert isinstance(final_event, DoneEvent)

    @pytest.mark.asyncio
    async def test_answer_confidence_is_valid(
        self,
        sample_retrieval_result: RetrievalResult,
    ):
        """Confidence is always one of high/medium/low."""
        from unittest.mock import patch, AsyncMock

        async def mock_stream(*args, **kwargs):
            # Yield a substantial answer
            text = "The FSI for residential zones is 2.5 [1]. " * 15
            yield type("E", (), {
                "is_error": False,
                "is_final": True,
                "chunk": text,
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "cost_usd": 0.001,
            })()

        with patch(
            "urban_rag.generate.orchestrator.generate_stream",
            side_effect=mock_stream,
        ):
            events = []
            async for event in answer(
                query="What is FSI?",
                retrieval_result=sample_retrieval_result,
            ):
                events.append(event)

        completion_events = [
            e for e in events if isinstance(e, GenerationCompletedEvent)
        ]
        assert len(completion_events) == 1
        confidence = completion_events[0].confidence
        assert confidence in ("high", "medium", "low")

    @pytest.mark.asyncio
    async def test_answer_citations_resolve_to_page_ids(
        self,
        sample_retrieval_result: RetrievalResult,
    ):
        """Citations resolve to page_ids in retrieval result."""
        from unittest.mock import patch

        async def mock_stream(*args, **kwargs):
            text = "The FSI is 2.5 [1] and parking is 1 per 100 sqm [2]."
            yield type("E", (), {
                "is_error": False,
                "is_final": True,
                "chunk": text,
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "cost_usd": 0.001,
            })()

        with patch(
            "urban_rag.generate.orchestrator.generate_stream",
            side_effect=mock_stream,
        ):
            events = []
            async for event in answer(
                query="What is FSI?",
                retrieval_result=sample_retrieval_result,
            ):
                events.append(event)

        completion_events = [
            e for e in events if isinstance(e, GenerationCompletedEvent)
        ]
        assert len(completion_events) == 1
        citations = completion_events[0].citations
        assert len(citations) == 2
        # Each citation should have a valid page_id from the candidates
        page_ids = {c.page_id for c in citations}
        candidate_page_ids = {c.page_id for c in sample_retrieval_result.candidates}
        assert page_ids.issubset(candidate_page_ids)


class TestSmokeTest:
    """Tests for smoke_test()."""

    def test_smoke_test_passes(self):
        """smoke_test() runs without errors."""
        result = smoke_test()
        assert result["passed"] is True
        assert result["confidence_logic_valid"] is True
        assert result["citation_validation_valid"] is True
        assert result["event_types_valid"] is True

    def test_smoke_test_confidence_edge_cases(self):
        """Smoke test validates confidence edge cases."""
        result = smoke_test()
        assert result["passed"] is True
