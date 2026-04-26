"""Unit tests for common/types.py — all pydantic models from PLAN Appendix B."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from urban_rag.common.types import (
    Answer,
    AnswerDiagnostics,
    AnswerResponse,
    ChunkRecord,
    Citation,
    DocumentRecord,
    LayoutBlock,
    PageMetadata,
    PageRecord,
    QueryRequest,
    RetrievalCandidate,
    RetrievalResult,
    SectionRecord,
    Trace,
    TraceSpan,
)


class TestDocumentRecord:
    """Tests for DocumentRecord (PLAN Appendix B §B.1)."""

    def test_required_fields(self) -> None:
        """doc_hash, filename, page_count are required."""
        rec = DocumentRecord(
            doc_hash="abc123",
            filename="test.pdf",
            page_count=10,
        )
        assert rec.doc_hash == "abc123"
        assert rec.page_count == 10

    def test_all_fields(self) -> None:
        """Full construction with all fields."""
        rec = DocumentRecord(
            doc_hash="a" * 64,
            filename="nbc_2016_vol1.pdf",
            title="NBC 2016 Vol 1",
            family="NBC",
            jurisdiction="IN",
            publisher="BIS",
            year=2016,
            version="v1",
            license="public_domain",
            page_count=2200,
            storage_uri="s3://urban-rag-source/nbc.pdf",
            ingested_at=datetime(2026, 1, 1),
            indexed_at=datetime(2026, 1, 2),
        )
        assert rec.family == "NBC"
        assert rec.jurisdiction == "IN"
        assert rec.year == 2016
        assert rec.license == "public_domain"

    def test_family_literal_validation(self) -> None:
        """Invalid family values are rejected."""
        with pytest.raises(ValidationError):
            DocumentRecord(
                doc_hash="abc",
                filename="x.pdf",
                page_count=1,
                family="INVALID",
            )

    def test_license_literal_validation(self) -> None:
        """Invalid license values are rejected."""
        with pytest.raises(ValidationError):
            DocumentRecord(
                doc_hash="abc",
                filename="x.pdf",
                page_count=1,
                license="invalid_license",
            )


class TestLayoutBlock:
    """Tests for LayoutBlock (PLAN Appendix B §B.3)."""

    def test_required_fields(self) -> None:
        """block_id, page_id, type, bbox, confidence required."""
        block = LayoutBlock(
            block_id="b1",
            page_id="doc1#p0001",
            type="paragraph",
            bbox=(0.0, 0.0, 100.0, 200.0),
            confidence=0.95,
        )
        assert block.block_id == "b1"
        assert block.type == "paragraph"
        assert block.confidence == 0.95

    def test_type_literal_validation(self) -> None:
        """Invalid block type is rejected."""
        with pytest.raises(ValidationError):
            LayoutBlock(
                block_id="b1",
                page_id="doc1#p0001",
                type="invalid_type",
                bbox=(0, 0, 100, 200),
                confidence=0.9,
            )

    def test_confidence_range(self) -> None:
        """Confidence must be 0-1."""
        with pytest.raises(ValidationError):
            LayoutBlock(
                block_id="b1",
                page_id="doc1#p0001",
                type="paragraph",
                bbox=(0, 0, 100, 200),
                confidence=1.5,
            )


class TestPageRecord:
    """Tests for PageRecord (PLAN Appendix B §B.2)."""

    def test_required_fields(self) -> None:
        """page_id, doc_id, page_num, dpi_used required."""
        page = PageRecord(
            page_id="abc#p0001",
            doc_id="abc",
            page_num=1,
            dpi_used=100,
        )
        assert page.page_id == "abc#p0001"
        assert page.page_num == 1
        assert page.page_type == "TEXT"  # default

    def test_page_type_defaults(self) -> None:
        """page_type defaults to TEXT."""
        page = PageRecord(
            page_id="abc#p0001",
            doc_id="abc",
            page_num=1,
            dpi_used=250,
        )
        assert page.page_type == "TEXT"
        assert page.layout == []

    def test_page_type_literal(self) -> None:
        """Invalid page_type rejected."""
        with pytest.raises(ValidationError):
            PageRecord(
                page_id="abc#p0001",
                doc_id="abc",
                page_num=1,
                dpi_used=100,
                page_type="INVALID",
            )


class TestSectionRecord:
    """Tests for SectionRecord (PLAN Appendix B §B.4)."""

    def test_required_fields(self) -> None:
        """section_id, doc_id, title, level, start_page, end_page required."""
        sec = SectionRecord(
            section_id="abc#s001",
            doc_id="abc",
            title="Chapter 4 Fire Safety",
            level=1,
            start_page=42,
            end_page=65,
        )
        assert sec.level == 1
        assert sec.title_path == []

    def test_hierarchy_fields(self) -> None:
        """Parent section and title_path work."""
        sec = SectionRecord(
            section_id="abc#s002",
            doc_id="abc",
            title="4.2 Definitions",
            level=2,
            parent_section_id="abc#s001",
            start_page=43,
            end_page=43,
            title_path=["Chapter 4 Fire Safety", "4.2 Definitions"],
        )
        assert sec.parent_section_id == "abc#s001"
        assert len(sec.title_path) == 2


class TestChunkRecord:
    """Tests for ChunkRecord (PLAN Appendix B §B.5)."""

    def test_required_fields(self) -> None:
        """chunk_id, doc_id, page_id, text, token_count required."""
        chunk = ChunkRecord(
            chunk_id="c1",
            doc_id="abc",
            page_id="abc#p0001",
            text="This is a chunk of text.",
            token_count=256,
        )
        assert chunk.embedding_model == "Alibaba-NLP/gte-modernbert-colbert"

    def test_text_channel_defaults(self) -> None:
        """embedding_model and chunk_index_in_section have defaults."""
        chunk = ChunkRecord(
            chunk_id="c1",
            doc_id="abc",
            page_id="abc#p0001",
            text="Sample text",
            token_count=100,
        )
        assert chunk.chunk_index_in_section == 0
        assert chunk.section_id is None


class TestRetrievalCandidate:
    """Tests for RetrievalCandidate (PLAN Appendix B §B.6)."""

    def test_required_fields(self) -> None:
        """page_id, score, page_image_uri required."""
        cand = RetrievalCandidate(
            page_id="abc#p0001",
            score=0.85,
            page_image_uri="s3://pages/abc/p0001.png",
        )
        assert cand.rerank_score is None
        assert cand.channel_scores == {}

    def test_channel_scores_and_ranks(self) -> None:
        """channel_scores and channel_ranks are tracked."""
        cand = RetrievalCandidate(
            page_id="abc#p0001",
            score=12.3,
            page_image_uri="s3://pages/abc/p0001.png",
            channel_scores={"visual": 10.0, "text": 0.8, "sparse": 4.2},
            channel_ranks={"visual": 1, "text": 5, "sparse": 12},
        )
        assert cand.channel_scores["visual"] == 10.0
        assert cand.channel_ranks["visual"] == 1


class TestRetrievalResult:
    """Tests for RetrievalResult (PLAN Appendix B §B.7)."""

    def test_required_query(self) -> None:
        """query is required."""
        res = RetrievalResult(query="What is FSI?")
        assert res.query == "What is FSI?"
        assert res.candidates == []
        assert res.retrieval_strategy == "hybrid"

    def test_flags_defaults(self) -> None:
        """flags default to empty dict."""
        res = RetrievalResult(query="Test")
        assert res.flags == {}


class TestCitation:
    """Tests for Citation (PLAN Appendix B §B.8)."""

    def test_required_fields(self) -> None:
        """idx, doc_id, page_id, page_num required."""
        cit = Citation(
            idx=1,
            doc_id="abc",
            page_id="abc#p0001",
            page_num=1,
        )
        assert cit.score == 0.0
        assert cit.section_title is None

    def test_idx_must_be_positive(self) -> None:
        """idx must be >= 1."""
        with pytest.raises(ValidationError):
            Citation(
                idx=0,
                doc_id="abc",
                page_id="abc#p0001",
                page_num=1,
            )


class TestAnswer:
    """Tests for Answer (PLAN Appendix B §B.8)."""

    def test_required_fields(self) -> None:
        """query_id, text required."""
        ans = Answer(
            query_id="q123",
            text="The FSI is 2.5.",
        )
        assert ans.refused is False
        assert ans.citations == []

    def test_refusal_fields(self) -> None:
        """refused and refusal_reason work together."""
        ans = Answer(
            query_id="q123",
            text="",
            refused=True,
            refusal_reason="out_of_corpus",
        )
        assert ans.refused is True
        assert ans.refusal_reason == "out_of_corpus"


class TestAnswerDiagnostics:
    """Tests for AnswerDiagnostics."""

    def test_defaults(self) -> None:
        """All fields default to empty/False."""
        diag = AnswerDiagnostics()
        assert diag.latency_ms == {}
        assert diag.backends == {}
        assert diag.candidate_count == {}
        assert diag.flags == {}


class TestAnswerResponse:
    """Tests for AnswerResponse."""

    def test_required_fields(self) -> None:
        """answer_markdown and query_id required."""
        resp = AnswerResponse(
            answer_markdown="The answer is **bold**.",
            query_id="q123",
        )
        assert resp.confidence == "medium"
        assert resp.citations == []

    def test_confidence_literal(self) -> None:
        """Invalid confidence is rejected."""
        with pytest.raises(ValidationError):
            AnswerResponse(
                answer_markdown="text",
                query_id="q123",
                confidence="very_high",
            )


class TestQueryRequest:
    """Tests for QueryRequest."""

    def test_defaults(self) -> None:
        """mode defaults to fast, top_k to 5."""
        req = QueryRequest(question="What is FAR?")
        assert req.mode == "fast"
        assert req.top_k == 5
        assert req.filters == {}

    def test_question_length_validation(self) -> None:
        """Question length 1-1000 enforced."""
        with pytest.raises(ValidationError):
            QueryRequest(question="")

        with pytest.raises(ValidationError):
            QueryRequest(question="x" * 1001)

    def test_top_k_bounds(self) -> None:
        """top_k must be 1-50."""
        with pytest.raises(ValidationError):
            QueryRequest(question="test", top_k=0)

        with pytest.raises(ValidationError):
            QueryRequest(question="test", top_k=51)


class TestPageMetadata:
    """Tests for PageMetadata."""

    def test_defaults(self) -> None:
        """dpi=100, language=en, page_type=text by default."""
        meta = PageMetadata(
            doc_hash="abc",
            doc_filename="test.pdf",
            page_num=1,
        )
        assert meta.dpi == 100
        assert meta.language == "en"
        assert meta.page_type == "text"


class TestTraceSpan:
    """Tests for TraceSpan."""

    def test_required_name(self) -> None:
        """name is required."""
        span = TraceSpan(name="retrieval")
        assert span.started_at is not None
        assert span.finished_at is None
        assert span.attributes == {}


class TestTrace:
    """Tests for Trace (PLAN Appendix B §B.9)."""

    def test_required_query_id(self) -> None:
        """query_id is required."""
        trace = Trace(query_id="q123")
        assert trace.query_id == "q123"
        assert trace.mode == "fast"

    def test_full_trace_with_results(self) -> None:
        """Full trace with retrieval result and answer."""
        retrieval = RetrievalResult(
            query="What is FSI?",
            candidates=[
                RetrievalCandidate(
                    page_id="abc#p0001",
                    score=0.85,
                    page_image_uri="s3://pages/abc/p0001.png",
                )
            ],
            latency_ms=150,
        )
        answer = Answer(
            query_id="q123",
            text="The FSI is 2.5 [1].",
            citations=[Citation(idx=1, doc_id="abc", page_id="abc#p0001", page_num=1)],
            latency_ms=2000,
            cost_usd=0.003,
        )
        trace = Trace(
            query_id="q123",
            mode="fast",
            corpus_version="v1.0.0",
            embed_model="colqwen2.5-v0.2",
            rerank_model="gemini-2.5-flash",
            gen_model="gemini-2.5-flash",
            retrieval_result=retrieval,
            answer=answer,
        )
        assert trace.retrieval_result.candidates[0].page_id == "abc#p0001"
        assert trace.answer.citations[0].idx == 1
