"""Shared pydantic types for Urban RAG.

This module is the canonical source of truth for all inter-module data contracts.
Every type is a pydantic v2 model — see PLAN.md Appendix B for the full schema reference.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Source documents
# ---------------------------------------------------------------------------


class DocumentRecord(BaseModel):
    """A document as recorded in the corpus manifest (PLAN Appendix B §B.1)."""

    doc_hash: str = Field(..., description="SHA256 of file content, hex")
    filename: str = Field(..., description="Original filename, sanitized")
    title: str = Field(default="", description="Human-friendly title")
    family: Literal[
        "NBC", "URDPFI", "SWM", "IRC", "IS", "MASTER_PLAN", "BBL", "OTHER"
    ] = Field(default="OTHER")
    jurisdiction: str | None = Field(
        default=None,
        description='ISO 3166-2 string, e.g. "IN", "IN-DL", "IN-MH-Mumbai"',
    )
    publisher: str | None = Field(default=None, description='BIS", "MoHUA", "ULB-Mumbai"')
    year: int | None = Field(default=None)
    version: str | None = Field(default=None, description='"v1", "v2", "2016", "2024"')
    license: Literal["public_domain", "gov_open", "fair_use_research", "unknown"] = (
        Field(default="unknown")
    )
    page_count: int = Field(..., ge=0)
    size_bytes: int = Field(default=0, ge=0, description="File size in bytes")
    storage_uri: str = Field(default="", description="s3://urban-rag-source/<sha>.pdf")
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    indexed_at: datetime | None = Field(default=None)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


class LayoutBlock(BaseModel):
    """A single layout block within a page (PLAN Appendix B §B.3)."""

    block_id: str = Field(..., description="Unique block identifier")
    page_id: str = Field(..., description="Foreign key to PageRecord")
    type: Literal[
        "heading", "paragraph", "table", "figure", "caption", "list", "footnote"
    ] = Field(...)
    text: str | None = Field(default=None, description="Text content if any")
    bbox: tuple[float, float, float, float] = Field(
        ...,
        description="(x0, y0, x1, y1) in image coordinates",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Parser confidence 0-1")


class PageRecord(BaseModel):
    """A single rendered page (PLAN Appendix B §B.2)."""

    page_id: str = Field(..., description='f"{doc_id}#p{page_num:04d}"')
    doc_id: str = Field(..., description="Foreign key to DocumentRecord")
    page_num: int = Field(..., ge=1, description="1-based page number")
    page_type: Literal["TEXT", "VISUAL", "BLANK"] = Field(default="TEXT")
    dpi_used: int = Field(..., description="100 or 250")
    image_uri: str = Field(default="", description="s3://urban-rag-pages/<doc_id>/<page>.png")
    extracted_text: str = Field(default="", description="Full page text (OCR'd or parsed)")
    layout: list[LayoutBlock] = Field(default_factory=list)
    section_id: str | None = Field(default=None, description="Foreign key to SectionRecord")
    section_title: str | None = Field(default=None)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


class SectionRecord(BaseModel):
    """A section in a document's hierarchy (PLAN Appendix B §B.4)."""

    section_id: str = Field(..., description='f"{doc_id}#s{idx:03d}"')
    doc_id: str = Field(..., description="Foreign key to DocumentRecord")
    title: str = Field(...)
    level: int = Field(..., ge=1, description="1=chapter, 2=section, 3=subsection")
    parent_section_id: str | None = Field(default=None)
    start_page: int = Field(..., ge=1)
    end_page: int = Field(..., ge=1)
    title_path: list[str] = Field(
        default_factory=list,
        description='["Chapter 4","4.3 Land Use","4.3.1 Residential"]',
    )


# ---------------------------------------------------------------------------
# Chunks (text channel)
# ---------------------------------------------------------------------------


class ChunkRecord(BaseModel):
    """A text chunk for the parallel text retrieval channel (PLAN Appendix B §B.5)."""

    chunk_id: str = Field(...)
    doc_id: str = Field(...)
    page_id: str = Field(...)
    section_id: str | None = Field(default=None)
    text: str = Field(..., description="256-512 token chunk")
    token_count: int = Field(..., ge=0)
    embedding_model: str = Field(
        default="Alibaba-NLP/gte-modernbert-colbert",
        description="Text embedding model used",
    )
    chunk_index_in_section: int = Field(default=0)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


class RetrievalCandidate(BaseModel):
    """A single retrieval candidate (PLAN Appendix B §B.6)."""

    page_id: str = Field(...)
    score: float = Field(..., description="Final fused score")
    channel_scores: dict[str, float] = Field(
        default_factory=dict,
        description='{"visual": 12.3, "text": 0.81, "sparse": 4.2}',
    )
    channel_ranks: dict[str, int] = Field(
        default_factory=dict,
        description='{"visual": 1, "text": 4, "sparse": 12}',
    )
    rerank_score: float | None = Field(default=None, description="Populated after VLM rerank")
    rerank_rationale: str | None = Field(default=None)
    page_image_uri: str = Field(...)
    extracted_text_excerpt: str = Field(
        default="",
        description="First 500 chars of page text",
    )
    section_title: str | None = Field(default=None)


class RetrievalResult(BaseModel):
    """Result of the full retrieval pipeline (PLAN Appendix B §B.7)."""

    query: str = Field(...)
    expanded_queries: list[str] = Field(default_factory=list)
    candidates: list[RetrievalCandidate] = Field(default_factory=list)
    latency_ms: int = Field(default=0)
    flags: dict[str, bool] = Field(
        default_factory=dict,
        description="degraded_mode, vlm_rerank_skipped, etc.",
    )
    retrieval_strategy: Literal["visual_primary", "text_primary", "hybrid"] = (
        Field(default="hybrid")
    )


# ---------------------------------------------------------------------------
# Citations & answers
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    """A citation reference within an answer (PLAN Appendix B §B.8)."""

    idx: int = Field(..., ge=1, description="[1], [2], ...")
    doc_id: str = Field(...)
    page_id: str = Field(...)
    page_num: int = Field(..., ge=1)
    doc_title: str = Field(default="")
    section_title: str | None = Field(default=None)
    score: float = Field(default=0.0)


class Answer(BaseModel):
    """A generated answer (PLAN Appendix B §B.8)."""

    query_id: str = Field(...)
    text: str = Field(..., description="Markdown, includes [k] markers")
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = Field(default=False)
    refusal_reason: str | None = Field(default=None)
    model: str = Field(default="gemini-2.5-flash")
    prompt_template_id: str = Field(default="answer.default")
    prompt_template_version: int = Field(default=1)
    completed_at: datetime = Field(default_factory=datetime.utcnow)
    latency_ms: int = Field(default=0)
    cost_usd: float = Field(default=0.0)


class AnswerDiagnostics(BaseModel):
    """Diagnostics attached to a generated answer."""

    latency_ms: dict[str, int] = Field(
        default_factory=dict,
        description="Latency breakdown per stage: encode, retrieve, rerank, generate, total",
    )
    backends: dict[str, str] = Field(
        default_factory=dict,
        description="Which backend was used for each stage",
    )
    candidate_count: dict[str, int] = Field(
        default_factory=dict,
        description="Candidate counts per channel: visual, text, sparse, fused, reranked",
    )
    flags: dict[str, bool] = Field(
        default_factory=dict,
        description="Feature flags such as degraded_mode, vlm_rerank_skipped",
    )


class AnswerResponse(BaseModel):
    """A complete answer response from the generation stage."""

    answer_markdown: str = Field(..., description="Answer text with inline [N] citation markers")
    citations: list[Citation] = Field(default_factory=list, description="All cited pages")
    confidence: Literal["high", "medium", "low"] = Field(
        default="medium", description="Confidence level of the answer"
    )
    diagnostics: AnswerDiagnostics = Field(default_factory=AnswerDiagnostics)
    query_id: str = Field(..., description="Unique query identifier for telemetry/feedback")


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """A query submitted to the RAG system."""

    question: str = Field(..., min_length=1, max_length=1000)
    mode: Literal["fast", "deep"] = Field(default="fast")
    top_k: int = Field(default=5, ge=1, le=50)
    filters: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Page metadata (used in embed/index payloads)
# ---------------------------------------------------------------------------


class PageMetadata(BaseModel):
    """Metadata for a single rendered page."""

    doc_hash: str
    doc_filename: str
    page_num: int
    page_type: Literal["text", "visual"] = "text"
    dpi: int = 100
    section_path: list[str] = Field(default_factory=list)
    language: str = "en"
    jurisdiction: str | None = None
    doc_type: str | None = None


# ---------------------------------------------------------------------------
# Trace (for replay)
# ---------------------------------------------------------------------------


class TraceSpan(BaseModel):
    """A single span within a trace."""

    name: str = Field(...)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: datetime | None = Field(default=None)
    attributes: dict[str, str | int | float] = Field(default_factory=dict)


class Trace(BaseModel):
    """A complete query trace for replay / observability (PLAN Appendix B §B.9)."""

    query_id: str = Field(...)
    user_hash: str | None = Field(default=None)
    mode: Literal["fast", "deep"] = Field(default="fast")
    corpus_version: str = Field(default="")
    embed_model: str = Field(default="")
    rerank_model: str = Field(default="")
    gen_model: str = Field(default="")
    retrieval_result: RetrievalResult = Field(
        default_factory=lambda: RetrievalResult(query="", candidates=[])
    )
    answer: Answer = Field(
        default_factory=lambda: Answer(query_id="", text="")
    )
    spans: list[TraceSpan] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: datetime | None = Field(default=None)
