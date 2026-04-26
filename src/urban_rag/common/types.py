"""Shared pydantic types for Urban RAG."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A citation reference within an answer."""

    page_id: str = Field(..., description="Unique page identifier (doc_hash:page_num)")
    doc_hash: str = Field(..., description="SHA256 of the source document")
    doc_filename: str = Field(..., description="Original filename")
    page_num: int = Field(..., ge=1, description="1-based page number")
    section_path: list[str] = Field(
        default_factory=list, description="Hierarchical section path, e.g. [NBC 2016, Part 4, 4.2]"
    )
    image_uri: str = Field(..., description="URI to the rendered page PNG")
    rerank_score: float | None = Field(
        default=None, description="VLM rerank score if applicable"
    )


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


class QueryRequest(BaseModel):
    """A query submitted to the RAG system."""

    question: str = Field(..., min_length=1, max_length=1000)
    mode: Literal["fast", "deep"] = Field(default="fast")
    top_k: int = Field(default=5, ge=1, le=50)
    filters: dict[str, str] = Field(default_factory=dict)


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


class DocumentRecord(BaseModel):
    """A document as recorded in the corpus manifest."""

    doc_hash: str
    filename: str
    num_pages: int
    num_visual_patches: int = 0
    num_text_chunks: int = 0
    language: str = "en"
    source_jurisdiction: str | None = None
    doc_type: str | None = None
    ingested_at: datetime
    indexed_at: datetime | None = None
    eval_status: Literal["pending", "passed", "failed"] = "pending"
    version: int = 1
    parser_used: str | None = None
    parser_warnings: list[str] = Field(default_factory=list)
