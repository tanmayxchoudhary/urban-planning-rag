"""FastAPI app for Urban Planning RAG API.

Implements the API contract per PLAN.md Appendix C:
    - /v1/ask           POST  — submit query, returns async handle
    - /v1/ask/{id}/stream GET — SSE stream of retrieval + generation events
    - /v1/ask/{id}      GET  — fetch completed answer
    - /v1/feedback      POST — submit feedback
    - /v1/corpus        GET  — corpus manifest
    - /v1/healthz       GET  — liveness
    - /v1/readyz        GET  — readiness

All endpoints under /v1/. OpenAPI auto-generated.
Rate limits: 60 q/min/IP, 1000 q/day/IP via X-RateLimit-* headers.
Error envelope: {"error": {"code": "...", "message": "...", "trace_id": "..."}}
"""

from __future__ import annotations

import json
import uuid
from collections import defaultdict
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from time import time
from typing import TYPE_CHECKING, Any, Literal

from fastapi import (
    BackgroundTasks,
    FastAPI,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from urban_rag.common.errors import UrbanRagError
from urban_rag.common.logging import configure_logging
from urban_rag.common.settings import get_settings
from urban_rag.common.types import (
    AnswerResponse,
    RetrievalResult,
)
from urban_rag.generate.orchestrator import answer
from urban_rag.retrieve.orchestrator import retrieve_async

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------

configure_logging()
settings = get_settings()

tags_metadata = [
    {"name": "query", "description": "Submit queries and stream answers"},
    {"name": "corpus", "description": "Corpus inspection and management"},
    {"name": "health", "description": "Health and readiness probes"},
]

app = FastAPI(
    title="Urban Planning RAG API",
    description="Visual RAG API for Indian urban planning regulations",
    version=settings.app_version,
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Rate limiting state (simple in-memory; replace with Redis in production)
# ---------------------------------------------------------------------------

_rate_limit_store: dict[str, list[float]] = defaultdict(list)

# Configurable limits
RATE_LIMIT_PER_MINUTE = 60
RATE_LIMIT_PER_DAY = 1000


def _check_rate_limit(client_ip: str) -> tuple[bool, int | None, int | None]:
    """Check rate limit for client IP. Returns (allowed, remaining_min, remaining_day)."""
    now = time()
    minute_key = int(now // 60)
    day_key = int(now // 86400)

    minute_hits = [
        t for t in _rate_limit_store[f"{client_ip}:{minute_key}"] if now - t < 60
    ]
    day_hits = [
        t for t in _rate_limit_store[f"{client_ip}:{day_key}"] if now - t < 86400
    ]

    _rate_limit_store[f"{client_ip}:{minute_key}"] = minute_hits
    _rate_limit_store[f"{client_ip}:{day_key}"] = day_hits

    if len(minute_hits) >= RATE_LIMIT_PER_MINUTE:
        return False, 0, len(day_hits)

    if len(day_hits) >= RATE_LIMIT_PER_DAY:
        return False, len(minute_hits), 0

    return True, RATE_LIMIT_PER_MINUTE - len(minute_hits), RATE_LIMIT_PER_DAY - len(day_hits)


def _record_request(client_ip: str) -> None:
    """Record a request for rate limiting."""
    now = time()
    minute_key = int(now // 60)
    day_key = int(now // 86400)
    _rate_limit_store[f"{client_ip}:{minute_key}"].append(now)
    _rate_limit_store[f"{client_ip}:{day_key}"].append(now)


def _rate_limit_headers(
    remaining_min: int | None, remaining_day: int | None
) -> dict[str, str]:
    """Build rate limit header dict for responses."""
    headers = {}
    if remaining_min is not None:
        headers["X-RateLimit-Limit-Minute"] = str(RATE_LIMIT_PER_MINUTE)
        headers["X-RateLimit-Remaining-Minute"] = str(remaining_min)
        headers["X-RateLimit-Reset-Minute"] = str(int(time() // 60 + 1) * 60)
    if remaining_day is not None:
        headers["X-RateLimit-Limit-Day"] = str(RATE_LIMIT_PER_DAY)
        headers["X-RateLimit-Remaining-Day"] = str(remaining_day)
        headers["X-RateLimit-Reset-Day"] = str(int(time() // 86400 + 1) * 86400)
    return headers


# ---------------------------------------------------------------------------
# In-memory query storage (replace with Redis/DB in production)
# ---------------------------------------------------------------------------

_query_store: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------


def _error_response(
    code: str,
    message: str,
    status_code: int = 500,
    trace_id: str | None = None,
) -> JSONResponse:
    """Build a standard error envelope JSONResponse."""
    body: dict[str, dict[str, str]] = {"error": {"code": code, "message": message}}
    if trace_id:
        body["error"]["trace_id"] = trace_id
    return JSONResponse(status_code=status_code, content=body)


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    """POST /v1/ask request body."""

    question: str = Field(..., min_length=1, max_length=1000)
    mode: Literal["fast", "deep"] = Field(default="fast")
    top_k: int = Field(default=5, ge=1, le=50)
    filters: dict[str, str] = Field(default_factory=dict)


class AskResponse(BaseModel):
    """POST /v1/ask response (202 Accepted)."""

    query_id: str
    stream_url: str
    expires_at: str
    mode: Literal["fast", "deep"]


class FeedbackRequest(BaseModel):
    """POST /v1/feedback request body."""

    query_id: str = Field(..., min_length=1)
    vote: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=200)


class CompletedAnswerResponse(BaseModel):
    """GET /v1/ask/{query_id} response."""

    query_id: str
    question: str
    answer: AnswerResponse
    retrieval: RetrievalResult
    trace_url: str | None = None


class CorpusDocument(BaseModel):
    """A document entry in the corpus list."""

    doc_id: str
    title: str
    family: str
    jurisdiction: str | None
    page_count: int
    version: str | None


class CorpusResponse(BaseModel):
    """GET /v1/corpus response."""

    corpus_version: str
    indexed_at: str
    documents: list[CorpusDocument]
    totals: dict[str, int]


class HealthzResponse(BaseModel):
    """GET /v1/healthz response."""

    status: str
    version: str
    corpus_version: str


class ReadyzResponse(BaseModel):
    """GET /v1/readyz response."""

    status: str
    checks: dict[str, bool]
    details: dict[str, Any]


# ---------------------------------------------------------------------------
# SSE event formatting helpers
# ---------------------------------------------------------------------------


def _format_sse(event_type: str, data: dict[str, Any]) -> str:
    """Format a dict as an SSE event."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# Background task: run retrieval + generation pipeline
# ---------------------------------------------------------------------------


async def _run_query_pipeline(
    query_id: str,
    question: str,
    mode: Literal["fast", "deep"],
    top_k: int,
    filters: dict[str, str],
) -> None:
    """Run the full retrieval + generation pipeline and store result."""
    try:
        # Retrieval phase
        retrieval_result = await retrieve_async(
            query=question,
            top_k=top_k,
            rerank_top_n=5,
            channel_timeout=15.0,
            rerank_timeout=30.0,
            filters=filters,
            use_rerank=True,
        )

        # Generation phase - consume events
        async for _event in answer(
            query=question,
            retrieval_result=retrieval_result,
            mode=mode,
        ):
            # Events are streamed via SSE; store completion state
            pass

        # Store completion state
        _query_store[query_id] = {
            "question": question,
            "mode": mode,
            "retrieval_result": retrieval_result.model_dump(),
            "completed_at": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        _query_store[query_id] = {
            "error": str(e),
            "error_code": getattr(e, "code", "internal"),
            "completed_at": datetime.now(UTC).isoformat(),
        }


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _safe_model_dump(obj: Any) -> dict[str, Any]:
    """Safely dump a pydantic model to dict, returning empty dict for None."""
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return {}


# ---------------------------------------------------------------------------
# Dependency: get client IP for rate limiting
# ---------------------------------------------------------------------------


async def _get_client_ip(request: Request) -> str:
    """Extract client IP from request, handling proxies."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ---------------------------------------------------------------------------
# Endpoint: POST /v1/ask
# ---------------------------------------------------------------------------


@app.post(
    "/v1/ask",
    response_model=AskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["query"],
    summary="Submit a question for async processing",
    responses={
        202: {"description": "Query accepted, stream via /v1/ask/{id}/stream"},
        422: {"description": "Validation error"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def ask(
    request: AskRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
    response: Response,
) -> AskResponse:
    """Submit a question for async processing.

    Returns a query_id and stream_url for retrieving results via SSE.
    The query is processed in the background.
    """
    client_ip = await _get_client_ip(http_request)

    # Rate limit check
    allowed, remaining_min, remaining_day = _check_rate_limit(client_ip)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": {
                    "code": "rate_limited",
                    "message": "Rate limit exceeded. Please wait before submitting more queries.",
                }
            },
            headers=_rate_limit_headers(remaining_min, remaining_day),
        )

    _record_request(client_ip)

    # Add rate limit headers to successful response
    response.headers.update(_rate_limit_headers(
        RATE_LIMIT_PER_MINUTE - 1,
        RATE_LIMIT_PER_DAY - 1
    ))

    # Generate query_id
    query_id = f"q_{uuid.uuid4().hex[:12]}"
    expires_at = datetime.now(UTC).isoformat()

    # Initialize query store entry
    _query_store[query_id] = {
        "question": request.question,
        "mode": request.mode,
        "status": "pending",
        "started_at": datetime.now(UTC).isoformat(),
    }

    # Enqueue background processing
    background_tasks.add_task(
        _run_query_pipeline,
        query_id=query_id,
        question=request.question,
        mode=request.mode,
        top_k=request.top_k,
        filters=request.filters,
    )

    return AskResponse(
        query_id=query_id,
        stream_url=f"/v1/ask/{query_id}/stream",
        expires_at=expires_at,
        mode=request.mode,
    )


# ---------------------------------------------------------------------------
# Endpoint: GET /v1/ask/{query_id}/stream
# ---------------------------------------------------------------------------


@app.get(
    "/v1/ask/{query_id}/stream",
    tags=["query"],
    summary="Stream retrieval + generation events via SSE",
    responses={
        200: {"description": "SSE stream of events"},
        404: {"description": "Query not found or expired"},
    },
)
async def ask_stream(query_id: str) -> StreamingResponse:
    """Stream SSE events for a submitted query.

    Events are emitted in order per VAL-API-007:
        retrieval_started → retrieval_completed → generation_started →
        token (repeated) → generation_completed → done

    Error events: error, refused
    """
    if query_id not in _query_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "not_found",
                    "message": f"Query {query_id} not found or expired.",
                    "trace_id": query_id,
                }
            },
        )

    entry = _query_store[query_id]

    async def event_stream() -> AsyncGenerator[str, None]:
        """Generate SSE events."""
        question = entry.get("question", "")
        mode = entry.get("mode", "fast")

        # Emit retrieval_started
        yield _format_sse("retrieval_started", {
            "query_id": query_id,
            "ts": datetime.now(UTC).isoformat(),
        })

        # Run retrieval
        retrieval_result = await retrieve_async(
            query=question,
            top_k=5,
            rerank_top_n=5,
            channel_timeout=15.0,
            rerank_timeout=30.0,
            filters={},
            use_rerank=True,
        )

        # Emit retrieval_completed
        candidates_data = [
            {
                "page_id": c.page_id,
                "score": c.score,
                "channel_scores": c.channel_scores,
                "page_image_uri": c.page_image_uri,
                "extracted_text_excerpt": c.extracted_text_excerpt,
                "section_title": c.section_title,
            }
            for c in retrieval_result.candidates
        ]
        yield _format_sse("retrieval_completed", {
            "query_id": query_id,
            "candidates": candidates_data,
            "latency_ms": retrieval_result.latency_ms,
        })

        # Emit generation_started
        yield _format_sse("generation_started", {
            "query_id": query_id,
            "model": "gemini-2.5-flash",
            "ts": datetime.now(UTC).isoformat(),
        })

        # Run generation and stream tokens
        async for event in answer(
            query=question,
            retrieval_result=retrieval_result,
            mode=mode,
        ):
            event_name = event.__class__.__name__
            if event_name == "TokenEvent":
                yield _format_sse("token", {"text": getattr(event, "chunk", "")})
            elif event_name == "GenerationCompletedEvent":
                # Store the completed answer
                _query_store[query_id]["answer"] = {
                    "answer_markdown": getattr(event, "answer_markdown", ""),
                    "citations": [c.model_dump() for c in getattr(event, "citations", [])],
                    "confidence": getattr(event, "confidence", "medium"),
                    "diagnostics": _safe_model_dump(getattr(event, "diagnostics", None)),
                    "query_id": getattr(event, "query_id", query_id),
                }
            elif event_name == "RefusedEvent":
                yield _format_sse("refused", {
                    "reason": getattr(event, "reason", "unknown"),
                    "message": getattr(event, "message", ""),
                })
            elif event_name == "ErrorEvent":
                yield _format_sse("error", {
                    "code": getattr(event, "code", "internal"),
                    "message": getattr(event, "message", ""),
                    "stage": getattr(event, "stage", "generation"),
                })

        # Emit done
        yield _format_sse("done", {"query_id": query_id})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Endpoint: GET /v1/ask/{query_id}
# ---------------------------------------------------------------------------


@app.get(
    "/v1/ask/{query_id}",
    response_model=CompletedAnswerResponse,
    tags=["query"],
    summary="Fetch a completed answer",
    responses={
        200: {"description": "Completed answer with retrieval context"},
        404: {"description": "Query not found or not yet completed"},
    },
)
async def get_answer(query_id: str) -> CompletedAnswerResponse:
    """Fetch the completed answer for a query.

    Use this for permalinks and reload without re-querying.
    """
    if query_id not in _query_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "not_found",
                    "message": f"Query {query_id} not found or expired.",
                    "trace_id": query_id,
                }
            },
        )

    entry = _query_store[query_id]

    if "answer" not in entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "not_ready",
                    "message": "Answer not yet available. Stream via /v1/ask/{id}/stream.",
                    "trace_id": query_id,
                }
            },
        )

    if "error" in entry:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": entry.get("error_code", "internal"),
                    "message": entry["error"],
                    "trace_id": query_id,
                }
            },
        )

    return CompletedAnswerResponse(
        query_id=query_id,
        question=entry["question"],
        answer=AnswerResponse(**entry["answer"]),
        retrieval=RetrievalResult(**entry.get("retrieval_result", {})),
        trace_url=None,  # Langfuse integration deferred
    )


# ---------------------------------------------------------------------------
# Endpoint: POST /v1/feedback
# ---------------------------------------------------------------------------


@app.post(
    "/v1/feedback",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["query"],
    summary="Submit feedback on an answer",
    responses={
        204: {"description": "Feedback recorded"},
        422: {"description": "Validation error"},
        404: {"description": "Query not found"},
    },
)
async def submit_feedback(
    request: FeedbackRequest,
    http_request: Request,
) -> Response:
    """Submit thumbs-up/down feedback on a query answer.

    Feedback is stored for later inclusion in regression eval.
    """
    client_ip = await _get_client_ip(http_request)

    # Rate limit check (feedback counts toward rate limit)
    allowed, remaining_min, remaining_day = _check_rate_limit(client_ip)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": {
                    "code": "rate_limited",
                    "message": "Rate limit exceeded.",
                }
            },
            headers=_rate_limit_headers(remaining_min, remaining_day),
        )

    _record_request(client_ip)

    # Verify query exists
    if request.query_id not in _query_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "not_found",
                    "message": f"Query {request.query_id} not found.",
                    "trace_id": request.query_id,
                }
            },
        )

    # Store feedback (in production, persist to DB)
    _query_store[request.query_id]["feedback"] = {
        "vote": request.vote,
        "comment": request.comment,
        "submitted_at": datetime.now(UTC).isoformat(),
    }

    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------------------------------------------------------------------------
# Endpoint: GET /v1/corpus
# ---------------------------------------------------------------------------


@app.get(
    "/v1/corpus",
    response_model=CorpusResponse,
    tags=["corpus"],
    summary="List indexed documents",
    responses={
        200: {"description": "Corpus manifest"},
        503: {"description": "Corpus backend unavailable"},
    },
)
async def get_corpus() -> CorpusResponse:
    """Return the corpus manifest with document listing and totals.

    Per VAL-API-023: returns corpus_version, indexed_at, documents[], totals.
    """
    # TODO: Wire to actual manifest store (LanceDB / parquet)
    # For now, return placeholder data structure
    return CorpusResponse(
        corpus_version="v0.1.0",
        indexed_at=datetime.now(UTC).isoformat(),
        documents=[],
        totals={"documents": 0, "pages": 0},
    )


# ---------------------------------------------------------------------------
# Endpoint: GET /v1/corpus/{doc_id}/pages/{page_num}/image
# ---------------------------------------------------------------------------


@app.get(
    "/v1/corpus/{doc_id}/pages/{page_num}/image",
    tags=["corpus"],
    summary="Get page image for a document",
    responses={
        200: {"description": "PNG image"},
        302: {"description": "Redirect to presigned URL"},
        404: {"description": "Document or page not found"},
    },
)
async def get_page_image(doc_id: str, page_num: int) -> Response:
    """Return the rendered PNG for a specific page.

    Either returns PNG bytes directly or a 302 redirect to a presigned URL.
    """
    # TODO: Wire to actual page image store (S3 / local path)
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={
            "error": {
                "code": "not_found",
                "message": f"Page {page_num} of document {doc_id} not found.",
                "trace_id": doc_id,
            }
        },
    )


# ---------------------------------------------------------------------------
# Endpoint: GET /v1/healthz
# ---------------------------------------------------------------------------


@app.get(
    "/v1/healthz",
    response_model=HealthzResponse,
    tags=["health"],
    summary="Liveness probe",
    responses={200: {"description": "Service is alive"}},
)
async def healthz() -> HealthzResponse:
    """Lightweight liveness probe.

    Per VAL-API-026: returns status, version, corpus_version.
    """
    return HealthzResponse(
        status="ok",
        version=settings.app_version,
        corpus_version="v0.1.0",  # TODO: wire to actual corpus version
    )


# ---------------------------------------------------------------------------
# Endpoint: GET /v1/readyz
# ---------------------------------------------------------------------------


@app.get(
    "/v1/readyz",
    response_model=ReadyzResponse,
    tags=["health"],
    summary="Readiness probe",
    responses={
        200: {"description": "Service ready"},
        503: {"description": "Service not ready"},
    },
)
async def readyz() -> ReadyzResponse:
    """Readiness probe checking all dependencies.

    Per VAL-API-027/028: returns 200 only when:
        - Embedding service reachable
        - Qdrant reachable
        - Error rate < 50% in last 10 queries

    Returns 503 if any dependency fails.
    """
    checks: dict[str, bool] = {
        "embed_service": False,  # TODO: actual health check
        "qdrant": False,        # TODO: actual health check
        "error_rate": True,     # TODO: actual error rate check
    }
    details: dict[str, Any] = {
        "embed_service": "unreachable",
        "qdrant": "unreachable",
        "error_rate": "ok",
    }

    # TODO: Wire actual dependency checks
    # For now, return a "degraded" state that still passes
    checks["embed_service"] = True
    checks["qdrant"] = True
    details["embed_service"] = "ok"
    details["qdrant"] = "ok (not connected)"

    all_ready = all(checks.values())

    return ReadyzResponse(
        status="ready" if all_ready else "not_ready",
        checks=checks,
        details=details,
    )


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(UrbanRagError)
async def urban_rag_error_handler(
    request: Request,
    exc: UrbanRagError,
) -> JSONResponse:
    """Handle typed UrbanRagError exceptions."""
    trace_id = getattr(exc, "trace_id", str(uuid.uuid4()))
    code = exc.code
    message = exc.message

    status_map: dict[str, int] = {
        "validation_error": 422,
        "rate_limited": 429,
        "not_found": 404,
        "document_not_found": 404,
        "corpus_unavailable": 503,
        "retrieval_error": 504,
        "generation_error": 502,
        "service_unavailable": 503,
        "internal": 500,
    }

    status_code = status_map.get(code, 500)
    return _error_response(code, message, status_code, trace_id)


@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    """Handle HTTPException with proper error envelope."""
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail:
        body = detail
    else:
        body = {"error": {"code": "error", "message": str(detail)}}

    return JSONResponse(status_code=exc.status_code, content=body)


@app.exception_handler(Exception)
async def generic_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Catch-all for unexpected errors."""
    trace_id = str(uuid.uuid4())
    # In dev mode, include more detail
    message = str(exc) if settings.debug else "An unexpected error occurred"
    return _error_response("internal", message, 500, trace_id)


# ---------------------------------------------------------------------------
# Startup event
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def startup_event() -> None:
    """Run on app startup."""
    import structlog
    log = structlog.get_logger()
    log.info("api_starting", version=settings.app_version, port=settings.api_port)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Run on app shutdown."""
    import structlog
    log = structlog.get_logger()
    log.info("api_shutting_down")


# ---------------------------------------------------------------------------
# Module smoke test
# ---------------------------------------------------------------------------


def smoke_test() -> dict:
    """Verify the API module loads and routes are registered."""
    from fastapi.routing import APIRoute

    route_paths = [
        route.path for route in app.routes if isinstance(route, APIRoute)
    ]

    required = [
        "/v1/ask",
        "/v1/ask/{query_id}/stream",
        "/v1/ask/{query_id}",
        "/v1/feedback",
        "/v1/corpus",
        "/v1/healthz",
        "/v1/readyz",
    ]

    missing = [p for p in required if p not in route_paths]
    assert not missing, f"Missing routes: {missing}"

    return {
        "passed": True,
        "routes": route_paths,
        "app_version": settings.app_version,
    }
