"""Prompt authoring for answer generation per PART VIII §8.3.

This module implements strict grounding instructions, refusal on insufficient context,
citation requirements, and deep mode variant per PLAN.md Part VIII §8.3.

Key behaviors:
    - Answer prompt: strict grounding, refusal on insufficient context, [k] citation markers
    - Deep mode: Gemini 2.5 Pro with stronger grounding check (self-critique pass)
    - Citation validation: verify [page_id] citations exist in candidate set
    - Hallucinated citations are stripped and replaced with [?]

Design:
    - Prompts are authored as structured strings, not template engines
    - Input variables are validated at build time
    - Template versions are tracked for telemetry
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

import structlog

from urban_rag.common.types import RetrievalCandidate

logger = structlog.get_logger(__name__, service="generate-prompts")

# ---------------------------------------------------------------------------
# Template version tracking (per PLAN.md §12.11 - prompt engineering as code)
# ---------------------------------------------------------------------------

CURRENT_ANSWER_TEMPLATE_VERSION = 1
CURRENT_DEEP_TEMPLATE_VERSION = 1

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AnswerMode(Enum):
    """Answer generation mode."""

    FAST = "fast"
    DEEP = "deep"


class RefusalReason(Enum):
    """Reasons for refusing to answer a query."""

    OUT_OF_CORPUS = "out_of_corpus"
    HOSTILE = "hostile"
    NON_PLANNING = "non_planning"
    INSUFFICIENT_CONTEXT = "insufficient_context"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PageMetadata:
    """Metadata for a page provided to the answer prompt.

    This is the minimal information needed to ground the answer and format citations.
    """

    page_id: str
    doc_title: str
    page_num: int
    section_title: str | None = None
    image_uri: str | None = None

    def to_citation_label(self) -> str:
        """Return the citation label for this page.

        Format: "{doc_title}, p.{page_num}"
        Example: "NBC 2016 Vol 1, p.142"
        """
        title = self.doc_title or "Document"
        return f"{title}, p.{self.page_num}"

    def to_page_marker(self) -> str:
        """Return the page marker string used in the prompt.

        Format: "[{page_id}] {doc_title}, Section: {section} | Page {num}"
        """
        section = self.section_title or "General"
        return f"[{self.page_id}] {self.doc_title}, Section: {section} | Page {self.page_num}"


@dataclass(frozen=True)
class AnswerPromptConfig:
    """Configuration for building an answer prompt."""

    mode: AnswerMode = AnswerMode.FAST
    question: str = ""
    pages: list[PageMetadata] | None = None
    system_instructions: str | None = None

    # Deep mode specific
    critique_threshold: float = 0.8  # Minimum support score to pass critique


# ---------------------------------------------------------------------------
# System prompt (shared between fast and deep)
# ---------------------------------------------------------------------------

_BASE_SYSTEM_PROMPT = (
    "You are an expert on Indian urban planning regulations including NBC, URDPFI, "
    "state master plans, development control rules (DCRs), IRC codes, BIS standards, "
    "and MoHUA guidance documents.\n\n"
    "DOMAIN SCOPE: You ONLY answer questions about Indian urban planning, building "
    "regulations, zoning, FSI/FAR norms, parking requirements, fire safety codes, "
    "structural design standards, and related planning topics.\n\n"
    "You answer questions strictly from the provided document pages. If a page does "
    "not directly support a claim, do not make that claim. "
    "Cite using [page_id] inline markers.\n\n"
    "CRITICAL RULES:\n"
    "1. Every factual claim MUST be followed by a [page_id] citation in square brackets.\n"
    "2. If none of the provided pages answer the question, say so plainly and explicitly.\n"
    "3. Never fabricate, extrapolate, or infer information not directly present in the "
    "cited pages.\n"
    "4. For ambiguous or incomplete information, note the limitation explicitly.\n"
    "5. Tables and diagrams are valid sources — cite them as you would text.\n"
    '6. If a query is not about Indian urban planning, refuse with: "I only answer '
    'questions about Indian urban planning regulations."\n\n'
    'OUTPUT FORMAT: Markdown with inline [N] citation markers where N is the 1-based '
    'index of the page in the provided list. End with a "Sources" section listing '
    "cited pages with their full names and page numbers."
)


# ---------------------------------------------------------------------------
# Fast mode answer prompt (PART VIII §8.1, §8.3)
# ---------------------------------------------------------------------------

_FAST_ANSWER_USER_PROMPT_TEMPLATE = """## Query: {question}

## Provided Pages ({num_pages} total):
{pages_list}

## Instructions:
1. Read each page carefully, especially tables and diagrams.
2. Compose an answer that directly addresses the query.
3. Use [N] inline citation markers where N is the index of the page (1-based).
4. If the pages don't contain the answer, say so explicitly and refuse to fabricate.
5. End with a "Sources" section listing cited pages.

## Answer:"""


# ---------------------------------------------------------------------------
# Deep mode prompt (PART VIII §8.4)
# ---------------------------------------------------------------------------

_DEEP_ANSWER_USER_PROMPT_TEMPLATE = """## Query: {question}

## Provided Pages ({num_pages} total):
{pages_list}

## Instructions:
1. Read each page carefully, especially tables and diagrams.
2. Compose an answer that directly addresses the query using ONLY information from
   the provided pages.
3. Use [N] inline citation markers where N is the index of the page (1-based).
4. After writing your answer, perform a SELF-CRITIQUE pass:
   - For each factual claim, verify it is DIRECTLY supported by the cited page
   - Check that you have not extrapolated or inferred beyond what the page states
   - Check that cited tables/diagrams actually contain the information attributed
5. If the self-critique reveals unsupported claims, revise or remove them.
6. If the pages don't contain the answer, say so explicitly and refuse.

FINAL OUTPUT FORMAT: Markdown with inline [N] citation markers. End with a
"Sources" section and a "Self-Critique Notes" section documenting revisions.

## Answer (with self-critique):"""


# ---------------------------------------------------------------------------
# Refusal prompts
# ---------------------------------------------------------------------------

_REFUSAL_PROMPTS: dict[RefusalReason, str] = {
    RefusalReason.OUT_OF_CORPUS: (
        "The query cannot be answered from the indexed corpus. The documents cover "
        "Indian urban planning regulations; this query is outside that scope."
    ),
    RefusalReason.HOSTILE: (
        "This query requests content that could facilitate wrongdoing. "
        "I cannot assist with this request."
    ),
    RefusalReason.NON_PLANNING: (
        "I only answer questions about Indian urban planning regulations (zoning, "
        "FSI/FAR, building codes, fire safety, parking, structural standards, etc.). "
        "Please rephrase your question to focus on these topics."
    ),
    RefusalReason.INSUFFICIENT_CONTEXT: (
        "The retrieved pages do not contain sufficient information to answer this "
        "query with confidence. Please try a more specific question or verify the "
        "topic is covered in the indexed documents."
    ),
}


# ---------------------------------------------------------------------------
# Public API - Prompt builders
# ---------------------------------------------------------------------------


def build_answer_prompt(
    question: str,
    candidates: list[RetrievalCandidate],
    mode: AnswerMode = AnswerMode.FAST,
    system_instructions: str | None = None,
) -> tuple[str, str, int]:
    """Build an answer generation prompt for the given query and candidates.

    Per PART VIII §8.1 and §8.3:
        - Strict grounding instructions
        - Citation requirements: use [N] markers where N is the index
        - Refusal on insufficient context
        - Deep mode uses Gemini 2.5 Pro with stronger grounding check

    Args:
        question: The user's natural language question.
        candidates: Retrieved candidates to ground the answer in.
        mode: Generation mode (FAST or DEEP).
        system_instructions: Optional additional system instructions to append.

    Returns:
        A tuple of (system_prompt, user_prompt, template_version).
        - system_prompt: The system instruction string.
        - user_prompt: The formatted user prompt with page metadata.
        - template_version: The template version for telemetry.

    Raises:
        ValueError: If question is empty or candidates is empty for non-refusal.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    if not candidates:
        # This is a "no candidates" case — caller should handle refusal separately
        # We build a minimal prompt that will produce a refusal
        template_version = CURRENT_ANSWER_TEMPLATE_VERSION
        system_prompt = _build_system_prompt(system_instructions)
        user_prompt = _FAST_ANSWER_USER_PROMPT_TEMPLATE.format(
            question=question,
            num_pages=0,
            pages_list="(No pages available — insufficient context)",
        )
        return system_prompt, user_prompt, template_version

    # Build page metadata list
    pages_metadata = _candidates_to_pages_metadata(candidates)

    # Build page list string
    pages_list = _format_pages_list(pages_metadata)

    # Select template based on mode
    if mode == AnswerMode.DEEP:
        user_template = _DEEP_ANSWER_USER_PROMPT_TEMPLATE
        template_version = CURRENT_DEEP_TEMPLATE_VERSION
    else:
        user_template = _FAST_ANSWER_USER_PROMPT_TEMPLATE
        template_version = CURRENT_ANSWER_TEMPLATE_VERSION

    system_prompt = _build_system_prompt(system_instructions)
    user_prompt = user_template.format(
        question=question,
        num_pages=len(pages_metadata),
        pages_list=pages_list,
    )

    logger.debug(
        "answer_prompt_built",
        question=question[:100],
        num_pages=len(pages_metadata),
        mode=mode.value,
        template_version=template_version,
    )

    return system_prompt, user_prompt, template_version


def build_refusal_prompt(
    question: str,
    reason: RefusalReason,
) -> tuple[str, str, int]:
    """Build a refusal response prompt.

    Per PART VIII §8.5:
        - Non-planning queries: refuse with scope statement
        - Out-of-corpus: explain corpus scope
        - Insufficient context: show closest candidates

    Args:
        question: The original user question.
        reason: The reason for refusal.

    Returns:
        A tuple of (system_prompt, user_prompt, template_version).
    """
    system_prompt = _BASE_SYSTEM_PROMPT
    refusal_text = _REFUSAL_PROMPTS.get(reason, _REFUSAL_PROMPTS[RefusalReason.OUT_OF_CORPUS])

    user_prompt = f"""## Query: {question}

## Response:
{refusal_text}

If you believe this is an error, please rephrase your question to focus on
Indian urban planning regulations."""

    return system_prompt, user_prompt, CURRENT_ANSWER_TEMPLATE_VERSION


def validate_citations(
    answer_text: str,
    candidates: list[RetrievalCandidate],
) -> tuple[str, list[int]]:
    """Validate and fix citations in the generated answer.

    Per PART VIII §8.3:
        - Every [N] citation is verified to exist in the candidate set
        - Hallucinated page_ids are stripped and replaced with [?]

    Args:
        answer_text: The generated answer text with [N] citation markers.
        candidates: The candidates that were provided to the generator.

    Returns:
        A tuple of (corrected_answer, invalid_indices).
        - corrected_answer: Answer with hallucinated citations replaced by [?]
        - invalid_indices: List of [N] indices that were hallucinated

    Example:
        >>> candidates = [...]
        >>> answer = "The FSI is 2.5 [1] and parking ratio is 1 per 100 sqm [2]."
        >>> corrected, invalid = validate_citations(answer, candidates)
        >>> # If [3] was hallucinated: corrected has [3] -> [?] and invalid = [3]
    """
    if not answer_text:
        return "", []

    # Build valid page indices (1-based)
    valid_indices = set(range(1, len(candidates) + 1))

    # Find all [N] patterns in answer
    citation_pattern = re.compile(r"\[(\d+)\]")
    matches = citation_pattern.findall(answer_text)

    # Find invalid citations
    invalid_indices: list[int] = []
    corrected_answer = answer_text

    for match in matches:
        idx = int(match)
        if idx not in valid_indices:
            # Hallucinated citation — replace with [?]
            invalid_indices.append(idx)
            corrected_answer = corrected_answer.replace(f"[{idx}]", "[?]")

    if invalid_indices:
        logger.warning(
            "hallucinated_citations_detected",
            invalid_indices=invalid_indices,
            total_citations=len(matches),
        )

    return corrected_answer, invalid_indices


def extract_citations_from_answer(
    answer_text: str,
) -> list[int]:
    """Extract all citation indices from an answer text.

    Args:
        answer_text: The generated answer text.

    Returns:
        List of citation indices found in the text (1-based).
    """
    if not answer_text:
        return []

    citation_pattern = re.compile(r"\[(\d+)\]")
    matches = citation_pattern.findall(answer_text)
    return [int(m) for m in matches]


def build_citations_list(
    candidates: list[RetrievalCandidate],
    cited_indices: list[int],
) -> list[dict]:
    """Build the citations list for the answer response.

    Per PLAN Appendix B §B.8:
        - Each citation includes page_id, doc_title, page_num, section_title, score

    Args:
        candidates: The candidates that were provided.
        cited_indices: The 1-based indices that were actually cited in the answer.

    Returns:
        List of citation dicts for the AnswerResponse.
    """
    citations: list[dict] = []

    for idx in cited_indices:
        if 1 <= idx <= len(candidates):
            candidate = candidates[idx - 1]
            citations.append({
                "idx": idx,
                "doc_id": (
                    candidate.page_id.split("#")[0] if "#" in candidate.page_id else ""
                ),
                "page_id": candidate.page_id,
                "page_num": (
                    int(candidate.page_id.split("#p")[1])
                    if "#p" in candidate.page_id else 0
                ),
                "doc_title": _extract_doc_title(candidate),
                "section_title": candidate.section_title,
                "score": candidate.rerank_score or candidate.score,
            })

    return citations


# ---------------------------------------------------------------------------
# Deep mode - self-critique prompt
# ---------------------------------------------------------------------------

_SELF_CRITIQUE_PROMPT_TEMPLATE = """## Self-Critique Pass

Review your answer above for factual accuracy against the provided pages.

Check each [N] citation:
- Does page {page_num} actually support the claim associated with [{idx}]?
- Did you extrapolate or infer beyond what the page explicitly states?
- Are cited tables or diagrams being interpreted correctly?

For each issue found, note it and then provide a REVISED version of the claim.

If ALL citations are fully supported, respond: "All claims verified - no revisions needed."

If revisions are needed:
1. List each problematic claim and why it is unsupported
2. Provide the corrected version or note if the claim should be removed
3. Update the citation to [?] if the claim cannot be supported

## Self-Critique Notes:"""


def build_self_critique_prompt(
    answer_text: str,
    candidates: list[RetrievalCandidate],
) -> str | None:
    """Build a self-critique prompt for deep mode verification.

    Per PART VIII §8.4:
        - After initial answer, run self-critique pass
        - If self-critique finds unsupported claims, regenerate with stronger grounding

    Args:
        answer_text: The initially generated answer text.
        candidates: The candidates that were cited.

    Returns:
        The self-critique prompt string, or None if no citations to verify.
    """
    if not answer_text or not candidates:
        return None

    cited_indices = extract_citations_from_answer(answer_text)
    if not cited_indices:
        return None

    # Build critique instructions for each cited page
    page_instructions = []
    for idx in cited_indices:
        if 1 <= idx <= len(candidates):
            candidate = candidates[idx - 1]
            page_num = int(candidate.page_id.split("#p")[1]) if "#p" in candidate.page_id else idx
            page_instructions.append(f"- [{idx}] Verify claims citing page {page_num}")

    if not page_instructions:
        return None

    return _SELF_CRITIQUE_PROMPT_TEMPLATE.format(
        page_num=page_instructions[0].split()[-2] if page_instructions else "?",
        idx=cited_indices[0] if cited_indices else "?",
    ) + "\n" + "\n".join(page_instructions)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _build_system_prompt(additional_instructions: str | None) -> str:
    """Build the system prompt with optional additional instructions."""
    if additional_instructions:
        return f"{_BASE_SYSTEM_PROMPT}\n\n## Additional Instructions:\n{additional_instructions}"
    return _BASE_SYSTEM_PROMPT


def _candidates_to_pages_metadata(
    candidates: list[RetrievalCandidate],
) -> list[PageMetadata]:
    """Convert RetrievalCandidates to PageMetadata for prompt building."""
    pages: list[PageMetadata] = []

    for candidate in candidates:
        # Extract doc_title from page_id or use a default
        doc_title = _extract_doc_title(candidate)

        # Extract page_num from page_id (format: "doc_hash#pNNNN")
        page_num = 0
        if "#p" in candidate.page_id:
            try:
                page_num = int(candidate.page_id.split("#p")[1])
            except (ValueError, IndexError):
                page_num = 0

        pages.append(PageMetadata(
            page_id=candidate.page_id,
            doc_title=doc_title,
            page_num=page_num,
            section_title=candidate.section_title,
            image_uri=candidate.page_image_uri,
        ))

    return pages


def _extract_doc_title(candidate: RetrievalCandidate) -> str:
    """Extract a human-readable document title from a candidate."""
    # The page_id format is typically "doc_hash#pNNNN"
    # We use the first part as a display name
    if "#" in candidate.page_id:
        doc_hash = candidate.page_id.split("#")[0]
        # Truncate hash for display (first 8 chars + ellipsis)
        return f"Document {doc_hash[:8]}..."
    return "Document"


def _format_pages_list(pages: list[PageMetadata]) -> str:
    """Format the pages list for the prompt."""
    if not pages:
        return "(No pages available)"

    lines = []
    for i, page in enumerate(pages, start=1):
        marker = page.to_page_marker()
        lines.append(f"Page {i}: {marker}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def smoke_test() -> dict:
    """Run a smoke test of the prompt module.

    Verifies:
        - Answer prompt builds correctly
        - Citation validation works
        - Refusal prompts build correctly
        - Deep mode prompt differs from fast mode

    Returns:
        Dict with smoke test results.
    """
    from urban_rag.common.types import RetrievalCandidate

    # Create synthetic candidates
    candidates = [
        RetrievalCandidate(
            page_id="doc1#p001",
            score=0.8,
            channel_scores={"visual": 10.0},
            channel_ranks={"visual": 1},
            page_image_uri="s3://doc1/p001.png",
            extracted_text_excerpt="FSI for residential zones is 2.5",
            section_title="3.2 Residential Zone Regulations",
        ),
        RetrievalCandidate(
            page_id="doc2#p010",
            score=0.75,
            channel_scores={"visual": 9.0},
            channel_ranks={"visual": 2},
            page_image_uri="s3://doc2/p010.png",
            extracted_text_excerpt="Parking requirements: 1 per 100 sqm",
            section_title="5.1 Parking Standards",
        ),
    ]

    # Test answer prompt (fast mode)
    system_prompt, user_prompt, version = build_answer_prompt(
        question="What is the FSI for residential zones?",
        candidates=candidates,
        mode=AnswerMode.FAST,
    )
    assert system_prompt, "System prompt should not be empty"
    assert user_prompt, "User prompt should not be empty"
    assert "What is the FSI" in user_prompt, "Question should be in prompt"
    assert "Page 1:" in user_prompt, "Pages should be listed"
    assert version == CURRENT_ANSWER_TEMPLATE_VERSION, "Version should match"

    # Test answer prompt (deep mode)
    _, deep_prompt, deep_version = build_answer_prompt(
        question="What is the FSI for residential zones?",
        candidates=candidates,
        mode=AnswerMode.DEEP,
    )
    assert "SELF-CRITIQUE" in deep_prompt, "Deep mode should include self-critique"
    assert deep_version == CURRENT_DEEP_TEMPLATE_VERSION, "Deep version should match"

    # Test citation validation
    test_answer = "The FSI is 2.5 [1] and parking is 1 per 100 sqm [2]. Unknown [99]."
    corrected, invalid = validate_citations(test_answer, candidates)
    assert corrected == "The FSI is 2.5 [1] and parking is 1 per 100 sqm [2]. Unknown [?].", \
        f"Hallucinated [99] should be replaced with [?], got: {corrected}"
    assert 99 in invalid, "[99] should be in invalid list"
    assert 1 not in invalid, "[1] is valid and should not be in invalid list"
    assert 2 not in invalid, "[2] is valid and should not be in invalid list"

    # Test citation extraction
    cited = extract_citations_from_answer(test_answer)
    assert cited == [1, 2, 99], f"Should extract [1], [2], [99], got: {cited}"

    # Test citations list building
    cited_indices = [1, 2]
    citations_list = build_citations_list(candidates, cited_indices)
    assert len(citations_list) == 2, f"Should have 2 citations, got {len(citations_list)}"
    assert citations_list[0]["idx"] == 1
    assert citations_list[0]["page_id"] == "doc1#p001"
    assert citations_list[1]["idx"] == 2
    assert citations_list[1]["page_id"] == "doc2#p010"

    # Test refusal prompt
    _, refusal_prompt, _ = build_refusal_prompt(
        question="What's the weather in Mumbai?",
        reason=RefusalReason.NON_PLANNING,
    )
    assert "I only answer questions about Indian urban planning" in refusal_prompt

    # Test empty candidates handling
    _, user_prompt_empty, _ = build_answer_prompt(
        question="What is FSI?",
        candidates=[],
    )
    assert "No pages available" in user_prompt_empty

    logger.info("prompt_smoke_test_passed")

    return {
        "passed": True,
        "fast_prompt_length": len(user_prompt),
        "deep_prompt_length": len(deep_prompt),
        "citation_validation_works": True,
        "refusal_prompt_works": True,
    }
