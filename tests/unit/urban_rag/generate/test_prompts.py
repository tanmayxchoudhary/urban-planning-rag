"""Unit tests for the generate prompts module.

Tests the prompt authoring per PART VIII §8.3:
    - Strict grounding instructions
    - Refusal on insufficient context
    - Citation requirements
    - Deep mode variant with stronger grounding check
"""

import pytest

from urban_rag.common.types import RetrievalCandidate
from urban_rag.generate.prompts import (
    AnswerMode,
    RefusalReason,
    PageMetadata,
    build_answer_prompt,
    build_refusal_prompt,
    validate_citations,
    extract_citations_from_answer,
    build_citations_list,
    build_self_critique_prompt,
    smoke_test,
    CURRENT_ANSWER_TEMPLATE_VERSION,
    CURRENT_DEEP_TEMPLATE_VERSION,
)


class TestBuildAnswerPrompt:
    """Tests for build_answer_prompt()."""

    @pytest.fixture
    def sample_candidates(self) -> list[RetrievalCandidate]:
        """Create sample retrieval candidates for testing."""
        return [
            RetrievalCandidate(
                page_id="doc1#p001",
                score=0.8,
                channel_scores={"visual": 10.0, "text": 0.8},
                channel_ranks={"visual": 1, "text": 2},
                page_image_uri="s3://doc1/p001.png",
                extracted_text_excerpt="FSI for residential zones is 2.5 according to Mumbai DCR.",
                section_title="3.2 Residential Zone Regulations",
            ),
            RetrievalCandidate(
                page_id="doc2#p010",
                score=0.75,
                channel_scores={"visual": 9.0, "text": 0.75},
                channel_ranks={"visual": 2, "text": 1},
                page_image_uri="s3://doc2/p010.png",
                extracted_text_excerpt="Parking requirements: 1 space per 100 sqm of floor area.",
                section_title="5.1 Parking Standards",
            ),
            RetrievalCandidate(
                page_id="nbc_abc123#p142",
                score=0.7,
                channel_scores={"visual": 8.5, "text": 0.7},
                channel_ranks={"visual": 3, "text": 3},
                page_image_uri="s3://nbc/p142.png",
                extracted_text_excerpt="Fire safety requirements for high-rise buildings.",
                section_title="Part 4 Section 4.3",
            ),
        ]

    def test_build_answer_prompt_returns_system_and_user(self, sample_candidates):
        """build_answer_prompt returns both system and user prompts."""
        system, user, version = build_answer_prompt(
            question="What is FSI for residential zones?",
            candidates=sample_candidates,
        )
        assert system, "System prompt should not be empty"
        assert user, "User prompt should not be empty"
        assert isinstance(system, str)
        assert isinstance(user, str)
        assert isinstance(version, int)

    def test_build_answer_prompt_includes_question(self, sample_candidates):
        """Question appears in the user prompt."""
        question = "What is the FSI for residential zones?"
        _, user, _ = build_answer_prompt(question=question, candidates=sample_candidates)
        assert question in user

    def test_build_answer_prompt_includes_pages(self, sample_candidates):
        """Pages are listed in the prompt with their metadata."""
        _, user, _ = build_answer_prompt(
            question="FSI question",
            candidates=sample_candidates,
        )
        assert "Page 1:" in user
        assert "Page 2:" in user
        assert "Page 3:" in user
        assert "doc1#p001" in user
        assert "Section: 3.2 Residential Zone Regulations" in user

    def test_build_answer_prompt_citations_rule(self, sample_candidates):
        """Prompt instructs to use [N] citation markers."""
        _, user, _ = build_answer_prompt(
            question="FSI question",
            candidates=sample_candidates,
        )
        assert "[N]" in user or "[1]" in user, "Should instruct to use [N] citation markers"
        assert "inline" in user.lower() or "citation" in user.lower()

    def test_build_answer_prompt_refusal_instruction(self, sample_candidates):
        """Prompt instructs to refuse when pages don't answer."""
        _, user, _ = build_answer_prompt(
            question="Unanswerable question",
            candidates=sample_candidates,
        )
        assert "don't contain the answer" in user.lower() or "refuse" in user.lower() or "say so" in user.lower()

    def test_build_answer_prompt_sources_section(self, sample_candidates):
        """Prompt instructs to end with Sources section."""
        _, user, _ = build_answer_prompt(
            question="FSI question",
            candidates=sample_candidates,
        )
        assert "Sources" in user

    def test_build_answer_prompt_fast_mode(self, sample_candidates):
        """Fast mode uses the standard template."""
        _, user_fast, version_fast = build_answer_prompt(
            question="FSI question",
            candidates=sample_candidates,
            mode=AnswerMode.FAST,
        )
        assert "SELF-CRITIQUE" not in user_fast, "Fast mode should not have self-critique"
        assert version_fast == CURRENT_ANSWER_TEMPLATE_VERSION

    def test_build_answer_prompt_deep_mode(self, sample_candidates):
        """Deep mode uses the enhanced template with self-critique."""
        _, user_deep, version_deep = build_answer_prompt(
            question="FSI question",
            candidates=sample_candidates,
            mode=AnswerMode.DEEP,
        )
        assert "SELF-CRITIQUE" in user_deep, "Deep mode should include self-critique"
        assert version_deep == CURRENT_DEEP_TEMPLATE_VERSION

    def test_build_answer_prompt_deep_mode_stronger_grounding(self, sample_candidates):
        """Deep mode includes stronger grounding check instructions."""
        _, user_deep, _ = build_answer_prompt(
            question="FSI question",
            candidates=sample_candidates,
            mode=AnswerMode.DEEP,
        )
        # Deep mode should have more stringent verification
        assert "verify" in user_deep.lower() or "check" in user_deep.lower()

    def test_build_answer_prompt_versioning(self, sample_candidates):
        """Template version is returned for telemetry."""
        _, _, fast_version = build_answer_prompt(
            question="Test",
            candidates=sample_candidates,
            mode=AnswerMode.FAST,
        )
        _, _, deep_version = build_answer_prompt(
            question="Test",
            candidates=sample_candidates,
            mode=AnswerMode.DEEP,
        )
        assert fast_version == CURRENT_ANSWER_TEMPLATE_VERSION
        assert deep_version == CURRENT_DEEP_TEMPLATE_VERSION

    def test_build_answer_prompt_empty_question_raises(self, sample_candidates):
        """Empty question raises ValueError."""
        with pytest.raises(ValueError, match="Question cannot be empty"):
            build_answer_prompt(question="", candidates=sample_candidates)

    def test_build_answer_prompt_whitespace_question_raises(self, sample_candidates):
        """Whitespace-only question raises ValueError."""
        with pytest.raises(ValueError, match="Question cannot be empty"):
            build_answer_prompt(question="   ", candidates=sample_candidates)

    def test_build_answer_prompt_empty_candidates_handled(self):
        """Empty candidates list is handled gracefully (no crash)."""
        system, user, version = build_answer_prompt(
            question="What is FSI?",
            candidates=[],
        )
        assert system, "System prompt should still be returned"
        assert "No pages available" in user
        assert version == CURRENT_ANSWER_TEMPLATE_VERSION

    def test_build_answer_prompt_system_instructions(self, sample_candidates):
        """Additional system instructions are appended."""
        system, _, _ = build_answer_prompt(
            question="Test",
            candidates=sample_candidates,
            system_instructions="Always be concise.",
        )
        assert "Always be concise" in system

    def test_build_answer_prompt_domain_scope(self, sample_candidates):
        """System prompt establishes domain scope."""
        system, _, _ = build_answer_prompt(
            question="Test",
            candidates=sample_candidates,
        )
        assert "Indian urban planning" in system or "urban planning" in system.lower()


class TestBuildRefusalPrompt:
    """Tests for build_refusal_prompt()."""

    def test_build_refusal_prompt_non_planning(self):
        """Non-planning query refusal."""
        _, user, version = build_refusal_prompt(
            question="What's the weather in Mumbai?",
            reason=RefusalReason.NON_PLANNING,
        )
        assert "I only answer questions about Indian urban planning" in user
        assert version == CURRENT_ANSWER_TEMPLATE_VERSION

    def test_build_refusal_prompt_out_of_corpus(self):
        """Out-of-corpus query refusal."""
        _, user, _ = build_refusal_prompt(
            question="How do I bake a cake?",
            reason=RefusalReason.OUT_OF_CORPUS,
        )
        assert "cannot be answered" in user.lower() or "outside" in user.lower() or "not" in user.lower()

    def test_refusal_prompt_insufficient_context(self):
        """Insufficient context refusal."""
        _, user, _ = build_refusal_prompt(
            question="Tell me about buildings",
            reason=RefusalReason.INSUFFICIENT_CONTEXT,
        )
        assert "insufficient" in user.lower() or "do not contain" in user.lower()

    def test_refusal_prompt_hostile(self):
        """Hostile query refusal."""
        _, user, _ = build_refusal_prompt(
            question="How do I forge a building permit?",
            reason=RefusalReason.HOSTILE,
        )
        assert "cannot assist" in user.lower() or "wrongdoing" in user.lower()

    def test_refusal_prompt_version(self):
        """Refusal prompt has version for tracking."""
        _, _, version = build_refusal_prompt(
            question="Test?",
            reason=RefusalReason.NON_PLANNING,
        )
        assert version == CURRENT_ANSWER_TEMPLATE_VERSION


class TestValidateCitations:
    """Tests for validate_citations()."""

    @pytest.fixture
    def sample_candidates(self) -> list[RetrievalCandidate]:
        """Create sample retrieval candidates."""
        return [
            RetrievalCandidate(
                page_id="doc1#p001",
                score=0.8,
                channel_scores={},
                channel_ranks={},
                page_image_uri="s3://doc1/p001.png",
                extracted_text_excerpt="Test excerpt 1",
            ),
            RetrievalCandidate(
                page_id="doc2#p010",
                score=0.75,
                channel_scores={},
                channel_ranks={},
                page_image_uri="s3://doc2/p010.png",
                extracted_text_excerpt="Test excerpt 2",
            ),
        ]

    def test_validate_citations_valid(self, sample_candidates):
        """Valid citations pass through unchanged."""
        answer = "The FSI is 2.5 [1] and parking is 1 per 100 sqm [2]."
        corrected, invalid = validate_citations(answer, sample_candidates)
        assert corrected == answer
        assert invalid == []

    def test_validate_citations_hallucinated(self, sample_candidates):
        """Hallucinated citations are replaced with [?]."""
        answer = "FSI is 2.5 [1] and unknown is [99]."
        corrected, invalid = validate_citations(answer, sample_candidates)
        assert "[?]" in corrected
        assert "[99]" not in corrected
        assert 99 in invalid
        assert 1 not in invalid

    def test_validate_citations_multiple_hallucinated(self, sample_candidates):
        """Multiple hallucinated citations are all replaced."""
        answer = "Facts: [1], [2], [99], [100], [5]."
        corrected, invalid = validate_citations(answer, sample_candidates)
        assert corrected.count("[?]") == 3  # [99], [100], [5] replaced
        assert 99 in invalid
        assert 100 in invalid
        assert 5 in invalid
        assert 1 not in invalid
        assert 2 not in invalid

    def test_validate_citations_empty_answer(self, sample_candidates):
        """Empty answer returns empty with no errors."""
        corrected, invalid = validate_citations("", sample_candidates)
        assert corrected == ""
        assert invalid == []

    def test_validate_citations_no_citations(self, sample_candidates):
        """Answer without citations passes through."""
        answer = "No citations here."
        corrected, invalid = validate_citations(answer, sample_candidates)
        assert corrected == answer
        assert invalid == []

    def test_validate_citations_empty_candidates(self):
        """Empty candidates rejects all citations."""
        answer = "Fact [1] and fact [2]."
        corrected, invalid = validate_citations(answer, [])
        assert "[?]" in corrected
        assert corrected.count("[?]") == 2
        assert 1 in invalid
        assert 2 in invalid


class TestExtractCitationsFromAnswer:
    """Tests for extract_citations_from_answer()."""

    def test_extract_single_citation(self):
        """Single citation is extracted."""
        answer = "The FSI is 2.5 [1]."
        assert extract_citations_from_answer(answer) == [1]

    def test_extract_multiple_citations(self):
        """Multiple citations are extracted in order."""
        answer = "Facts: [1], [2], [3], [5]."
        assert extract_citations_from_answer(answer) == [1, 2, 3, 5]

    def test_extract_citations_gaps(self):
        """Citations with gaps are all extracted."""
        answer = "Facts [1] and [3] and [10]."
        assert extract_citations_from_answer(answer) == [1, 3, 10]

    def test_extract_citations_none(self):
        """No citations returns empty list."""
        assert extract_citations_from_answer("No citations here.") == []

    def test_extract_citations_empty(self):
        """Empty string returns empty list."""
        assert extract_citations_from_answer("") == []


class TestBuildCitationsList:
    """Tests for build_citations_list()."""

    @pytest.fixture
    def sample_candidates(self) -> list[RetrievalCandidate]:
        """Create sample retrieval candidates."""
        return [
            RetrievalCandidate(
                page_id="mumbai_dcr#p001",
                score=0.8,
                channel_scores={},
                channel_ranks={},
                rerank_score=9.5,
                page_image_uri="s3://doc1/p001.png",
                extracted_text_excerpt="Test excerpt 1",
                section_title="Residential Zone Rules",
            ),
            RetrievalCandidate(
                page_id="nbc_2016#p142",
                score=0.75,
                channel_scores={},
                channel_ranks={},
                rerank_score=8.0,
                page_image_uri="s3://doc2/p142.png",
                extracted_text_excerpt="Test excerpt 2",
                section_title="Fire Safety",
            ),
        ]

    def test_build_citations_list_basic(self, sample_candidates):
        """Basic citations list is built correctly."""
        citations = build_citations_list(sample_candidates, [1, 2])
        assert len(citations) == 2
        assert citations[0]["idx"] == 1
        assert citations[0]["page_id"] == "mumbai_dcr#p001"
        assert citations[1]["idx"] == 2
        assert citations[1]["page_id"] == "nbc_2016#p142"

    def test_build_citations_list_page_nums(self, sample_candidates):
        """Page numbers are extracted correctly."""
        citations = build_citations_list(sample_candidates, [1, 2])
        assert citations[0]["page_num"] == 1
        assert citations[1]["page_num"] == 142

    def test_build_citations_list_section_titles(self, sample_candidates):
        """Section titles are included."""
        citations = build_citations_list(sample_candidates, [1, 2])
        assert citations[0]["section_title"] == "Residential Zone Rules"
        assert citations[1]["section_title"] == "Fire Safety"

    def test_build_citations_list_scores(self, sample_candidates):
        """Scores use rerank_score when available."""
        citations = build_citations_list(sample_candidates, [1, 2])
        assert citations[0]["score"] == 9.5  # rerank_score
        assert citations[1]["score"] == 8.0  # rerank_score

    def test_build_citations_list_invalid_index(self, sample_candidates):
        """Invalid indices are skipped."""
        citations = build_citations_list(sample_candidates, [1, 99])
        assert len(citations) == 1
        assert citations[0]["idx"] == 1


class TestBuildSelfCritiquePrompt:
    """Tests for build_self_critique_prompt()."""

    @pytest.fixture
    def sample_candidates(self) -> list[RetrievalCandidate]:
        """Create sample retrieval candidates."""
        return [
            RetrievalCandidate(
                page_id="doc1#p001",
                score=0.8,
                channel_scores={},
                channel_ranks={},
                page_image_uri="s3://doc1/p001.png",
                extracted_text_excerpt="Test excerpt",
            ),
        ]

    def test_self_critique_prompt_for_cited_answer(self, sample_candidates):
        """Self-critique prompt is generated for answers with citations."""
        answer = "FSI is 2.5 [1]."
        prompt = build_self_critique_prompt(answer, sample_candidates)
        assert prompt is not None
        assert "Self-Critique" in prompt or "SELF-CRITIQUE" in prompt
        assert "[1]" in prompt or "page 1" in prompt.lower()

    def test_self_critique_prompt_no_citations(self, sample_candidates):
        """No self-critique prompt for uncited answer."""
        answer = "No citations here."
        prompt = build_self_critique_prompt(answer, sample_candidates)
        assert prompt is None

    def test_self_critique_prompt_empty_answer(self, sample_candidates):
        """No self-critique prompt for empty answer."""
        prompt = build_self_critique_prompt("", sample_candidates)
        assert prompt is None

    def test_self_critique_prompt_empty_candidates(self):
        """No self-critique prompt when no candidates."""
        prompt = build_self_critique_prompt("FSI is 2.5 [1].", [])
        assert prompt is None


class TestAnswerModeEnum:
    """Tests for AnswerMode enum."""

    def test_answer_mode_fast(self):
        """AnswerMode.FAST exists and has correct value."""
        assert AnswerMode.FAST.value == "fast"

    def test_answer_mode_deep(self):
        """AnswerMode.DEEP exists and has correct value."""
        assert AnswerMode.DEEP.value == "deep"

    def test_answer_mode_is_literal(self):
        """AnswerMode values are strings."""
        assert isinstance(AnswerMode.FAST.value, str)
        assert isinstance(AnswerMode.DEEP.value, str)


class TestRefusalReasonEnum:
    """Tests for RefusalReason enum."""

    def test_refusal_reasons_exist(self):
        """All expected refusal reasons exist."""
        assert RefusalReason.OUT_OF_CORPUS.value == "out_of_corpus"
        assert RefusalReason.HOSTILE.value == "hostile"
        assert RefusalReason.NON_PLANNING.value == "non_planning"
        assert RefusalReason.INSUFFICIENT_CONTEXT.value == "insufficient_context"


class TestPageMetadata:
    """Tests for PageMetadata dataclass."""

    def test_page_metadata_to_citation_label(self):
        """to_citation_label formats correctly."""
        pm = PageMetadata(
            page_id="nbc#p142",
            doc_title="NBC 2016 Vol 1",
            page_num=142,
            section_title="Fire Safety",
        )
        assert pm.to_citation_label() == "NBC 2016 Vol 1, p.142"

    def test_page_metadata_to_page_marker(self):
        """to_page_marker formats correctly."""
        pm = PageMetadata(
            page_id="nbc#p142",
            doc_title="NBC 2016 Vol 1",
            page_num=142,
            section_title="Fire Safety",
        )
        marker = pm.to_page_marker()
        assert "[nbc#p142]" in marker
        assert "NBC 2016 Vol 1" in marker
        assert "Section: Fire Safety" in marker
        assert "Page 142" in marker

    def test_page_metadata_defaults(self):
        """Optional fields have defaults."""
        pm = PageMetadata(
            page_id="doc#p1",
            doc_title="Test",
            page_num=1,
        )
        assert pm.section_title is None
        assert pm.image_uri is None


class TestPromptIntegration:
    """Integration tests for the full prompt pipeline."""

    @pytest.fixture
    def full_candidates(self) -> list[RetrievalCandidate]:
        """Create full retrieval candidates for integration tests."""
        return [
            RetrievalCandidate(
                page_id="mumbai_dp_2034#p089",
                score=0.82,
                channel_scores={"visual": 11.0, "text": 0.82},
                channel_ranks={"visual": 1, "text": 2},
                rerank_score=9.2,
                page_image_uri="s3://corpus/mumbai_dp_2034/p089.png",
                extracted_text_excerpt="Table 4.2: FSI values for residential zone — Mumbai Metropolitan Region",
                section_title="4.2 FSI Regulations",
            ),
            RetrievalCandidate(
                page_id="urdpfi_v2#p215",
                score=0.78,
                channel_scores={"visual": 10.5, "text": 0.78},
                channel_ranks={"visual": 2, "text": 1},
                rerank_score=8.7,
                page_image_uri="s3://corpus/urdpfi_v2/p215.png",
                extracted_text_excerpt="FSI for residential use: 2.0 to 3.0 depending on width of road",
                section_title="Chapter 5 — Floor Space Index",
            ),
        ]

    def test_full_pipeline_fast_mode(self, full_candidates):
        """Full pipeline in fast mode produces valid prompt."""
        question = "What is the FSI for residential zones in Mumbai?"
        system, user, version = build_answer_prompt(
            question=question,
            candidates=full_candidates,
            mode=AnswerMode.FAST,
        )

        # System prompt has domain scope
        assert "urban planning" in system.lower()

        # User prompt has question and pages
        assert question in user
        assert "Page 1:" in user
        assert "Page 2:" in user
        assert "mumbai_dp_2034#p089" in user
        assert "urdpfi_v2#p215" in user

        # Citation validation on a generated answer
        generated = "According to the Mumbai DP 2034 [1], FSI is 2.5 for residential."
        validated, invalid = validate_citations(generated, full_candidates)
        assert "[1]" in validated
        assert invalid == []

    def test_full_pipeline_deep_mode(self, full_candidates):
        """Full pipeline in deep mode includes self-critique."""
        system, user, version = build_answer_prompt(
            question="What is FSI for residential?",
            candidates=full_candidates,
            mode=AnswerMode.DEEP,
        )

        # Deep mode has enhanced instructions
        assert "SELF-CRITIQUE" in user
        assert "verify" in user.lower()

        # Self-critique prompt can be built from the answer
        answer_with_citation = "FSI is [1] and [2]."
        critique = build_self_critique_prompt(answer_with_citation, full_candidates)
        assert critique is not None
        assert "Self-Critique" in critique or "SELF-CRITIQUE" in critique

    def test_full_pipeline_citation_roundtrip(self, full_candidates):
        """Citations survive the full build → validate roundtrip."""
        question = "FSI for residential Mumbai?"

        # Build prompt
        _, user, _ = build_answer_prompt(
            question=question,
            candidates=full_candidates,
            mode=AnswerMode.FAST,
        )
        assert "Page 1:" in user
        assert "Page 2:" in user

        # Simulate generation with citation
        generated = "Mumbai residential FSI is 2.5 [1] per DP 2034 and 2.0-3.0 [2] per URDPFI."
        corrected, invalid = validate_citations(generated, full_candidates)
        assert invalid == []
        assert "[1]" in corrected
        assert "[2]" in corrected

        # Extract and build citations list
        cited = extract_citations_from_answer(corrected)
        assert cited == [1, 2]

        citations = build_citations_list(full_candidates, cited)
        assert len(citations) == 2
        assert citations[0]["page_id"] == "mumbai_dp_2034#p089"
        assert citations[1]["page_id"] == "urdpfi_v2#p215"


class TestSmokeTest:
    """Tests for the smoke_test() function."""

    def test_smoke_test_passes(self):
        """smoke_test() runs without errors and returns pass."""
        result = smoke_test()
        assert result["passed"] is True
        assert result["citation_validation_works"] is True
        assert result["refusal_prompt_works"] is True
