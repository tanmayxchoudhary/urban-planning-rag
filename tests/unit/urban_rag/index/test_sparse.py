"""Unit tests for src/urban_rag/index/sparse.py — BM25 sparse indexer."""

from __future__ import annotations

import pytest

from urban_rag.index.sparse import (
    BM25_K1,
    BM25_B,
    BM25Scorer,
    _tokenize,
    _compute_term_frequencies,
)


class TestBM25Constants:
    """Tests for BM25 parameter constants."""

    def test_bm25_k1_value(self) -> None:
        """Test BM25_K1 is 1.5 (standard IR value)."""
        assert BM25_K1 == 1.5

    def test_bm25_b_value(self) -> None:
        """Test BM25_B is 0.75 (standard IR value)."""
        assert BM25_B == 0.75


class TestTokenize:
    """Tests for _tokenize helper."""

    def test_lowercase(self) -> None:
        """Text is converted to lowercase."""
        tokens = _tokenize("HELLO World")
        assert "hello" in tokens
        assert "world" in tokens
        assert "HELLO" not in tokens

    def test_split_on_non_alphanumeric(self) -> None:
        """Text is split on non-alphanumeric characters."""
        tokens = _tokenize("hello-world foo.bar")
        assert "hello" in tokens
        assert "world" in tokens
        assert "foo" in tokens
        assert "bar" in tokens

    def test_filter_short_tokens(self) -> None:
        """Tokens with length < 2 are filtered out."""
        tokens = _tokenize("a b c dd ee ff")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "c" not in tokens
        assert "dd" in tokens
        assert "ee" in tokens
        assert "ff" in tokens

    def test_empty_result_for_special_chars(self) -> None:
        """Special character strings produce empty token list."""
        tokens = _tokenize("--- ...")
        assert tokens == []

    def test_legal_document_tokenization(self) -> None:
        """Legal document text is tokenized correctly."""
        text = "SCHEDULE II — Floor Space Index"
        tokens = _tokenize(text)
        assert "schedule" in tokens
        assert "ii" in tokens
        assert "floor" in tokens
        assert "space" in tokens
        assert "index" in tokens


class TestComputeTermFrequencies:
    """Tests for _compute_term_frequencies helper."""

    def test_single_document(self) -> None:
        """Test term frequencies for a single document."""
        texts = ["hello world hello"]
        tfs = _compute_term_frequencies(texts)
        assert len(tfs) == 1
        assert tfs[0]["hello"] == 2
        assert tfs[0]["world"] == 1

    def test_multiple_documents(self) -> None:
        """Test term frequencies for multiple documents."""
        texts = ["cat dog", "cat mouse", "dog mouse"]
        tfs = _compute_term_frequencies(texts)
        assert len(tfs) == 3
        assert tfs[0]["cat"] == 1
        assert tfs[0]["dog"] == 1
        assert tfs[1]["cat"] == 1
        assert tfs[1]["mouse"] == 1
        assert tfs[2]["dog"] == 1
        assert tfs[2]["mouse"] == 1

    def test_empty_texts(self) -> None:
        """Test with empty texts list."""
        tfs = _compute_term_frequencies([])
        assert tfs == []


class TestBM25ScorerInit:
    """Tests for BM25Scorer initialization."""

    def test_init_single_document(self) -> None:
        """Test BM25Scorer with a single document."""
        texts = ["hello world hello"]
        scorer = BM25Scorer(texts)
        assert scorer.N == 1
        assert len(scorer._vocab) > 0
        assert scorer._avgdl > 0

    def test_init_multiple_documents(self) -> None:
        """Test BM25Scorer with multiple documents."""
        texts = ["cat dog", "cat mouse", "dog mouse"]
        scorer = BM25Scorer(texts)
        assert scorer.N == 3
        assert len(scorer._vocab) == 3  # cat, dog, mouse (3 unique terms)
        assert scorer._avgdl > 0

    def test_idf_computed(self) -> None:
        """Test that IDF values are computed for all terms."""
        texts = ["the quick brown fox", "the slow green turtle"]
        scorer = BM25Scorer(texts)
        for term in scorer._vocab:
            assert term in scorer._idf


class TestBM25ScorerComputeSparseVector:
    """Tests for BM25Scorer.compute_sparse_vector method."""

    def test_compute_sparse_vector(self) -> None:
        """Test sparse vector computation for a document."""
        texts = [" FSI residential zones ", " FAR commercial areas "]
        scorer = BM25Scorer(texts)
        indices, values = scorer.compute_sparse_vector("FSI residential")
        assert len(indices) > 0
        assert len(values) == len(indices)
        assert all(v >= 0 for v in values)

    def test_sparse_vector_top_k(self) -> None:
        """Test that top_k limits the number of terms."""
        texts = [" ".join(["term"] * 100) for _ in range(10)]
        scorer = BM25Scorer(texts)
        indices, values = scorer.compute_sparse_vector(" ".join(["term"] * 50), top_k=5)
        assert len(indices) <= 5
        assert len(values) <= 5

    def test_empty_document(self) -> None:
        """Test sparse vector for empty document returns empty."""
        texts = ["hello world"]
        scorer = BM25Scorer(texts)
        # An empty query returns empty sparse vector
        indices, values = scorer.compute_sparse_vector("")
        # Tokenize of empty string is empty
        assert len(indices) == 0


class TestBM25ScorerScoreQuery:
    """Tests for BM25Scorer.score_query method."""

    def test_score_query(self) -> None:
        """Test query scoring."""
        texts = ["FSI residential Mumbai", "FAR commercial Delhi", "FSI FAR zones"]
        scorer = BM25Scorer(texts)
        scores = scorer.score_query("FSI")
        assert len(scores) == 3
        assert all(s >= 0 for s in scores.values())

    def test_score_query_no_results(self) -> None:
        """Test query with no matching terms returns empty."""
        texts = ["FSI residential", "FAR commercial"]
        scorer = BM25Scorer(texts)
        scores = scorer.score_query("xyzabc123")
        # No terms match, so either empty or all zeros
        assert len(scores) == 0 or all(s == 0 for s in scores.values())

    def test_scores_sorted_by_relevance(self) -> None:
        """Test that matching documents get higher scores."""
        texts = [
            "FSI residential zone rules",
            "FAR commercial building requirements",
            "FSI FAR floor space index",
        ]
        scorer = BM25Scorer(texts)
        scores = scorer.score_query("FSI")
        # Document 0 and 2 contain FSI
        assert scores[0] > 0
        assert scores[2] > 0
        # Document 1 doesn't contain FSI
        assert scores[1] == 0

    def test_empty_query(self) -> None:
        """Test empty query returns empty dict."""
        texts = ["FSI residential"]
        scorer = BM25Scorer(texts)
        scores = scorer.score_query("")
        assert scores == {}


class TestSparseIndexIntegration:
    """Integration-style tests for sparse indexing (without Qdrant)."""

    def test_bm25_scores_distinguish_documents(self) -> None:
        """Test that BM25 correctly distinguishes relevant from non-relevant docs."""
        texts = [
            "Mumbai Development Plan 2034 residential FSI",
            "Delhi Master Plan 2021 commercial FAR",
            "Bangalore Zoning Regulations FSI FAR",
        ]
        scorer = BM25Scorer(texts)

        # Query for Mumbai-specific content
        mumbai_scores = scorer.score_query("Mumbai")
        # Query for FSI-related content
        fsi_scores = scorer.score_query("FSI")

        # Mumbai doc should score higher for Mumbai query
        assert mumbai_scores[0] > mumbai_scores[1]
        assert mumbai_scores[0] > mumbai_scores[2]

        # At least one doc should score for FSI (the first or third)
        # Note: "commercial" doesn't contain FSI
        assert fsi_scores[0] > 0 or fsi_scores[2] > 0
