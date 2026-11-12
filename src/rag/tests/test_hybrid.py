"""Tests for Reciprocal Rank Fusion."""

from src.rag import hybrid_retriever


def test_rrf_promotes_chunks_found_by_both_retrievers(
    monkeypatch,
    chunk_factory,
):
    shared = chunk_factory(text="shared chunk", chunk_index=0)
    vector_only = chunk_factory(text="vector only", chunk_index=1)
    bm25_only = chunk_factory(
        text="bm25 only",
        document_id="doc-b",
        filename="b.pdf",
        chunk_index=0,
    )

    monkeypatch.setattr(
        hybrid_retriever,
        "search",
        lambda *args, **kwargs: [
            {**shared, "score": 0.9},
            {**vector_only, "score": 0.8},
        ],
    )
    monkeypatch.setattr(
        hybrid_retriever,
        "search_bm25",
        lambda *args, **kwargs: [
            {**shared, "score": 5.0},
            {**bm25_only, "score": 3.0},
        ],
    )

    results = hybrid_retriever.search_hybrid("query")

    assert len(results) == 3
    assert results[0]["text"] == "shared chunk"
    assert results[0]["retrievers"] == ["bm25", "vector"]
    assert results[0]["rrf_score"] == round(2 / 61, 6)
    assert results[0]["retriever_scores"] == {"vector": 0.9, "bm25": 5.0}


def test_rrf_respects_top_k(monkeypatch, chunk_factory):
    chunk = chunk_factory()
    monkeypatch.setattr(
        hybrid_retriever,
        "search",
        lambda *args, **kwargs: [{**chunk, "score": 0.9}],
    )
    monkeypatch.setattr(
        hybrid_retriever,
        "search_bm25",
        lambda *args, **kwargs: [],
    )

    assert len(hybrid_retriever.search_hybrid("query", top_k=1)) == 1


def test_rrf_returns_empty_when_both_retrievers_are_empty(monkeypatch):
    monkeypatch.setattr(
        hybrid_retriever,
        "search",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        hybrid_retriever,
        "search_bm25",
        lambda *args, **kwargs: [],
    )

    assert hybrid_retriever.search_hybrid("query") == []


def test_rrf_passes_document_filter_and_candidate_limit(
    monkeypatch,
    chunk_factory,
):
    calls = {}

    def fake_search(query, top_k, document_id=None, min_score=None):
        calls["vector"] = (query, top_k, document_id, min_score)
        return []

    def fake_bm25(query, top_k, document_id=None):
        calls["bm25"] = (query, top_k, document_id)
        return []

    monkeypatch.setattr(hybrid_retriever, "search", fake_search)
    monkeypatch.setattr(hybrid_retriever, "search_bm25", fake_bm25)

    hybrid_retriever.search_hybrid(
        "query",
        candidate_k=7,
        document_id="doc-a",
    )

    assert calls["vector"] == ("query", 7, "doc-a", 0.0)
    assert calls["bm25"] == ("query", 7, "doc-a")


def test_hybrid_disables_vector_relevance_cutoff(monkeypatch):
    calls = {}

    def fake_search(query, top_k, document_id=None, min_score=None):
        calls["min_score"] = min_score
        return []

    monkeypatch.setattr(hybrid_retriever, "search", fake_search)
    monkeypatch.setattr(
        hybrid_retriever,
        "search_bm25",
        lambda *args, **kwargs: [],
    )

    hybrid_retriever.search_hybrid("query")

    assert calls["min_score"] == 0.0
