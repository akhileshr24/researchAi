"""Tests for BM25 keyword retrieval."""

from src.rag import bm25_retriever


def _chunks(chunk_factory):
    return [
        chunk_factory(
            text="Vector search finds semantically similar chunks.",
            chunk_index=0,
        ),
        chunk_factory(
            text="BM25 ranks keyword matches.",
            document_id="doc-b",
            filename="b.pdf",
            chunk_index=1,
        ),
        chunk_factory(
            text="Reranking improves ordering with a cross encoder.",
            document_id="doc-c",
            filename="c.pdf",
            chunk_index=2,
        ),
    ]


def test_bm25_ranks_matching_chunk_first(monkeypatch, chunk_factory):
    monkeypatch.setattr(
        bm25_retriever,
        "get_all_chunks",
        lambda: _chunks(chunk_factory),
    )

    results = bm25_retriever.search_bm25("vector search")

    assert results
    assert "Vector search" in results[0]["text"]
    assert results[0]["score"] > 0


def test_bm25_filters_by_document(monkeypatch, chunk_factory):
    monkeypatch.setattr(
        bm25_retriever,
        "get_all_chunks",
        lambda: _chunks(chunk_factory),
    )

    results = bm25_retriever.search_bm25(
        "keyword matches",
        document_id="doc-b",
    )

    assert results
    assert all(result["document_id"] == "doc-b" for result in results)


def test_bm25_respects_top_k(monkeypatch, chunk_factory):
    monkeypatch.setattr(
        bm25_retriever,
        "get_all_chunks",
        lambda: _chunks(chunk_factory),
    )

    assert len(bm25_retriever.search_bm25("chunks", top_k=1)) == 1


def test_bm25_returns_empty_without_tokens(monkeypatch):
    monkeypatch.setattr(
        bm25_retriever,
        "get_all_chunks",
        lambda: (_ for _ in ()).throw(AssertionError("must not load chunks")),
    )

    assert bm25_retriever.search_bm25("!!!") == []


def test_bm25_returns_empty_when_nothing_matches(monkeypatch, chunk_factory):
    monkeypatch.setattr(
        bm25_retriever,
        "get_all_chunks",
        lambda: _chunks(chunk_factory),
    )

    assert bm25_retriever.search_bm25("zebra") == []
