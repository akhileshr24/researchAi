"""Tests for cross-encoder reranking."""

from src.rag import reranker


class FakeReranker:
    def __init__(self, scores):
        self.scores = scores
        self.pairs = []

    def predict(self, pairs):
        self.pairs = pairs
        return self.scores


def test_rerank_sorts_by_cross_encoder_score(monkeypatch, chunk_factory):
    candidates = [
        chunk_factory(text="weak match", chunk_index=0),
        chunk_factory(text="strong match", chunk_index=1),
        chunk_factory(text="negative match", chunk_index=2),
    ]
    fake = FakeReranker([0.2, 0.9, -1.5])
    monkeypatch.setattr(reranker, "get_reranker", lambda: fake)

    results = reranker.rerank("query", candidates, top_k=2)

    assert [result["text"] for result in results] == [
        "strong match",
        "weak match",
    ]
    assert results[0]["rerank_score"] == 0.9
    assert fake.pairs == [
        ("query", "weak match"),
        ("query", "strong match"),
        ("query", "negative match"),
    ]


def test_rerank_returns_empty_for_no_candidates(monkeypatch):
    monkeypatch.setattr(
        reranker,
        "get_reranker",
        lambda: (_ for _ in ()).throw(AssertionError("must not load model")),
    )

    assert reranker.rerank("query", []) == []


def test_rerank_returns_empty_for_empty_query(monkeypatch, chunk_factory):
    monkeypatch.setattr(
        reranker,
        "get_reranker",
        lambda: (_ for _ in ()).throw(AssertionError("must not load model")),
    )

    assert reranker.rerank("  ", [chunk_factory()]) == []
