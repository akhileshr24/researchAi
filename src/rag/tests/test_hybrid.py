"""Tests for Reciprocal Rank Fusion."""

import pytest

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


def _weighted_fixture(monkeypatch, chunk_factory):
    """Shared/vector-only/bm25-only chunk fixture with stubbed branches."""
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


def test_rrf_defaults_match_explicit_unweighted_parameters(monkeypatch, chunk_factory):
    _weighted_fixture(monkeypatch, chunk_factory)

    default = hybrid_retriever.search_hybrid("query")
    explicit = hybrid_retriever.search_hybrid(
        "query",
        rrf_k=hybrid_retriever.RRF_K,
        vector_weight=1.0,
        bm25_weight=1.0,
    )

    assert [r["rrf_score"] for r in default] == [
        r["rrf_score"] for r in explicit
    ]
    assert default[0]["rrf_score"] == round(2 / 61, 6)


def test_rrf_explicit_k_changes_fusion(monkeypatch, chunk_factory):
    _weighted_fixture(monkeypatch, chunk_factory)

    results = hybrid_retriever.search_hybrid("query", rrf_k=10)

    # shared chunk: rank 1 in both branches -> 2 / (10 + 1)
    assert results[0]["text"] == "shared chunk"
    assert results[0]["rrf_score"] == round(2 / 11, 6)


def test_rrf_vector_weight_changes_fusion(monkeypatch, chunk_factory):
    _weighted_fixture(monkeypatch, chunk_factory)

    results = hybrid_retriever.search_hybrid("query", vector_weight=2.0)

    # shared: 2/(60+1) + 1/(60+1); vector-only at rank 2: 2/(60+2)
    assert results[0]["text"] == "shared chunk"
    assert results[0]["rrf_score"] == round(3 / 61, 6)
    assert results[1]["text"] == "vector only"
    assert results[1]["rrf_score"] == round(2 / 62, 6)
    assert results[2]["text"] == "bm25 only"
    assert results[2]["rrf_score"] == round(1 / 62, 6)


def test_rrf_bm25_weight_changes_fusion(monkeypatch, chunk_factory):
    _weighted_fixture(monkeypatch, chunk_factory)

    results = hybrid_retriever.search_hybrid("query", bm25_weight=2.0)

    # shared: 1/(60+1) + 2/(60+1); bm25-only at rank 2: 2/(60+2)
    assert results[0]["text"] == "shared chunk"
    assert results[0]["rrf_score"] == round(3 / 61, 6)
    assert results[1]["text"] == "bm25 only"
    assert results[1]["rrf_score"] == round(2 / 62, 6)


def test_rrf_missing_branch_contributes_zero(monkeypatch, chunk_factory):
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

    results = hybrid_retriever.search_hybrid("query", vector_weight=2.0)

    assert len(results) == 1
    assert results[0]["rrf_score"] == round(2 / 61, 6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rrf_k": 0},
        {"rrf_k": -5},
        {"vector_weight": 0.0},
        {"vector_weight": -1.0},
        {"bm25_weight": 0.0},
        {"bm25_weight": -2.5},
    ],
)
def test_rrf_rejects_invalid_k_and_weights(chunk_factory, monkeypatch, kwargs):
    _weighted_fixture(monkeypatch, chunk_factory)

    with pytest.raises(ValueError):
        hybrid_retriever.search_hybrid("query", **kwargs)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("weight_name", ["vector_weight", "bm25_weight"])
def test_rrf_rejects_non_finite_weights(
    chunk_factory,
    monkeypatch,
    weight_name,
    bad_value,
):
    _weighted_fixture(monkeypatch, chunk_factory)

    with pytest.raises(ValueError, match=f"{weight_name} must be a finite"):
        hybrid_retriever.search_hybrid("query", **{weight_name: bad_value})
