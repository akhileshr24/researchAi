"""Combine vector and BM25 rankings with Reciprocal Rank Fusion."""

import math

from src.config.settings import RERANK_CANDIDATES, RRF_K, TOP_K
from src.rag.bm25_retriever import search_bm25
from src.rag.vectorstore import search


def search_hybrid(
    query: str,
    top_k: int = TOP_K,
    candidate_k: int | None = None,
    rrf_k: int = RRF_K,
    document_id: str | None = None,
    vector_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> list[dict]:
    """Fuse vector and BM25 results with weighted Reciprocal Rank Fusion.

    Each branch contributes ``weight / (rrf_k + rank)`` with 1-based
    ranks; a chunk missing from one branch contributes zero for that
    branch. Defaults (``rrf_k=settings.RRF_K``, both weights 1.0) keep
    the original unweighted behaviour.
    """
    if rrf_k <= 0:
        raise ValueError(f"rrf_k must be positive, got {rrf_k}")
    if not math.isfinite(vector_weight) or vector_weight <= 0:
        raise ValueError(
            f"vector_weight must be a finite positive number, "
            f"got {vector_weight!r}"
        )
    if not math.isfinite(bm25_weight) or bm25_weight <= 0:
        raise ValueError(
            f"bm25_weight must be a finite positive number, "
            f"got {bm25_weight!r}"
        )

    if candidate_k is None:
        candidate_k = RERANK_CANDIDATES

    # Keep all vector candidates for fusion; the cross-encoder decides
    # relevance after RRF instead of dropping weak-but-relevant chunks here.
    vector_results = search(
        query,
        top_k=candidate_k,
        document_id=document_id,
        min_score=0.0,
    )
    bm25_results = search_bm25(
        query,
        top_k=candidate_k,
        document_id=document_id,
    )

    fused = {}
    rankings = [
        ("vector", vector_results, vector_weight),
        ("bm25", bm25_results, bm25_weight),
    ]

    for retriever, results, weight in rankings:
        for rank, chunk in enumerate(results, start=1):
            key = (chunk["document_id"], chunk["chunk_index"])
            entry = fused.setdefault(key, {
                "text": chunk["text"],
                "document_id": chunk["document_id"],
                "filename": chunk["filename"],
                "page": chunk["page"],
                "chunk_index": chunk["chunk_index"],
                "start_index": chunk["start_index"],
                "rrf_score": 0.0,
                "retrievers": set(),
                "retriever_scores": {},
            })

            entry["rrf_score"] += weight / (rrf_k + rank)
            entry["retrievers"].add(retriever)
            entry["retriever_scores"][retriever] = chunk["score"]

    ranked = sorted(
        fused.values(),
        key=lambda entry: entry["rrf_score"],
        reverse=True,
    )

    for entry in ranked:
        entry["rrf_score"] = round(entry["rrf_score"], 6)
        entry["retrievers"] = sorted(entry["retrievers"])

    return ranked[:top_k]
