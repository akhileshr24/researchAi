"""Combine vector and BM25 rankings with Reciprocal Rank Fusion."""

from src.config.settings import RERANK_CANDIDATES, RRF_K, TOP_K
from src.rag.bm25_retriever import search_bm25
from src.rag.vectorstore import search


def search_hybrid(
    query: str,
    top_k: int = TOP_K,
    candidate_k: int | None = None,
    rrf_k: int = RRF_K,
    document_id: str | None = None,
) -> list[dict]:
    """Fuse vector and BM25 results by summing ``1 / (rrf_k + rank)``."""
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
        ("vector", vector_results),
        ("bm25", bm25_results),
    ]

    for retriever, results in rankings:
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

            entry["rrf_score"] += 1 / (rrf_k + rank)
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
