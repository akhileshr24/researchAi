
from src.rag.vectorstore import search
from src.rag.bm25_retriever import search_bm25
from src.config.settings import TOP_K


def search_hybrid(question, top_k=TOP_K, rrf_k=60):

    # 1. Retrieve chunks from both search systems
    vector_results = search(question, top_k=top_k)
    bm25_results = search_bm25(question, top_k=top_k)

    # 2. Store combined results
    fused_results = {}

    # 3. Combine both rankings using RRF
    rankings = [
        ("vector", vector_results),
        ("bm25", bm25_results)
    ]

    for search_type, results in rankings:
        for rank, chunk in enumerate(results, start=1):

            # Identify each unique chunk
            key = (
                chunk["filename"],
                chunk["page"],
                chunk["chunk_index"]
            )

            if key not in fused_results:
                fused_results[key] = {
                    "text": chunk["text"],
                    "page": chunk["page"],
                    "filename": chunk["filename"],
                    "chunk_index": chunk["chunk_index"],
                    "start_index": chunk["start_index"],
                    "rrf_score": 0.0
                }

            # Add reciprocal rank score
            fused_results[key]["rrf_score"] += (
                1 / (rrf_k + rank)
            )

    # 4. Sort by combined RRF score
    ranked_results = sorted(
        fused_results.values(),
        key=lambda item: item["rrf_score"],
        reverse=True
    )

    # 5. Return final top K chunks
    return ranked_results[:top_k]
