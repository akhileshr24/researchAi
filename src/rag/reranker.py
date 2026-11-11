
from functools import lru_cache

from sentence_transformers import CrossEncoder

from src.rag.hybrid_retriever import search_hybrid
from src.config.settings import TOP_K


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def get_reranker():
    return CrossEncoder(RERANKER_MODEL)


def rerank_hybrid(question, top_k=TOP_K, candidate_k=10):

    # 1. Retrieve candidate chunks using Hybrid Search
    candidates = search_hybrid(
        question,
        top_k=candidate_k
    )

    if not candidates:
        return []

    # 2. Create query-document pairs
    pairs = [
        (question, chunk["text"])
        for chunk in candidates
    ]

    # 3. Calculate relevance scores
    model = get_reranker()
    scores = model.predict(pairs)

    # 4. Attach reranking scores
    ranked_results = []

    for chunk, score in zip(candidates, scores):
        result = chunk.copy()
        result["rerank_score"] = float(score)
        ranked_results.append(result)

    # 5. Sort by reranking score
    ranked_results.sort(
        key=lambda item: item["rerank_score"],
        reverse=True
    )

    return ranked_results[:top_k]
