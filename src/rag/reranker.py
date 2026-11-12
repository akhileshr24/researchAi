"""Rerank retrieved candidates with a cross-encoder model."""

from functools import lru_cache

from sentence_transformers import CrossEncoder

from src.config.settings import RERANKER_MODEL, TOP_K


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL)


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = TOP_K,
) -> list[dict]:
    """Score query/chunk pairs and return the best chunks."""
    query = query.strip()

    if not query or not candidates or top_k <= 0:
        return []

    pairs = [(query, candidate["text"]) for candidate in candidates]
    scores = get_reranker().predict(pairs)

    ranked = []

    for candidate, score in zip(candidates, scores):
        result = candidate.copy()
        result["rerank_score"] = round(float(score), 6)
        ranked.append(result)

    ranked.sort(
        key=lambda result: result["rerank_score"],
        reverse=True,
    )

    return ranked[:top_k]
