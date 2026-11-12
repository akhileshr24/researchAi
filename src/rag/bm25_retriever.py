"""Keyword retrieval with BM25."""

import re

from rank_bm25 import BM25L

from src.config.settings import TOP_K
from src.rag.vectorstore import get_all_chunks

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens."""
    return TOKEN_PATTERN.findall(text.lower())


def search_bm25(
    query: str,
    top_k: int = TOP_K,
    document_id: str | None = None,
) -> list[dict]:
    """Rank stored chunks by BM25 score."""
    query_tokens = tokenize(query)

    if not query_tokens or top_k <= 0:
        return []

    chunks = get_all_chunks()

    if document_id:
        chunks = [
            chunk
            for chunk in chunks
            if chunk["document_id"] == document_id
        ]

    if not chunks:
        return []

    bm25 = BM25L([tokenize(chunk["text"]) for chunk in chunks])
    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )

    results = []

    for index in ranked_indices:
        if len(results) >= top_k:
            break

        score = float(scores[index])

        if score <= 0:
            break

        chunk = chunks[index].copy()
        chunk["score"] = round(score, 4)
        results.append(chunk)

    return results
