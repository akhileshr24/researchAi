
import re
from rank_bm25 import BM25Okapi

from src.rag.vectorstore import get_all_chunks
from src.config.settings import TOP_K


def tokenize(text):
    """Convert text into lowercase word tokens."""
    return re.findall(r"\b\w+\b", text.lower())


def search_bm25(question, top_k=TOP_K):
    # 1. Get all stored document chunks
    chunks = get_all_chunks()

    if not chunks:
        return []

    # 2. Tokenize every document chunk
    tokenized_corpus = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    # 3. Build BM25 index
    bm25 = BM25Okapi(tokenized_corpus)

    # 4. Tokenize the user's question
    query_tokens = tokenize(question)

    if not query_tokens:
        return []

    # 5. Calculate BM25 relevance scores
    scores = bm25.get_scores(query_tokens)

    # 6. Rank chunks by score
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    # 7. Return ranked chunks with metadata
    results = []

    for index in ranked_indices:
        score = float(scores[index])

        # Ignore chunks with no lexical match
        if score <= 0:
            continue

        chunk = chunks[index].copy()
        chunk["score"] = round(score, 3)
        results.append(chunk)

    return results
