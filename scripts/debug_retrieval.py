"""Debug the hybrid retrieval + cross-encoder pipeline for one document.

Runs the two retrieval stages against the stored ChromaDB corpus and prints
the final ranked context, so retrieval quality can be inspected chunk by
chunk without going through the QA/LLM layer.

Usage (from the repository root)::

    uv run python scripts/debug_retrieval.py <document_id>

    uv run python scripts/debug_retrieval.py 4fd44b31a3a08548 \
        --question "Explain the four rules of deep work"

``document_id`` is required; ``--question`` defaults to the deep-work
question used during initial development.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running the script directly from ``scripts/``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.hybrid_retriever import search_hybrid
from src.rag.reranker import rerank
from src.config.settings import RERANK_CANDIDATES, MIN_RERANK_SCORE

DEFAULT_QUESTION = (
    "Explain the four rules of deep work and how each rule "
    "contributes to building a sustainable deep work habit."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect hybrid retrieval and cross-encoder reranking "
            "for one document."
        ),
    )
    parser.add_argument(
        "document_id",
        help="ChromaDB document ID to restrict retrieval to (required).",
    )
    parser.add_argument(
        "--question",
        default=DEFAULT_QUESTION,
        help="Question to retrieve for (default: built-in deep work question).",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    question = args.question

    # Stage 1: Hybrid retrieval
    candidates = search_hybrid(
        question,
        top_k=RERANK_CANDIDATES,
        document_id=args.document_id,
    )

    print(f"\nHybrid candidates: {len(candidates)}")

    # Stage 2: Cross-Encoder reranking
    ranked = rerank(question, candidates, top_k=20)

    print(f"Reranked candidates: {len(ranked)}")
    print(f"Minimum rerank score: {MIN_RERANK_SCORE}")

    # Stage 3: Inspect final context
    for index, result in enumerate(ranked, start=1):
        print("\n" + "=" * 80)
        print(f"RANK: {index}")
        print(f"PAGE: {result['page']}")
        print(f"CHUNK: {result['chunk_index']}")
        print(f"RRF SCORE: {result['rrf_score']}")
        print(f"RERANK SCORE: {result['rerank_score']}")
        print(f"RETRIEVERS: {result['retrievers']}")
        print(f"ACCEPTED: {result['rerank_score'] >= MIN_RERANK_SCORE}")
        print("\nTEXT:\n")
        print(result["text"])


if __name__ == "__main__":
    main()
