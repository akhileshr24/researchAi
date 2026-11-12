"""Compare retrieval methods for the Hybrid RAG pipeline.

Run with::

    uv run python -m src.rag.evaluation.evaluate

The evaluation is deliberately independent of the LLM: it measures whether the
labelled chunks appear in each method's top results. Results are printed as an
aggregate table with per-question diagnostics and saved as JSON under
``reports/rag_evaluation/``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean

from src.config.settings import RERANK_CANDIDATES
from src.rag.bm25_retriever import search_bm25
from src.rag.evaluation.dataset import (
    LABELS_VERIFIED,
    LABEL_STATUS,
    EvaluationQuestion,
    get_questions,
    validate_dataset,
)
from src.rag.evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.rag.hybrid_retriever import search_hybrid
from src.rag.reranker import rerank
from src.rag.vectorstore import get_all_chunks, search

EVAL_TOP_K = 5
VECTOR_MIN_SCORE = 0.0
CANDIDATE_LIMIT = RERANK_CANDIDATES
REPORT_DIR = Path("reports/rag_evaluation")
REPORT_PATH = REPORT_DIR / "evaluation_results.json"
METRIC_KEYS = ("precision@5", "recall@5", "mrr", "ndcg@5")

METHODS: tuple[tuple[str, str], ...] = (
    ("Vector", "vector"),
    ("BM25", "bm25"),
    ("Hybrid RRF", "hybrid"),
    ("Hybrid + Cross-Encoder", "hybrid_rerank"),
)


@dataclass
class RetrievedItem:
    """One chunk returned by a retrieval method."""

    rank: int
    document_id: str
    chunk_index: int
    filename: str
    page: int
    grade: int
    relevant: bool
    score: float | None
    text_preview: str


@dataclass
class MethodResult:
    """Metrics and retrieved chunks for one method on one question."""

    method: str
    retrieved: list[RetrievedItem]
    metrics: dict[str, float]


@dataclass
class QuestionResult:
    """All method results for a single evaluation question."""

    question_id: str
    question: str
    relevant_chunks: list[dict]
    methods: dict[str, MethodResult]


def _chunk_key(chunk: dict) -> tuple[str, int]:
    return (chunk["document_id"], chunk["chunk_index"])


def _score(chunk: dict) -> float | None:
    """Return whichever relevance score the retrieval method produced."""
    for key in ("rerank_score", "rrf_score", "score"):
        if key in chunk:
            return float(chunk[key])
    return None


def retrieve(method: str, question: str) -> list[dict]:
    """Run one retrieval method and return its final top-k chunks."""
    if method == "vector":
        return search(
            question,
            top_k=EVAL_TOP_K,
            min_score=VECTOR_MIN_SCORE,
        )

    if method == "bm25":
        return search_bm25(question, top_k=EVAL_TOP_K)

    if method == "hybrid":
        return search_hybrid(
            question,
            top_k=EVAL_TOP_K,
            candidate_k=CANDIDATE_LIMIT,
        )

    if method == "hybrid_rerank":
        candidates = search_hybrid(
            question,
            top_k=CANDIDATE_LIMIT,
            candidate_k=CANDIDATE_LIMIT,
        )
        return rerank(question, candidates, top_k=EVAL_TOP_K)

    raise ValueError(f"Unknown retrieval method: {method}")


def evaluate_question(question: EvaluationQuestion) -> QuestionResult:
    """Evaluate every retrieval method independently for one question."""
    relevance = question.relevance_grades()
    relevant_keys = set(relevance)
    method_results: dict[str, MethodResult] = {}

    for method_name, method_id in METHODS:
        chunks = retrieve(method_id, question.question)
        retrieved_keys = [_chunk_key(chunk) for chunk in chunks]

        retrieved = [
            RetrievedItem(
                rank=rank,
                document_id=chunk["document_id"],
                chunk_index=chunk["chunk_index"],
                filename=chunk["filename"],
                page=chunk["page"],
                grade=relevance.get(_chunk_key(chunk), 0),
                relevant=_chunk_key(chunk) in relevant_keys,
                score=_score(chunk),
                text_preview=" ".join(chunk["text"].split())[:280],
            )
            for rank, chunk in enumerate(chunks, start=1)
        ]

        method_results[method_name] = MethodResult(
            method=method_name,
            retrieved=retrieved,
            metrics={
                "precision@5": precision_at_k(
                    retrieved_keys,
                    relevant_keys,
                    EVAL_TOP_K,
                ),
                "recall@5": recall_at_k(
                    retrieved_keys,
                    relevant_keys,
                    EVAL_TOP_K,
                ),
                "mrr": reciprocal_rank(retrieved_keys, relevant_keys),
                "ndcg@5": ndcg_at_k(
                    retrieved_keys,
                    relevance,
                    EVAL_TOP_K,
                ),
            },
        )

    return QuestionResult(
        question_id=question.question_id,
        question=question.question,
        relevant_chunks=[
            {
                "document_id": label.document_id,
                "chunk_index": label.chunk_index,
                "grade": label.grade,
            }
            for label in question.relevant_chunks
        ],
        methods=method_results,
    )


def aggregate_metrics(
    results: list[QuestionResult],
) -> dict[str, dict[str, float]]:
    """Return the mean of each metric, grouped by retrieval method."""
    if not results:
        raise ValueError("Cannot aggregate an empty result set")

    return {
        method_name: {
            metric: fmean(
                result.methods[method_name].metrics[metric]
                for result in results
            )
            for metric in METRIC_KEYS
        }
        for method_name, _ in METHODS
    }


def build_report(
    results: list[QuestionResult],
    aggregate: dict[str, dict[str, float]],
) -> dict:
    """Build the JSON-serializable evaluation report."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "labels_verified": LABELS_VERIFIED,
        "label_status": LABEL_STATUS,
        "configuration": {
            "top_k": EVAL_TOP_K,
            "candidate_limit": CANDIDATE_LIMIT,
            "vector_min_score": VECTOR_MIN_SCORE,
            "methods": [name for name, _ in METHODS],
        },
        "aggregate": aggregate,
        "questions": [
            {
                "question_id": result.question_id,
                "question": result.question,
                "relevant_chunks": result.relevant_chunks,
                "methods": {
                    method_name: {
                        "metrics": result.methods[method_name].metrics,
                        "retrieved": [
                            {
                                "rank": item.rank,
                                "document_id": item.document_id,
                                "chunk_index": item.chunk_index,
                                "filename": item.filename,
                                "page": item.page,
                                "grade": item.grade,
                                "relevant": item.relevant,
                                "score": item.score,
                                "text_preview": item.text_preview,
                            }
                            for item in result.methods[method_name].retrieved
                        ],
                    }
                    for method_name, _ in METHODS
                },
            }
            for result in results
        ],
    }


def print_summary(aggregate: dict[str, dict[str, float]]) -> None:
    """Print the aggregate comparison table."""
    print("\nRetrieval comparison")
    print(
        "| Method                | Precision@5 | Recall@5 | MRR    | nDCG@5 |"
    )
    print(
        "|-----------------------|-------------|----------|--------|--------|"
    )

    for method_name, _ in METHODS:
        metrics = aggregate[method_name]
        print(
            f"| {method_name:<21} "
            f"| {metrics['precision@5']:>11.4f} "
            f"| {metrics['recall@5']:>8.4f} "
            f"| {metrics['mrr']:>6.4f} "
            f"| {metrics['ndcg@5']:>6.4f} |"
        )


def print_question_results(results: list[QuestionResult]) -> None:
    """Print per-question metrics and retrieved chunk identifiers."""
    print("\nPer-question results (* marks a labelled relevant chunk)")

    for result in results:
        print(f"\n[{result.question_id}] {result.question}")
        print(
            "  ground truth: "
            + ", ".join(
                f"{label['document_id']}#{label['chunk_index']}"
                f"(grade {label['grade']})"
                for label in result.relevant_chunks
            )
        )

        for method_name, _ in METHODS:
            method = result.methods[method_name]
            metrics = method.metrics
            print(
                f"  {method_name:<22} "
                f"P@5={metrics['precision@5']:.3f} "
                f"R@5={metrics['recall@5']:.3f} "
                f"MRR={metrics['mrr']:.3f} "
                f"nDCG@5={metrics['ndcg@5']:.3f}"
            )

            if not method.retrieved:
                print("      (no results)")
                continue

            print(
                "      "
                + ", ".join(
                    f"{item.rank}. {item.filename} p{item.page} "
                    f"chunk {item.chunk_index}"
                    f"{'*' if item.relevant else ''}"
                    for item in method.retrieved
                )
            )


def _stored_chunk_keys() -> set[tuple[str, int]]:
    return {_chunk_key(chunk) for chunk in get_all_chunks()}


def main() -> None:
    """Run the full evaluation, print diagnostics and save the JSON report."""
    questions = get_questions()
    validate_dataset(questions, _stored_chunk_keys())

    results = [evaluate_question(question) for question in questions]
    aggregate = aggregate_metrics(results)

    print_summary(aggregate)
    print_question_results(results)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(
            build_report(results, aggregate),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\nJSON report: {REPORT_PATH}")

    if not LABELS_VERIFIED:
        print(
            "NOTE: relevance labels are provisional and require manual "
            "verification before metrics are treated as final."
        )


if __name__ == "__main__":
    main()
