"""Compare retrieval methods for the Hybrid RAG pipeline.

Run with::

    uv run python -m src.rag.evaluation.evaluate

The evaluation is deliberately independent of the LLM: it measures whether the
labelled chunks appear in each method's top results. Results are printed as an
aggregate table with per-question diagnostics, and one run generates all three
reports under ``reports/rag_evaluation/``: the detailed JSON results, a JSON
summary and a Markdown report.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
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
from src.rag.evaluation.report import write_markdown_report
from src.rag.hybrid_retriever import search_hybrid
from src.rag.reranker import rerank
from src.rag.vectorstore import get_all_chunks, search

EVAL_TOP_K = 5
VECTOR_MIN_SCORE = 0.0
CANDIDATE_LIMIT = RERANK_CANDIDATES
REPORT_DIR = Path("reports/rag_evaluation")
REPORT_PATH = REPORT_DIR / "evaluation_results.json"
SUMMARY_PATH = REPORT_DIR / "evaluation_summary.json"
MARKDOWN_PATH = REPORT_DIR / "latest.md"
LOCK_FILE_PATH = Path("uv.lock")
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
    latency_ms: float = 0.0


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


# REPRODUCIBILITY METADATA


def _git_commit_sha() -> str:
    """Return the current Git commit SHA, or "" when Git is unavailable."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""

    if completed.returncode != 0:
        return ""

    return completed.stdout.strip()


def _python_version() -> str:
    """Return the running Python version, e.g. ``3.13.16``."""
    return sys.version.split()[0]


def _sha256_of_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)

    return digest.hexdigest()


def _lock_file_hash() -> str:
    """Return the SHA-256 of ``uv.lock``, or "" when it does not exist."""
    if not LOCK_FILE_PATH.is_file():
        return ""

    return _sha256_of_file(LOCK_FILE_PATH)


def _dataset_hash() -> str:
    """Return a stable SHA-256 of the canonical dataset representation."""
    payload = json.dumps(
        [asdict(question) for question in get_questions()],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def reproducibility_metadata() -> dict[str, str]:
    """Return the reproducibility block shared by all generated reports."""
    return {
        "git_commit_sha": _git_commit_sha(),
        "python_version": _python_version(),
        "lock_file_sha256": _lock_file_hash(),
        "dataset_sha256": _dataset_hash(),
    }


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
        started = time.perf_counter()
        chunks = retrieve(method_id, question.question)
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
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
            latency_ms=max(latency_ms, 0.0),
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
    """Return the mean of each metric and of retrieval latency, by method.

    ``mean_latency_ms`` is the arithmetic mean of the per-question
    ``latency_ms`` values. Both JSON reports copy these aggregates, so the
    two files can never disagree on metric or latency values.
    """
    if not results:
        raise ValueError("Cannot aggregate an empty result set")

    return {
        method_name: {
            metric: fmean(
                result.methods[method_name].metrics[metric]
                for result in results
            )
            for metric in METRIC_KEYS
        } | {
            "mean_latency_ms": round(
                fmean(
                    result.methods[method_name].latency_ms
                    for result in results
                ),
                3,
            ),
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
        "reproducibility": reproducibility_metadata(),
        "aggregate": aggregate,
        "questions": [
            {
                "question_id": result.question_id,
                "question": result.question,
                "relevant_chunks": result.relevant_chunks,
                "methods": {
                    method_name: {
                        "metrics": result.methods[method_name].metrics,
                        "latency_ms": result.methods[method_name].latency_ms,
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


def build_summary(report: dict, corpus_chunks: int) -> dict:
    """Derive the evaluation summary from the detailed report.

    Aggregate metric and latency values are copied from ``report``
    unchanged; nothing is recalculated, so both JSON files always describe
    the same evaluation run with identical values.
    """
    method_names = report["configuration"]["methods"]

    return {
        "generated_at": report["generated_at"],
        "dataset_size": len(report["questions"]),
        "corpus_chunks": corpus_chunks,
        "configuration": report["configuration"],
        "reproducibility": report["reproducibility"],
        "aggregate_metrics": {
            method_name: dict(report["aggregate"][method_name])
            for method_name in method_names
        },
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


def main() -> None:
    """Run the evaluation and generate all three reports from one run."""
    questions = get_questions()
    chunks = get_all_chunks()
    validate_dataset(questions, {_chunk_key(chunk) for chunk in chunks})

    results = [evaluate_question(question) for question in questions]
    aggregate = aggregate_metrics(results)
    report = build_report(results, aggregate)
    summary = build_summary(report, corpus_chunks=len(chunks))

    print_summary(aggregate)
    print_question_results(results)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown_report(MARKDOWN_PATH, report, corpus_chunks=len(chunks))

    print(f"\nJSON report: {REPORT_PATH}")
    print(f"JSON summary: {SUMMARY_PATH}")
    print(f"Markdown report: {MARKDOWN_PATH}")

    if not LABELS_VERIFIED:
        print(
            "NOTE: relevance labels are provisional and require manual "
            "verification before metrics are treated as final."
        )


if __name__ == "__main__":
    main()
