"""Markdown rendering of the RAG evaluation report.

``write_markdown_report`` renders ``reports/rag_evaluation/latest.md`` from
the same in-memory report dict that produces the JSON files, so all three
reports always describe the same evaluation run. Mean latency is averaged
from the report's per-question latency values.

The dataset currently contains a small number of questions, so the pipeline
comparison table shows raw deltas only. Statistical significance is
deliberately not claimed.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from statistics import fmean

# Display labels for the Markdown tables. JSON keys are never renamed.
DISPLAY_NAMES: dict[str, str] = {
    "Vector": "Vector",
    "BM25": "BM25",
    "Hybrid RRF": "Hybrid RRF",
    "Hybrid + Cross-Encoder": "Hybrid + CrossEncoder",
}

PIPELINE_STAGES = ("BM25", "Hybrid RRF", "Hybrid + Cross-Encoder")
COMPARISON_METRICS = ("recall@5", "ndcg@5")

NO_RANK_PLACEHOLDER = "-"


def _display_name(method: str) -> str:
    return DISPLAY_NAMES.get(method, method)


def _metric(value: float) -> str:
    return f"{value:.4f}"


def _delta(value: float) -> str:
    return f"{value:+.4f}"


def _first_relevant_rank(items: Sequence[dict]) -> str:
    for item in items:
        if item["relevant"]:
            return str(item["rank"])

    return NO_RANK_PLACEHOLDER


def _mean_latency_ms(report: dict, method: str) -> float:
    """Return the mean per-question latency for one method."""
    return round(
        fmean(
            question["methods"][method]["latency_ms"]
            for question in report["questions"]
        ),
        3,
    )


def _headline_table(report: dict) -> str:
    methods = report["configuration"]["methods"]
    lines = [
        "| Method | P@5 | R@5 | MRR | nDCG@5 | Mean latency (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for method in methods:
        metrics = report["aggregate"][method]
        lines.append(
            f"| {_display_name(method)} "
            f"| {_metric(metrics['precision@5'])} "
            f"| {_metric(metrics['recall@5'])} "
            f"| {_metric(metrics['mrr'])} "
            f"| {_metric(metrics['ndcg@5'])} "
            f"| {_metric(_mean_latency_ms(report, method))} |"
        )

    return "\n".join(lines)


def _pipeline_table(report: dict) -> str:
    aggregate = report["aggregate"]
    lines = [
        "| Comparison | Recall@5 delta | nDCG@5 delta |",
        "|---|---:|---:|",
    ]

    for method, previous in zip(PIPELINE_STAGES, report["configuration"]["methods"]):
        deltas = [
            aggregate[method][metric] - aggregate[previous][metric]
            for metric in COMPARISON_METRICS
        ]
        lines.append(
            f"| {_display_name(method)} vs {_display_name(previous)} "
            f"| {_delta(deltas[0])} "
            f"| {_delta(deltas[1])} |"
        )

    return "\n".join(lines)


def _per_question_table(report: dict) -> str:
    methods = report["configuration"]["methods"]
    header = "| Question ID | " + " | ".join(
        f"{_display_name(method)} R@5" for method in methods
    ) + " | " + " | ".join(
        f"{_display_name(method)} first rel. rank" for method in methods
    ) + " |"
    separator = "|---" + "|---:" * (2 * len(methods)) + "|"

    lines = [header, separator]

    for question in report["questions"]:
        recall_cells = [
            _metric(question["methods"][method]["metrics"]["recall@5"])
            for method in methods
        ]
        rank_cells = [
            _first_relevant_rank(question["methods"][method]["retrieved"])
            for method in methods
        ]
        lines.append(
            f"| {question['question_id']} | "
            + " | ".join(recall_cells)
            + " | "
            + " | ".join(rank_cells)
            + " |"
        )

    return "\n".join(lines)


def _reproducibility_block(report: dict, corpus_chunks: int | None) -> str:
    metadata = report["reproducibility"]
    configuration = report["configuration"]
    corpus_line = (
        f"Corpus chunks: {corpus_chunks}"
        if corpus_chunks is not None
        else "Corpus chunks: see evaluation_summary.json"
    )

    return "\n".join([
        "## Reproducibility",
        "",
        f"- Git commit SHA: `{metadata['git_commit_sha'] or 'unavailable'}`",
        f"- Python version: `{metadata['python_version']}`",
        f"- Lock file SHA-256: `{metadata['lock_file_sha256'] or 'unavailable'}`",
        f"- Dataset SHA-256: `{metadata['dataset_sha256']}`",
        f"- Dataset size: {len(report['questions'])} questions",
        f"- {corpus_line}",
        f"- top_k: {configuration['top_k']}",
        f"- candidate_limit: {configuration['candidate_limit']}",
        f"- vector_min_score: {configuration['vector_min_score']}",
    ])


def _footer(report: dict) -> str:
    labels_verified = "Yes" if report["labels_verified"] else "No"

    return "\n".join([
        "## Footer",
        "",
        f"- Dataset size: {len(report['questions'])} questions",
        f"- Labels verified: {labels_verified}",
    ])


def render_markdown(report: dict, corpus_chunks: int | None = None) -> str:
    """Render the full Markdown evaluation report as a string."""
    corpus_line = (
        f"Corpus chunks: {corpus_chunks}"
        if corpus_chunks is not None
        else "Corpus chunks: see evaluation_summary.json"
    )

    sections = [
        "# ResearchAI RAG Evaluation",
        "",
        f"Generated: {report['generated_at']}",
        "",
        _reproducibility_block(report, corpus_chunks),
        "",
        "## Headline metrics",
        "",
        _headline_table(report),
        "",
        "## Pipeline comparison",
        "",
        _pipeline_table(report),
        "",
        "Note: deltas are raw differences on a small dataset "
        f"({len(report['questions'])} questions); "
        "no statistical significance is claimed.",
        "",
        "## Per-question breakdown",
        "",
        _per_question_table(report),
        "",
        _footer(report),
        "",
    ]

    return "\n".join(sections)


def write_markdown_report(
    path: str | Path,
    report: dict,
    corpus_chunks: int | None = None,
) -> None:
    """Write the Markdown evaluation report to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_markdown(report, corpus_chunks=corpus_chunks),
        encoding="utf-8",
    )
