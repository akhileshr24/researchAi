"""Tests for evaluation orchestration, latency, metadata and reporting.

All retrieval is monkeypatched with deterministic fake chunks: no ChromaDB,
no embedding models, no cross-encoder models and no external APIs.
"""

import json
import re
import sys
from hashlib import sha256
from pathlib import Path

import pytest

from src.config import settings as settings_module
from src.rag.evaluation import evaluate, report
from src.rag.evaluation.dataset import EvaluationQuestion, RelevanceLabel

METHOD_NAMES = ("Vector", "BM25", "Hybrid RRF", "Hybrid + Cross-Encoder")
METRIC_KEYS = ("precision@5", "recall@5", "mrr", "ndcg@5")


def _chunk(
    document_id: str = "doc-a",
    chunk_index: int = 0,
    filename: str = "a.pdf",
    page: int = 1,
    text: str = "filler text",
) -> dict:
    return {
        "document_id": document_id,
        "chunk_index": chunk_index,
        "filename": filename,
        "page": page,
        "text": text,
    }


def _question() -> EvaluationQuestion:
    return EvaluationQuestion(
        question_id="q1",
        question="A test question?",
        relevant_chunks=(
            RelevanceLabel("doc-a", 0, 3),
            RelevanceLabel("doc-a", 1, 2),
        ),
    )


def _mixed_retrieval(method: str, question: str, **kwargs) -> list[dict]:
    """Deterministic per-method retrieval with some relevant chunks."""
    return [
        _chunk(chunk_index=0, text="first relevant chunk"),
        _chunk(document_id="doc-z", chunk_index=9),
        _chunk(chunk_index=1, text="second relevant chunk"),
        _chunk(document_id="doc-z", chunk_index=8),
    ]


def _irrelevant_retrieval(method: str, question: str, **kwargs) -> list[dict]:
    """Deterministic retrieval with no relevant chunks."""
    return [
        _chunk(document_id="doc-z", chunk_index=7),
        _chunk(document_id="doc-z", chunk_index=8),
    ]


def _evaluated(monkeypatch, retrieval) -> evaluate.QuestionResult:
    monkeypatch.setattr(evaluate, "retrieve", retrieval)
    return evaluate.evaluate_question(_question())


def _method_result(metrics: dict[str, float]) -> evaluate.MethodResult:
    return evaluate.MethodResult(
        method="method",
        retrieved=[],
        metrics=dict(metrics),
    )


def _question_result(
    question_id: str,
    metrics: dict[str, float],
) -> evaluate.QuestionResult:
    return evaluate.QuestionResult(
        question_id=question_id,
        question="A question?",
        relevant_chunks=[],
        methods={
            name: _method_result(metrics)
            for name in METHOD_NAMES
        },
    )


def _built_report(monkeypatch) -> dict:
    result = _evaluated(monkeypatch, _mixed_retrieval)
    aggregate = evaluate.aggregate_metrics([result])
    return evaluate.build_report([result], aggregate)


# ---------- evaluate_question ----------


def test_evaluate_question_includes_all_four_methods(monkeypatch):
    result = _evaluated(monkeypatch, _mixed_retrieval)

    assert tuple(result.methods) == METHOD_NAMES


def test_no_relevant_retrieved_gives_zero_scores(monkeypatch):
    result = _evaluated(monkeypatch, _irrelevant_retrieval)

    for name in METHOD_NAMES:
        metrics = result.methods[name].metrics

        assert metrics["precision@5"] == 0.0
        assert metrics["recall@5"] == 0.0
        assert metrics["mrr"] == 0.0
        assert metrics["ndcg@5"] == 0.0


def test_evaluate_question_marks_relevant_chunks(monkeypatch):
    result = _evaluated(monkeypatch, _mixed_retrieval)
    retrieved = result.methods["Vector"].retrieved

    assert [item.relevant for item in retrieved] == [True, False, True, False]
    assert [item.grade for item in retrieved] == [3, 0, 2, 0]


# ---------- aggregate_metrics ----------


def test_aggregate_metrics_is_exact_arithmetic_mean():
    first = _question_result(
        "q1",
        {
            "precision@5": 0.5,
            "recall@5": 0.5,
            "mrr": 1.0,
            "ndcg@5": 0.5,
        },
    )
    second = _question_result(
        "q2",
        {
            "precision@5": 0.25,
            "recall@5": 0.25,
            "mrr": 0.5,
            "ndcg@5": 0.25,
        },
    )

    aggregate = evaluate.aggregate_metrics([first, second])

    assert aggregate["Vector"]["precision@5"] == 0.375
    assert aggregate["Vector"]["recall@5"] == 0.375
    assert aggregate["Vector"]["mrr"] == 0.75
    assert aggregate["Vector"]["ndcg@5"] == 0.375


def test_aggregate_metrics_rejects_empty_results():
    with pytest.raises(ValueError, match="empty result set"):
        evaluate.aggregate_metrics([])


# ---------- build_report ----------


def test_build_report_returns_expected_keys(monkeypatch):
    result = _evaluated(monkeypatch, _mixed_retrieval)
    aggregate = evaluate.aggregate_metrics([result])
    report_dict = evaluate.build_report([result], aggregate)

    assert set(report_dict) == {
        "generated_at",
        "labels_verified",
        "label_status",
        "configuration",
        "reproducibility",
        "aggregate",
        "questions",
    }


def test_report_questions_have_expected_structure(monkeypatch):
    report_dict = _built_report(monkeypatch)
    question = report_dict["questions"][0]

    assert set(question) == {
        "question_id",
        "question",
        "relevant_chunks",
        "methods",
    }

    method_entry = question["methods"]["Vector"]

    assert set(method_entry) == {"metrics", "latency_ms", "retrieved"}
    assert set(METRIC_KEYS) == set(method_entry["metrics"])
    assert set(method_entry["retrieved"][0]) == {
        "rank",
        "document_id",
        "chunk_index",
        "filename",
        "page",
        "grade",
        "relevant",
        "score",
        "text_preview",
    }


def test_report_configuration_and_reproducibility(monkeypatch):
    report_dict = _built_report(monkeypatch)

    assert report_dict["configuration"] == {
        "top_k": 5,
        "candidate_limit": 20,
        "vector_min_score": 0.0,
        "rrf_k": settings_module.RRF_K,
        "vector_weight": 1.0,
        "bm25_weight": 1.0,
        "methods": list(METHOD_NAMES),
    }
    assert set(report_dict["reproducibility"]) == {
        "git_commit_sha",
        "python_version",
        "lock_file_sha256",
        "dataset_sha256",
    }


# ---------- latency ----------


def test_latency_is_non_negative_number(monkeypatch):
    result = _evaluated(monkeypatch, _mixed_retrieval)

    for name in METHOD_NAMES:
        latency = result.methods[name].latency_ms

        assert isinstance(latency, float)
        assert latency >= 0.0


def test_summary_mean_latency_matches_report(monkeypatch):
    report_dict = _built_report(monkeypatch)
    summary = evaluate.build_summary(report_dict, corpus_chunks=10)

    for name in METHOD_NAMES:
        latencies = [
            question["methods"][name]["latency_ms"]
            for question in report_dict["questions"]
        ]
        expected = sum(latencies) / len(latencies)

        assert summary["aggregate_metrics"][name]["mean_latency_ms"] == (
            pytest.approx(expected, abs=1e-6)
        )


def test_detailed_report_aggregate_contains_mean_latency(monkeypatch):
    report_dict = _built_report(monkeypatch)

    for name in METHOD_NAMES:
        mean_latency = report_dict["aggregate"][name]["mean_latency_ms"]

        assert isinstance(mean_latency, float)
        assert mean_latency >= 0.0


def test_summary_aggregate_contains_mean_latency(monkeypatch):
    summary = evaluate.build_summary(_built_report(monkeypatch), corpus_chunks=10)

    for name in METHOD_NAMES:
        mean_latency = summary["aggregate_metrics"][name]["mean_latency_ms"]

        assert isinstance(mean_latency, float)
        assert mean_latency >= 0.0


def test_mean_latency_identical_in_detailed_report_and_summary(monkeypatch):
    report_dict = _built_report(monkeypatch)
    summary = evaluate.build_summary(report_dict, corpus_chunks=10)

    for name in METHOD_NAMES:
        assert (
            report_dict["aggregate"][name]["mean_latency_ms"]
            == summary["aggregate_metrics"][name]["mean_latency_ms"]
        )


def test_mean_latency_is_arithmetic_mean_of_per_question_latencies(monkeypatch):
    report_dict = _built_report(monkeypatch)

    for name in METHOD_NAMES:
        latencies = [
            question["methods"][name]["latency_ms"]
            for question in report_dict["questions"]
        ]
        expected = round(sum(latencies) / len(latencies), 3)

        assert report_dict["aggregate"][name]["mean_latency_ms"] == expected


# ---------- build_summary ----------


def test_summary_matches_report_exactly(monkeypatch):
    report_dict = _built_report(monkeypatch)
    summary = evaluate.build_summary(report_dict, corpus_chunks=10)

    assert set(summary) == {
        "generated_at",
        "dataset_size",
        "corpus_chunks",
        "configuration",
        "reproducibility",
        "aggregate_metrics",
    }
    assert summary["generated_at"] == report_dict["generated_at"]
    assert summary["dataset_size"] == 1
    assert summary["corpus_chunks"] == 10
    assert summary["configuration"] == report_dict["configuration"]
    assert summary["reproducibility"] == report_dict["reproducibility"]

    for name in METHOD_NAMES:
        summary_metrics = summary["aggregate_metrics"][name]

        for key in METRIC_KEYS:
            assert summary_metrics[key] == report_dict["aggregate"][name][key]


# ---------- reproducibility metadata ----------


def test_dataset_hash_is_stable_and_64_hex():
    first = evaluate._dataset_hash()
    second = evaluate._dataset_hash()

    assert first == second
    assert re.fullmatch(r"[0-9a-f]{64}", first)


def test_lock_file_hash_of_existing_file(tmp_path):
    target = tmp_path / "uv.lock"
    target.write_bytes(b"lock file bytes")

    expected_hash = sha256(b"lock file bytes").hexdigest()

    assert evaluate._sha256_of_file(target) == expected_hash
    assert re.fullmatch(r"[0-9a-f]{64}", expected_hash)


def test_lock_file_hash_missing_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "LOCK_FILE_PATH", tmp_path / "missing.lock")

    assert evaluate._lock_file_hash() == ""


def test_git_commit_sha_success(monkeypatch):
    class Completed:
        returncode = 0
        stdout = "  abc123def  \n"

    monkeypatch.setattr(
        evaluate.subprocess,
        "run",
        lambda *args, **kwargs: Completed(),
    )

    assert evaluate._git_commit_sha() == "abc123def"


def test_git_commit_sha_unavailable_returns_empty(monkeypatch):
    def failing_run(*args, **kwargs):
        raise FileNotFoundError("git not installed")

    monkeypatch.setattr(evaluate.subprocess, "run", failing_run)
    assert evaluate._git_commit_sha() == ""


def test_git_commit_sha_error_returns_empty(monkeypatch):
    class Completed:
        returncode = 128
        stdout = ""
        stderr = "fatal: not a git repository"

    monkeypatch.setattr(
        evaluate.subprocess,
        "run",
        lambda *args, **kwargs: Completed(),
    )

    assert evaluate._git_commit_sha() == ""


def test_python_version_matches_interpreter():
    assert evaluate._python_version() == sys.version.split()[0]


def test_report_rrf_k_matches_settings(monkeypatch):
    # build_report reads `settings.RRF_K` (module attribute) at call time;
    # evaluate.py imports it as `from src.config import settings`, so the
    # monkeypatch target `src.config.settings.RRF_K` is the same object.
    monkeypatch.setattr("src.config.settings.RRF_K", 17)

    report_dict = _built_report(monkeypatch)

    assert report_dict["configuration"]["rrf_k"] == 17


def test_report_records_actual_rrf_parameters(monkeypatch):
    result = _evaluated(monkeypatch, _mixed_retrieval)
    aggregate = evaluate.aggregate_metrics([result])

    report_dict = evaluate.build_report(
        [result],
        aggregate,
        rrf_k=30,
        vector_weight=2.0,
        bm25_weight=1.0,
    )

    configuration = report_dict["configuration"]

    assert configuration["rrf_k"] == 30
    assert configuration["vector_weight"] == 2.0
    assert configuration["bm25_weight"] == 1.0


def test_report_defaults_use_settings_rrf_k_and_unit_weights(monkeypatch):
    report_dict = _built_report(monkeypatch)
    configuration = report_dict["configuration"]

    assert configuration["rrf_k"] == settings_module.RRF_K
    assert configuration["vector_weight"] == 1.0
    assert configuration["bm25_weight"] == 1.0


# ---------- retrieve() parameter propagation ----------


def _record_search_hybrid(monkeypatch, candidates: list[dict]) -> dict:
    """Stub evaluate.search_hybrid and return the recording dict."""
    calls: dict = {}

    def fake_search_hybrid(
        question,
        top_k,
        candidate_k=None,
        rrf_k=None,
        document_id=None,
        vector_weight=1.0,
        bm25_weight=1.0,
    ):
        calls["question"] = question
        calls["top_k"] = top_k
        calls["candidate_k"] = candidate_k
        calls["rrf_k"] = rrf_k
        calls["vector_weight"] = vector_weight
        calls["bm25_weight"] = bm25_weight
        return candidates

    monkeypatch.setattr(evaluate, "search_hybrid", fake_search_hybrid)
    return calls


def test_retrieve_hybrid_propagates_rrf_parameters(monkeypatch):
    calls = _record_search_hybrid(monkeypatch, [])

    result = evaluate.retrieve(
        "hybrid",
        "What is RAG?",
        rrf_k=30,
        vector_weight=2.0,
        bm25_weight=0.5,
    )

    assert result == []
    assert calls == {
        "question": "What is RAG?",
        "top_k": 5,
        "candidate_k": 20,
        "rrf_k": 30,
        "vector_weight": 2.0,
        "bm25_weight": 0.5,
    }


def test_retrieve_hybrid_defaults_use_settings_and_unit_weights(monkeypatch):
    calls = _record_search_hybrid(monkeypatch, [])

    evaluate.retrieve("hybrid", "What is RAG?")

    assert calls["top_k"] == 5
    assert calls["candidate_k"] == 20
    assert calls["rrf_k"] == settings_module.RRF_K
    assert calls["vector_weight"] == 1.0
    assert calls["bm25_weight"] == 1.0


def test_retrieve_hybrid_rerank_propagates_rrf_parameters(monkeypatch):
    calls = _record_search_hybrid(
        monkeypatch,
        [{"document_id": "doc-a", "chunk_index": 0}],
    )
    reranked = [
        {"document_id": "doc-a", "chunk_index": 0, "rerank_score": 0.9},
    ]
    rerank_calls: dict = {}

    def fake_rerank(query, candidates, top_k=None):
        rerank_calls["query"] = query
        rerank_calls["candidates"] = candidates
        rerank_calls["top_k"] = top_k
        return reranked

    monkeypatch.setattr(evaluate, "rerank", fake_rerank)

    result = evaluate.retrieve(
        "hybrid_rerank",
        "What is RAG?",
        rrf_k=45,
        vector_weight=0.5,
        bm25_weight=2.0,
    )

    assert result == reranked  # reranker output returned unchanged
    assert calls == {
        "question": "What is RAG?",
        "top_k": 20,
        "candidate_k": 20,
        "rrf_k": 45,
        "vector_weight": 0.5,
        "bm25_weight": 2.0,
    }
    assert rerank_calls == {
        "query": "What is RAG?",
        "candidates": [{"document_id": "doc-a", "chunk_index": 0}],
        "top_k": 5,
    }


def test_retrieve_hybrid_rerank_defaults_use_settings_and_unit_weights(
    monkeypatch,
):
    calls = _record_search_hybrid(monkeypatch, [])
    monkeypatch.setattr(evaluate, "rerank", lambda *args, **kwargs: [])

    evaluate.retrieve("hybrid_rerank", "What is RAG?")

    assert calls["top_k"] == 20
    assert calls["candidate_k"] == 20
    assert calls["rrf_k"] == settings_module.RRF_K
    assert calls["vector_weight"] == 1.0
    assert calls["bm25_weight"] == 1.0


# ---------- CLI arguments ----------


def test_parse_args_defaults_to_none_out():
    args = evaluate.parse_args([])

    assert args.out is None


def test_parse_args_accepts_out_path():
    args = evaluate.parse_args(["--out", "reports/rag_evaluation/sweep/x"])

    assert args.out == Path("reports/rag_evaluation/sweep/x")


def _baseline_report_hashes() -> dict[str, str]:
    """SHA-256 of the three real baseline reports (protection check)."""
    names = (
        "evaluation_results.json",
        "evaluation_summary.json",
        "latest.md",
    )
    return {
        name: sha256(
            (Path("reports/rag_evaluation") / name).read_bytes()
        ).hexdigest()
        for name in names
    }


def _stub_main_env(monkeypatch) -> None:
    """Make evaluate.main() runnable without ChromaDB or models."""
    monkeypatch.setattr(evaluate, "get_questions", lambda: [_question()])
    monkeypatch.setattr(
        evaluate,
        "get_all_chunks",
        lambda: [
            _chunk(document_id="doc-a", chunk_index=0, text="relevant one"),
            _chunk(document_id="doc-a", chunk_index=1, text="relevant two"),
        ],
    )
    monkeypatch.setattr(
        evaluate,
        "evaluate_question",
        lambda question, **kwargs: _question_result(
            question.question_id,
            {key: 0.0 for key in METRIC_KEYS},
        ),
    )


def test_cli_out_writes_three_files_into_dir(tmp_path, monkeypatch):
    _stub_main_env(monkeypatch)
    out_dir = tmp_path / "sweep"
    before = _baseline_report_hashes()

    exit_code = evaluate.main(["--out", str(out_dir)])

    after = _baseline_report_hashes()

    assert exit_code == 0
    assert (out_dir / "evaluation_results.json").is_file()
    assert (out_dir / "evaluation_summary.json").is_file()
    assert (out_dir / "latest.md").is_file()
    assert after == before  # default reports untouched

    results = json.loads(
        (out_dir / "evaluation_results.json").read_text(encoding="utf-8")
    )
    assert results["configuration"]["rrf_k"] == settings_module.RRF_K


def test_cli_rrf_parameters_are_recorded_in_saved_reports(tmp_path, monkeypatch):
    _stub_main_env(monkeypatch)
    out_dir = tmp_path / "sweep"
    before = _baseline_report_hashes()

    exit_code = evaluate.main([
        "--out", str(out_dir),
        "--rrf-k", "45",
        "--vector-weight", "0.5",
        "--bm25-weight", "2",
    ])
    after = _baseline_report_hashes()

    assert exit_code == 0
    assert after == before  # baseline untouched

    results = json.loads(
        (out_dir / "evaluation_results.json").read_text(encoding="utf-8")
    )
    configuration = results["configuration"]

    assert configuration["rrf_k"] == 45
    assert configuration["vector_weight"] == 0.5
    assert configuration["bm25_weight"] == 2.0


def test_cli_without_out_writes_to_default_dir(tmp_path, monkeypatch):
    _stub_main_env(monkeypatch)
    monkeypatch.setattr(evaluate, "REPORT_DIR", tmp_path)  # never touches real dir

    exit_code = evaluate.main([])

    assert exit_code == 0
    assert (tmp_path / "evaluation_results.json").is_file()
    assert (tmp_path / "evaluation_summary.json").is_file()
    assert (tmp_path / "latest.md").is_file()
    assert not (tmp_path / "sweep").exists()  # no --out dir was created

    results = json.loads(
        (tmp_path / "evaluation_results.json").read_text(encoding="utf-8")
    )
    assert results["configuration"]["rrf_k"] == settings_module.RRF_K


# ---------- Markdown report ----------


def _markdown(monkeypatch, tmp_path) -> str:
    path = tmp_path / "latest.md"
    report_dict = _built_report(monkeypatch)
    report.write_markdown_report(path, report_dict, corpus_chunks=604)
    return path.read_text(encoding="utf-8")


def test_markdown_contains_headline_table(monkeypatch, tmp_path):
    text = _markdown(monkeypatch, tmp_path)
    headline = text.split("## Headline metrics", 1)[1].split(
        "## Pipeline comparison", 1
    )[0]

    assert "| Method |" in headline
    assert headline.count("| Vector |") == 1
    assert headline.count("| BM25 |") == 1
    assert headline.count("| Hybrid RRF |") == 1
    assert headline.count("| Hybrid + CrossEncoder |") == 1


def test_markdown_contains_required_sections(monkeypatch, tmp_path):
    text = _markdown(monkeypatch, tmp_path)

    assert "# ResearchAI RAG Evaluation" in text
    assert "Generated:" in text
    assert "## Reproducibility" in text
    assert "## Headline metrics" in text
    assert "## Pipeline comparison" in text
    assert "## Per-question breakdown" in text
    assert "Dataset SHA-256" in text
    assert "Lock file SHA-256" in text


def test_markdown_contains_pipeline_comparisons(monkeypatch, tmp_path):
    text = _markdown(monkeypatch, tmp_path)

    assert "BM25 vs Vector" in text
    assert "Hybrid RRF vs BM25" in text
    assert "Hybrid + CrossEncoder vs Hybrid RRF" in text


def test_markdown_note_uses_actual_dataset_size(monkeypatch, tmp_path):
    path = tmp_path / "latest.md"
    report_dict = _built_report(monkeypatch)  # 1 question, not 17

    report.write_markdown_report(path, report_dict, corpus_chunks=10)
    text = path.read_text(encoding="utf-8")

    assert f"({len(report_dict['questions'])} questions)" in text
    assert "(1 questions)" in text
    assert "(17 questions)" not in text


def test_markdown_note_tracks_dataset_size_changes(monkeypatch, tmp_path):
    path = tmp_path / "latest.md"
    report_dict = _built_report(monkeypatch)
    report_dict["questions"] = report_dict["questions"] * 3  # 3 questions

    report.write_markdown_report(path, report_dict, corpus_chunks=10)
    text = path.read_text(encoding="utf-8")

    assert "(3 questions)" in text
    assert "(1 questions)" not in text
    assert "(17 questions)" not in text


def test_markdown_contains_per_question_results(monkeypatch, tmp_path):
    text = _markdown(monkeypatch, tmp_path)
    per_question = text.split("## Per-question breakdown", 1)[1]

    assert "| q1 |" in per_question
    assert "first rel. rank" in per_question


def test_markdown_footer_reflects_labels_verified_flag(monkeypatch, tmp_path):
    path = tmp_path / "latest.md"
    report_dict = _built_report(monkeypatch)

    # The committed dataset is marked as reviewed; the footer renders the
    # report's flag, so patch it to check both branches.
    report_dict["labels_verified"] = False
    report.write_markdown_report(path, report_dict, corpus_chunks=604)
    footer = path.read_text(encoding="utf-8").split("## Footer", 1)[1]
    assert "Labels verified: No" in footer

    report_dict["labels_verified"] = True
    report.write_markdown_report(path, report_dict, corpus_chunks=604)
    footer = path.read_text(encoding="utf-8").split("## Footer", 1)[1]
    assert "Labels verified: Yes" in footer
