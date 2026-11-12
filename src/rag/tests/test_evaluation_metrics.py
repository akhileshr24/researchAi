"""Tests for retrieval metrics and evaluation-dataset validation."""

from math import log2

import pytest

from src.rag.evaluation.dataset import (
    EvaluationQuestion,
    RelevanceLabel,
    validate_dataset,
)
from src.rag.evaluation.metrics import (
    dcg_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def _question(
    question_id: str = "q1",
    labels: tuple[tuple[str, int, int], ...] = (("doc-a", 0, 3),),
) -> EvaluationQuestion:
    return EvaluationQuestion(
        question_id=question_id,
        question="A test question?",
        relevant_chunks=tuple(
            RelevanceLabel(document_id, chunk_index, grade)
            for document_id, chunk_index, grade in labels
        ),
    )


# ---------- Precision@K ----------


def test_precision_at_k_all_results_relevant():
    assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == 1.0


def test_precision_at_k_partially_relevant():
    assert precision_at_k(["a", "x", "b", "y"], {"a", "b"}, 4) == 0.5


def test_precision_at_k_uses_k_as_denominator():
    assert precision_at_k(["a"], {"a"}, 4) == 0.25


def test_precision_at_k_no_relevant_results():
    assert precision_at_k(["x", "y"], {"a"}, 2) == 0.0


def test_precision_at_k_empty_results():
    assert precision_at_k([], {"a"}, 3) == 0.0


def test_precision_at_k_empty_ground_truth():
    assert precision_at_k(["a"], set(), 3) == 0.0


# ---------- Recall@K ----------


def test_recall_at_k_all_relevant_retrieved():
    assert recall_at_k(["a", "b"], {"a", "b"}, 2) == 1.0


def test_recall_at_k_partial_results():
    assert recall_at_k(["a", "x", "y"], {"a", "b", "c"}, 3) == 1 / 3


def test_recall_at_k_no_relevant_results():
    assert recall_at_k(["x", "y"], {"a"}, 2) == 0.0


def test_recall_at_k_empty_results():
    assert recall_at_k([], {"a"}, 3) == 0.0


def test_recall_at_k_empty_ground_truth():
    assert recall_at_k(["a"], set(), 3) == 0.0


def test_recall_at_k_counts_unique_relevant_items():
    assert recall_at_k(["a"], ["a", "a"], 1) == 1.0


# ---------- Reciprocal rank ----------


def test_reciprocal_rank_first_result():
    assert reciprocal_rank(["a", "b"], {"a"}) == 1.0


def test_reciprocal_rank_second_result():
    assert reciprocal_rank(["x", "a", "b"], {"a", "b"}) == 0.5


def test_reciprocal_rank_third_result():
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)


def test_reciprocal_rank_no_relevant_results():
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_reciprocal_rank_empty_results_or_ground_truth():
    assert reciprocal_rank([], {"a"}) == 0.0
    assert reciprocal_rank(["a"], set()) == 0.0


# ---------- nDCG@K ----------


def test_ndcg_is_one_for_perfect_ranking():
    relevance = {"a": 3, "b": 2}
    assert ndcg_at_k(["a", "b"], relevance, 2) == pytest.approx(1.0)


def test_ndcg_uses_graded_relevance_and_penalizes_order():
    relevance = {"a": 3, "b": 1}
    # Relevant chunk graded 1 appears before the chunk graded 3.
    assert ndcg_at_k(["b", "a"], relevance, 2) == pytest.approx(
        0.7098097414,
        rel=1e-9,
    )


def test_ndcg_ignores_non_relevant_items_between_hits():
    relevance = {"a": 3, "b": 1}
    perfect = ndcg_at_k(["a", "b"], relevance, 2)
    interleaved = ndcg_at_k(["a", "x", "b"], relevance, 3)

    expected_dcg = 7 + 1 / log2(4)
    ideal_dcg = 7 + 1 / log2(3)

    assert interleaved < perfect
    assert interleaved == pytest.approx(
        expected_dcg / ideal_dcg,
        rel=1e-9,
    )


def test_ndcg_respects_cutoff():
    relevance = {"c": 3}

    assert ndcg_at_k(["a", "b", "c"], relevance, 2) == 0.0
    assert ndcg_at_k(["a", "b", "c"], relevance, 3) == pytest.approx(0.5)


def test_ndcg_empty_results():
    assert ndcg_at_k([], {"a": 3}, 3) == 0.0


def test_ndcg_empty_ground_truth():
    assert ndcg_at_k(["a"], {}, 3) == 0.0


def test_ndcg_zero_grades_are_not_relevant():
    assert ndcg_at_k(["a"], {"a": 0}, 3) == 0.0


def test_dcg_uses_graded_relevance():
    assert dcg_at_k(["a", "x"], {"a": 3}, 2) == pytest.approx(7.0)


# ---------- Invalid input ----------


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_metrics_reject_invalid_k(k):
    with pytest.raises(ValueError, match="k must be a positive integer"):
        precision_at_k(["a"], {"a"}, k)

    with pytest.raises(ValueError, match="k must be a positive integer"):
        recall_at_k(["a"], {"a"}, k)

    with pytest.raises(ValueError, match="k must be a positive integer"):
        ndcg_at_k(["a"], {"a": 3}, k)


def test_ndcg_rejects_negative_grades():
    with pytest.raises(ValueError, match="cannot be negative"):
        ndcg_at_k(["a"], {"a": -1}, 3)


def test_ndcg_rejects_non_numeric_grades():
    with pytest.raises(ValueError, match="must be numeric"):
        ndcg_at_k(["a"], {"a": "high"}, 3)


# ---------- Dataset validation ----------


def test_relevance_label_rejects_invalid_grade():
    with pytest.raises(ValueError, match="between 1 and 3"):
        RelevanceLabel("doc-a", 0, 0)

    with pytest.raises(ValueError, match="between 1 and 3"):
        RelevanceLabel("doc-a", 0, 4)


def test_evaluation_question_requires_relevant_chunks():
    with pytest.raises(ValueError, match="no relevant chunks"):
        EvaluationQuestion("q1", "Question?", ())


def test_validate_dataset_accepts_available_chunks():
    validate_dataset(
        [_question(labels=(("doc-a", 0, 3), ("doc-a", 1, 2)))],
        {("doc-a", 0), ("doc-a", 1)},
    )


def test_validate_dataset_rejects_missing_chunks():
    with pytest.raises(ValueError, match="not stored"):
        validate_dataset(
            [_question(labels=(("doc-a", 99, 3),))],
            {("doc-a", 0)},
        )


def test_validate_dataset_rejects_duplicate_question_ids():
    with pytest.raises(ValueError, match="Duplicate question_id"):
        validate_dataset([
            _question(question_id="same"),
            _question(question_id="same"),
        ])


def test_validate_dataset_rejects_empty_dataset():
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_dataset([])
