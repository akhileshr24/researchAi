
"""Evaluation questions and graded ground-truth chunk labels.

Each label identifies a stored chunk by (document_id, chunk_index).
The labels below were manually reviewed against the text of the chunks
stored in the local ChromaDB collection; no IDs are guessed.

Relevance grades
----------------
3
    The chunk directly answers the question.
2
    The chunk strongly supports the answer.
1
    The chunk provides related context.
0
    Not relevant (omitted from the labels).

REVIEW STATUS
-------------
All listed labels have been manually reviewed against stored chunk text.
This does not guarantee that every relevant chunk in the corpus is labelled.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

# Source documents currently stored in data/chroma.
DEEP_WORK_ID = "4fd44b31a3a08548"
TEST_PDF_ID = "890dc7fc62041257"

LABELS_VERIFIED = True
LABEL_STATUS = "manually reviewed against stored chunk text"

MIN_GRADE = 1
MAX_GRADE = 3


@dataclass(frozen=True)
class RelevanceLabel:
    """One graded relevance judgement for a stored chunk."""

    document_id: str
    chunk_index: int
    grade: int = MAX_GRADE

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("document_id cannot be empty")

        if self.chunk_index < 0:
            raise ValueError("chunk_index cannot be negative")

        if not MIN_GRADE <= self.grade <= MAX_GRADE:
            raise ValueError(
                f"relevance grade must be between {MIN_GRADE} and "
                f"{MAX_GRADE}, got {self.grade}"
            )

    def key(self) -> tuple[str, int]:
        """Return the chunk identifier used by the retrieval functions."""
        return (self.document_id, self.chunk_index)


@dataclass(frozen=True)
class EvaluationQuestion:
    """A question with its graded ground-truth chunks."""

    question_id: str
    question: str
    relevant_chunks: tuple[RelevanceLabel, ...]
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.question_id.strip():
            raise ValueError("question_id cannot be empty")

        if not self.question.strip():
            raise ValueError("question cannot be empty")

        if not self.relevant_chunks:
            raise ValueError(
                f"question '{self.question_id}' has no relevant chunks"
            )

    def relevance_grades(self) -> dict[tuple[str, int], int]:
        """Return {(document_id, chunk_index): grade} for this question."""
        return {
            label.key(): label.grade
            for label in self.relevant_chunks
        }


def _deep(chunk_index: int, grade: int = MAX_GRADE) -> RelevanceLabel:
    return RelevanceLabel(DEEP_WORK_ID, chunk_index, grade)


def _test(chunk_index: int, grade: int = MAX_GRADE) -> RelevanceLabel:
    return RelevanceLabel(TEST_PDF_ID, chunk_index, grade)


_QUESTIONS: tuple[EvaluationQuestion, ...] = (
    EvaluationQuestion(
        question_id="ai-ml-dl-difference",
        question=(
            "What is the difference between artificial intelligence, "
            "machine learning, and deep learning?"
        ),
        relevant_chunks=(
            _test(2, 3),
            _test(1, 1),
            _test(7, 1),
        ),
    ),
    EvaluationQuestion(
        question_id="overfitting-test-set",
        question=(
            "What is overfitting and why is a held-out test set important?"
        ),
        relevant_chunks=(
            _test(5, 3),
            _test(4, 1),
        ),
    ),
    EvaluationQuestion(
        question_id="rag-pipeline",
        question="How does Retrieval-Augmented Generation work?",
        relevant_chunks=(
            _test(10, 3),
            _test(11, 2),
        ),
    ),
    EvaluationQuestion(
        question_id="classification-regression-metrics",
        question=(
            "What are common evaluation metrics for classification and "
            "regression models?"
        ),
        relevant_chunks=(
            _test(6, 3),
            _test(5, 1),
        ),
    ),
    EvaluationQuestion(
        question_id="machine-learning-types",
        question="What are the main types of machine learning?",
        relevant_chunks=(
            _test(4, 3),
        ),
    ),
    EvaluationQuestion(
        question_id="rag-evaluation",
        question="How should a RAG system be evaluated?",
        relevant_chunks=(
            _test(11, 3),
            _test(12, 1),
        ),
    ),
    EvaluationQuestion(
        question_id="llm-limitations",
        question="What are the limitations of large language models?",
        relevant_chunks=(
            _test(8, 3),
            _test(9, 3),
        ),
    ),
    EvaluationQuestion(
        question_id="deep-work-hypothesis",
        question="What is the Deep Work Hypothesis?",
        relevant_chunks=(
            _deep(29, 3),
            _deep(30, 3),
        ),
    ),
    EvaluationQuestion(
        question_id="productivity-formula",
        question="What is the formula for high-quality work produced?",
        relevant_chunks=(
            _deep(78, 3),
            _deep(79, 2),
            _deep(81, 1),
        ),
    ),
    EvaluationQuestion(
        question_id="attention-residue",
        question=(
            "What is attention residue and how does it affect performance?"
        ),
        relevant_chunks=(
            _deep(79, 2),
            _deep(81, 3),
            _deep(82, 2),
            _deep(83, 2),
        ),
    ),
    EvaluationQuestion(
        question_id="deliberate-practice",
        question="What is deliberate practice and what does it require?",
        relevant_chunks=(
            _deep(66, 3),
            _deep(67, 3),
            _deep(68, 3),
            _deep(69, 1),
        ),
    ),
    EvaluationQuestion(
        question_id="principle-of-least-resistance",
        question="What is the Principle of Least Resistance?",
        relevant_chunks=(
            _deep(109, 2),
            _deep(111, 3),
            _deep(112, 2),
        ),
    ),
    EvaluationQuestion(
        question_id="monastic-philosophy",
        question="What is the monastic philosophy of deep work scheduling?",
        relevant_chunks=(
            _deep(198, 3),
            _deep(199, 2),
        ),
    ),
    EvaluationQuestion(
        question_id="bimodal-philosophy",
        question="What is the bimodal philosophy of deep work?",
        relevant_chunks=(
            _deep(206, 3),
            _deep(208, 2),
        ),
    ),
    EvaluationQuestion(
        question_id="rhythmic-philosophy",
        question="What is the rhythmic philosophy of deep work scheduling?",
        relevant_chunks=(
            _deep(211, 1),
            _deep(213, 3),
        ),
    ),
    EvaluationQuestion(
        question_id="fixed-schedule-productivity",
        question="What is fixed-schedule productivity?",
        relevant_chunks=(
            _deep(459, 3),
            _deep(468, 2),
            _deep(469, 2),
        ),
    ),
    EvaluationQuestion(
        question_id="any-benefit-vs-craftsman",
        question=(
            "What is the any-benefit approach to network tool selection, "
            "and what is the alternative?"
        ),
        relevant_chunks=(
            _deep(359, 3),
            _deep(360, 2),
            _deep(369, 3),
            _deep(370, 2),
            _deep(371, 1),
        ),
    ),
)


def get_questions() -> tuple[EvaluationQuestion, ...]:
    """Return the evaluation questions in a stable order."""
    return _QUESTIONS


def validate_dataset(
    questions: Sequence[EvaluationQuestion] | None = None,
    available_chunks: Collection[tuple[str, int]] | None = None,
) -> None:
    """Validate question IDs, labels and optional chunk availability.

    ``available_chunks`` should contain the (document_id, chunk_index)
    pairs of the stored chunks. When supplied, a missing label raises a
    ValueError that names the offending question and chunk.
    """
    items = list(get_questions() if questions is None else questions)

    if not items:
        raise ValueError("Evaluation dataset cannot be empty")

    seen_ids: set[str] = set()

    for item in items:
        if item.question_id in seen_ids:
            raise ValueError(
                f"Duplicate question_id: {item.question_id}"
            )
        seen_ids.add(item.question_id)

        label_keys = [label.key() for label in item.relevant_chunks]

        if len(label_keys) != len(set(label_keys)):
            raise ValueError(
                f"Question '{item.question_id}' contains duplicate labels"
            )

        if available_chunks is None:
            continue

        missing = [
            key for key in label_keys
            if key not in available_chunks
        ]

        if missing:
            formatted = ", ".join(
                f"document_id={document_id}, chunk_index={chunk_index}"
                for document_id, chunk_index in missing
            )
            raise ValueError(
                f"Question '{item.question_id}' references chunks that are "
                f"not stored: {formatted}"
            )
