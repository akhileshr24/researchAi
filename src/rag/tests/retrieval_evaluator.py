
import math


def precision_at_k(retrieved, relevant, k):
    """
    Fraction of top-k retrieved chunks that are relevant.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    top_k = retrieved[:k]

    if not top_k:
        return 0.0

    relevant_count = sum(
        1 for chunk in top_k if chunk in relevant
    )

    return relevant_count / k


def recall_at_k(retrieved, relevant, k):
    """
    Fraction of all relevant chunks found in top-k.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    if not relevant:
        return 0.0

    top_k = retrieved[:k]

    relevant_count = sum(
        1 for chunk in top_k if chunk in relevant
    )

    return relevant_count / len(relevant)


def reciprocal_rank(retrieved, relevant):
    """
    Reciprocal rank of the first relevant result.
    """
    for rank, chunk in enumerate(retrieved, start=1):
        if chunk in relevant:
            return 1 / rank

    return 0.0


def ndcg_at_k(retrieved, relevance_grades, k):
    """
    Normalized Discounted Cumulative Gain.
    Relevance grades: 0, 1, 2, etc.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    def dcg(grades):
        return sum(
            (2 ** grade - 1) / math.log2(rank + 1)
            for rank, grade in enumerate(grades, start=1)
        )

    actual_grades = [
        relevance_grades.get(chunk, 0)
        for chunk in retrieved[:k]
    ]

    ideal_grades = sorted(
        relevance_grades.values(),
        reverse=True
    )[:k]

    ideal_score = dcg(ideal_grades)

    if ideal_score == 0:
        return 0.0

    return dcg(actual_grades) / ideal_score
