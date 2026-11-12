"""Information-retrieval metrics for ranked retrieval results.

All functions accept ``retrieved`` as an ordered sequence of item identifiers
(most relevant first) and compare it with ground-truth identifiers. Identifiers
only need to be hashable, so callers may use strings or
``(document_id, chunk_index)`` tuples.

Definitions
-----------
Precision@K
    Fraction of the top K retrieved items that are relevant. The denominator is
    K, even when fewer than K items were returned.
Recall@K
    Fraction of all unique relevant items that appear in the top K.
Reciprocal rank
    ``1 / rank`` of the first relevant item, or 0 when none is retrieved. The
    mean over queries is Mean Reciprocal Rank (MRR).
nDCG@K
    ``DCG@K / ideal DCG@K`` where
    ``DCG@K = sum((2 ** grade - 1) / log2(rank + 1))`` for ranks starting at 1.

Edge cases
----------
Queries with no relevant items return 0.0 for every metric, as do empty result
lists. Invalid ``k`` values and negative relevance grades raise ``ValueError``
so mistakes are visible instead of silently producing meaningless numbers.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from math import log2
from typing import Hashable, TypeVar

Item = TypeVar("Item", bound=Hashable)


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}")


def _top_k(retrieved: Sequence[Item], k: int) -> list[Item]:
    _validate_k(k)
    return list(retrieved[:k])


def precision_at_k(
    retrieved: Sequence[Item],
    relevant: Collection[Item],
    k: int,
) -> float:
    """Return the fraction of the top ``k`` results that are relevant."""
    top = _top_k(retrieved, k)

    if not top or not relevant:
        return 0.0

    hits = sum(1 for item in top if item in relevant)
    return hits / k


def recall_at_k(
    retrieved: Sequence[Item],
    relevant: Collection[Item],
    k: int,
) -> float:
    """Return the fraction of unique relevant items found in the top ``k``."""
    top = _top_k(retrieved, k)
    relevant_items = set(relevant)

    if not relevant_items:
        return 0.0

    hits = sum(1 for item in top if item in relevant_items)
    return hits / len(relevant_items)


def reciprocal_rank(
    retrieved: Sequence[Item],
    relevant: Collection[Item],
) -> float:
    """Return ``1 / rank`` of the first relevant result, or 0 if absent."""
    relevant_items = set(relevant)

    if not relevant_items:
        return 0.0

    for rank, item in enumerate(retrieved, start=1):
        if item in relevant_items:
            return 1.0 / rank

    return 0.0


def _validate_grades(relevance: Mapping[Item, int | float]) -> None:
    for item, grade in relevance.items():
        if isinstance(grade, bool) or not isinstance(grade, (int, float)):
            raise ValueError(
                f"Relevance grade for {item!r} must be numeric, got {grade!r}"
            )

        if grade < 0:
            raise ValueError(
                f"Relevance grade for {item!r} cannot be negative"
            )


def dcg_at_k(
    retrieved: Sequence[Item],
    relevance: Mapping[Item, int | float],
    k: int,
) -> float:
    """Return discounted cumulative gain at ``k`` using graded relevance."""
    _validate_k(k)
    _validate_grades(relevance)

    return sum(
        (2 ** relevance.get(item, 0) - 1) / log2(rank + 1)
        for rank, item in enumerate(retrieved[:k], start=1)
    )


def ndcg_at_k(
    retrieved: Sequence[Item],
    relevance: Mapping[Item, int | float],
    k: int,
) -> float:
    """Return normalized discounted cumulative gain at ``k``."""
    _validate_k(k)
    _validate_grades(relevance)

    if not relevance:
        return 0.0

    ideal_grades = sorted(relevance.values(), reverse=True)[:k]
    ideal_dcg = sum(
        (2 ** grade - 1) / log2(rank + 1)
        for rank, grade in enumerate(ideal_grades, start=1)
    )

    if ideal_dcg == 0:
        return 0.0

    return dcg_at_k(retrieved, relevance, k) / ideal_dcg
