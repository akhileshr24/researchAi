
from src.rag.reranker import rerank_hybrid
from src.rag.tests.evaluation_dataset import evaluation_dataset

from src.rag.tests.retrieval_evaluator import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
)

TOP_K = 4


def get_chunk_ids(results):
    return [
        (result["page"], result["chunk_index"])
        for result in results
    ]


totals = {
    "precision": 0,
    "recall": 0,
    "mrr": 0,
    "ndcg": 0,
}

answerable_count = 0
correct_rejections = 0
unanswerable_count = 0

for case in evaluation_dataset:
    question = case["question"]

    results = rerank_hybrid(
        question,
        top_k=TOP_K,
        candidate_k=10,
    )

    retrieved_ids = get_chunk_ids(results)

    print("\nQUESTION:", question)

    if not case["answerable"]:
        unanswerable_count += 1

        if not results:
            correct_rejections += 1

        print("Retrieved chunks:", len(results))
        continue

    grades = case["relevance_grades"]
    relevant_ids = set(grades.keys())

    metrics = {
        "precision": precision_at_k(
            retrieved_ids, relevant_ids, TOP_K
        ),
        "recall": recall_at_k(
            retrieved_ids, relevant_ids, TOP_K
        ),
        "mrr": reciprocal_rank(
            retrieved_ids, relevant_ids
        ),
        "ndcg": ndcg_at_k(
            retrieved_ids, grades, TOP_K
        ),
    }

    for metric, value in metrics.items():
        totals[metric] += value

    answerable_count += 1

    for rank, result in enumerate(results, start=1):
        print(
            f"Rank {rank} | "
            f"Page {result['page']} | "
            f"Score {result['rerank_score']:.3f}"
        )


print("\n===== RERANKER EVALUATION =====")

for metric, total in totals.items():
    print(
        f"{metric}: "
        f"{total / answerable_count:.3f}"
    )

print("\n===== NO-ANSWER EVALUATION =====")
print("Unanswerable:", unanswerable_count)
print("Correct rejections:", correct_rejections)
print(
    "False-positive rate:",
    (unanswerable_count - correct_rejections)
    / unanswerable_count
)
