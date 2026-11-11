
from src.rag.vectorstore import search
from src.rag.bm25_retriever import search_bm25
from src.rag.hybrid_retriever import search_hybrid

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


retrievers = {
    "Vector Search": search,
    "BM25 Search": search_bm25,
    "Hybrid Search": search_hybrid,
}


def evaluate_retriever(name, retriever):
    print("\n" + "=" * 60)
    print(f"RETRIEVER: {name}")
    print("=" * 60)

    totals = {
        "precision": 0,
        "recall": 0,
        "mrr": 0,
        "ndcg": 0,
    }

    answerable_count = 0
    unanswerable_count = 0
    correct_rejections = 0

    for case in evaluation_dataset:
        question = case["question"]
        results = retriever(question, top_k=TOP_K)

        retrieved_ids = get_chunk_ids(results)

        if not case["answerable"]:
            unanswerable_count += 1

            if not results:
                correct_rejections += 1

            print(
                f"\nUnanswerable: {question}"
                f"\nRetrieved chunks: {len(results)}"
            )
            continue

        relevance_grades = case["relevance_grades"]
        relevant_ids = set(relevance_grades.keys())

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
                retrieved_ids, relevance_grades, TOP_K
            ),
        }

        for metric, value in metrics.items():
            totals[metric] += value

        answerable_count += 1

    print("\n--- ANSWERABLE METRICS ---")

    for metric, total in totals.items():
        print(
            f"{metric}: "
            f"{total / answerable_count:.3f}"
        )

    false_positives = unanswerable_count - correct_rejections

    print("\n--- NO-ANSWER METRICS ---")
    print(f"Unanswerable questions: {unanswerable_count}")
    print(f"Correct rejections: {correct_rejections}")
    print(
        f"False positives: {false_positives}"
    )
    print(
        f"False-positive rate: "
        f"{false_positives / unanswerable_count:.3f}"
    )


for name, retriever in retrievers.items():
    evaluate_retriever(name, retriever)
