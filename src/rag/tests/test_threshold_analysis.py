
from src.rag.reranker import rerank_hybrid
from src.rag.tests.evaluation_dataset import evaluation_dataset

scores = []

for case in evaluation_dataset:
    results = rerank_hybrid(
        case["question"],
        top_k=10,
        candidate_k=10,
    )

    top_score = results[0]["rerank_score"] if results else float("-inf")

    scores.append({
        "question": case["question"],
        "answerable": case["answerable"],
        "score": top_score,
    })


thresholds = [-10, -8, -6, -5, -4, -2, -1, 0, 1, 2, 4, 6]

print("\n===== THRESHOLD ANALYSIS =====")

for threshold in thresholds:
    accepted_answerable = sum(
        item["answerable"] and item["score"] >= threshold
        for item in scores
    )

    rejected_unanswerable = sum(
        not item["answerable"] and item["score"] < threshold
        for item in scores
    )

    false_rejections = sum(
        item["answerable"] and item["score"] < threshold
        for item in scores
    )

    false_acceptances = sum(
        not item["answerable"] and item["score"] >= threshold
        for item in scores
    )

    print(f"\nThreshold: {threshold}")
    print(f"Accepted answerable: {accepted_answerable}/13")
    print(f"Rejected unanswerable: {rejected_unanswerable}/3")
    print(f"False rejections: {false_rejections}")
    print(f"False acceptances: {false_acceptances}")
