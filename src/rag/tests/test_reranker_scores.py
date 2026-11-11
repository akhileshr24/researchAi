
from src.rag.reranker import rerank_hybrid
from src.rag.tests.evaluation_dataset import evaluation_dataset


for case in evaluation_dataset:
    question = case["question"]

    # Retrieve and rerank the full candidate pool
    results = rerank_hybrid(
        question,
        top_k=10,
        candidate_k=10,
    )

    print("\n" + "=" * 65)
    print("QUESTION:", question)

    if not results:
        print("No candidates retrieved")
        continue

    if not case["answerable"]:
        print("TYPE: UNANSWERABLE")
        print(
            "Top score:",
            round(results[0]["rerank_score"], 3)
        )
        print(
            "All scores:",
            [round(r["rerank_score"], 3) for r in results]
        )
        continue

    relevant_ids = set(case["relevance_grades"].keys())

    relevant_scores = []
    irrelevant_scores = []

    for result in results:
        chunk_id = (result["page"], result["chunk_index"])
        score = result["rerank_score"]

        if chunk_id in relevant_ids:
            relevant_scores.append(score)
        else:
            irrelevant_scores.append(score)

    print("TYPE: ANSWERABLE")
    print(
        "Relevant scores:",
        [round(s, 3) for s in relevant_scores]
    )
    print(
        "Irrelevant scores:",
        [round(s, 3) for s in irrelevant_scores]
    )
    print("Top score:", round(results[0]["rerank_score"], 3))
