
from src.rag.reranker import rerank_hybrid

questions = [
    "What is quantum computing?",
    "What is machine learning?",
    "What is the capital of Japan?",
]

for question in questions:
    print("\nQUESTION:", question)

    results = rerank_hybrid(question)

    for rank, result in enumerate(results, start=1):
        print(
            f"\nRank: {rank}"
            f" | Score: {result['rerank_score']:.4f}"
            f" | Page: {result['page']}"
        )
        print("Text:", result["text"][:200])
