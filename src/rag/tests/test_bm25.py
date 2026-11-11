
from src.rag.bm25_retriever import search_bm25

questions = [
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "quantum computing"
]

for question in questions:
    print(f"\nQuestion: {question}")

    results = search_bm25(question)

    for result in results:
        print(
            f"Score: {result['score']} | "
            f"Page: {result['page']}"
        )
        print("Text:", result["text"][:150])
