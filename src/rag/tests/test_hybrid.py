from src.rag.hybrid_retriever import search_hybrid

questions = [
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "quantum computing"
]

for question in questions:
    print(f"\nQuestion: {question}")

    results = search_hybrid(question)

    for result in results:
        print(
            f"RRF Score: {result['rrf_score']:.5f} | "
            f"Page: {result['page']}"
        )
        print("Text:", result["text"][:150])
