
from src.rag.vectorstore import search
from src.rag.bm25_retriever import search_bm25
from src.rag.hybrid_retriever import search_hybrid


questions = [
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "quantum computing"
]


def print_results(method, results):
    print(f"\n--- {method} ---")

    if not results:
        print("No results found.")
        return

    for rank, result in enumerate(results, start=1):
        print(
            f"\nRank: {rank} | "
            f"Page: {result['page']} | "
            f"Score: {result.get('score', result.get('rrf_score', 'N/A'))}"
        )

        print("Text:", result["text"][:150].replace("\n", " "))


for question in questions:
    print("\n" + "=" * 70)
    print(f"QUESTION: {question}")
    print("=" * 70)

    vector_results = search(question)
    bm25_results = search_bm25(question)
    hybrid_results = search_hybrid(question)

    print_results("VECTOR SEARCH", vector_results)
    print_results("BM25 SEARCH", bm25_results)
    print_results("HYBRID SEARCH", hybrid_results)
