
from src.rag.vectorstore import search
from src.rag.bm25_retriever import search_bm25
from src.rag.hybrid_retriever import search_hybrid
from src.config.settings import TOP_K

questions = [
    "What is quantum computing?",
    "Who won the FIFA World Cup in 2022?",
    "What is the capital of Japan?",
]


def display_results(name, results, score_key):
    print(f"\n--- {name} ---")

    if not results:
        print("No results")
        return

    for rank, result in enumerate(results, start=1):
        print(
            f"\nRank: {rank} | "
            f"Page: {result['page']} | "
            f"Chunk: {result['chunk_index']} | "
            f"Score: {result[score_key]}"
        )
        print("Text:", result["text"][:250].replace("\n", " "))


for question in questions:
    print("\n" + "=" * 70)
    print("QUESTION:", question)
    print("=" * 70)

    vector_results = search(question, top_k=TOP_K)

    print("\nRaw Vector Results:")
    print(vector_results)
    display_results("Vector Search", vector_results, "score")
    display_results("BM25 Search", search_bm25(question), "score")
    display_results("Hybrid Search", search_hybrid(question), "rrf_score")
