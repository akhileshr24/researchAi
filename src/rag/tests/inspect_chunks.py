
from src.rag.vectorstore import get_all_chunks


chunks = get_all_chunks()

print(f"\nTotal chunks: {len(chunks)}")

for chunk in chunks:
    print("\n" + "=" * 80)

    print(
        f"Filename: {chunk['filename']} | "
        f"Page: {chunk['page']} | "
        f"Chunk index: {chunk['chunk_index']}"
    )

    print("-" * 80)
    print(chunk["text"])
