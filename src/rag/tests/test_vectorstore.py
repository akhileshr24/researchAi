
from src.rag.loader import load_pdf
from src.rag.chunker import chunk_pages
from src.rag.vectorstore import add_chunks, search

PDF_PATH = "docs/assets/test.pdf"

# Load PDF
pages = load_pdf(PDF_PATH)

# Create chunks
chunks = chunk_pages(pages, PDF_PATH)

# Store in ChromaDB
add_chunks(chunks)

print("Chunks stored successfully!")

# Test different questions
questions = [
    "What is machine learning?",
    "What is deep learning?",
    "What is quantum computing?"
]

for question in questions:
    print(f"\nQUESTION: {question}")

    results = search(question)

    for result in results:
        print("\nPage:", result["page"])
        print("Chunk:", result["chunk_index"])
        print("Score:", result["score"])
        print("Text:", result["text"][:200])
