from src.rag.loader import load_pdf
from src.rag.chunker import chunk_pages

PDF_PATH = "docs/assets/test.pdf"

# Step 1: Load PDF
pages = load_pdf(PDF_PATH)

# Step 2: Create chunks
chunks = chunk_pages(pages, PDF_PATH)

# Step 3: Display results
print("Pages:", len(pages))
print("Chunks:", len(chunks))

if chunks:
    print("\nFirst chunk:")
    print(chunks[0]["text"][:300])
print("\nChunk metadata:")

for chunk in chunks[:3]:
    print("\nPage:", chunk["page"])
    print("Chunk index:", chunk["chunk_index"])
    print("Start index:", chunk["start_index"])
    print("Text:", chunk["text"][:150])