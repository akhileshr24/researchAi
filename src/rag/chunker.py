"""Split page text into overlapping chunks for retrieval."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import CHUNK_OVERLAP, CHUNK_SIZE


def chunk_pages(
    pages: list[dict],
    filename: str,
    document_id: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Create chunks that keep document, page and position metadata."""
    if not pages:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    chunks = []

    for page in pages:
        for document in splitter.create_documents([page["text"]]):
            text = document.page_content.strip()

            if not text:
                continue

            chunks.append({
                "text": text,
                "page": page["page"],
                "filename": filename,
                "document_id": document_id,
                "chunk_index": len(chunks),
                "start_index": document.metadata.get("start_index", 0),
                "total_pages": len(pages),
            })

    return chunks
