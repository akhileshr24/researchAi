"""Ingest PDFs into the persistent vector store."""

import hashlib
from pathlib import Path

from src.rag.chunker import chunk_pages
from src.rag.loader import load_pdf
from src.rag.vectorstore import (
    add_chunks,
    delete_document,
    get_document,
    list_documents,
)


def ingest_pdf(pdf_path: str | Path, replace: bool = False) -> dict:
    """Validate, chunk, embed and store one PDF.

    Document IDs are content hashes, so ingesting the same file twice is a
    no-op. With ``replace=True`` a document is re-indexed and any older
    document with the same filename is removed.
    """
    path = Path(pdf_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    document_id = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    existing = get_document(document_id)

    if existing and not replace:
        return {
            "status": "skipped",
            "message": "Document already ingested",
            **existing,
        }

    pages = load_pdf(path)

    if not any(page["text"] for page in pages):
        raise ValueError(f"No extractable text found in '{path.name}'")

    chunks = chunk_pages(pages, path.name, document_id)

    if not chunks:
        raise ValueError(f"No text chunks created for '{path.name}'")

    total_chunks = add_chunks(chunks)

    if replace:
        for document in list_documents():
            if (
                document["filename"] == path.name
                and document["document_id"] != document_id
            ):
                delete_document(document["document_id"])

    return {
        "status": "replaced" if existing else "ingested",
        "document_id": document_id,
        "filename": path.name,
        "total_pages": len(pages),
        "total_chunks": total_chunks,
    }
