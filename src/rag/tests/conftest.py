"""Shared fixtures and helpers for the RAG test suite."""

import re
from pathlib import Path
from uuid import uuid4

import chromadb
import pytest

from src.rag import vectorstore


def _escape_pdf_text(text: str) -> str:
    return (
        text.replace("\\", r"\\")
        .replace("(", r"\(")
        .replace(")", r"\)")
    )


def make_pdf(path: Path, pages: list[str] | list[list[str]]) -> Path:
    """Write a small text PDF without extra dependencies.

    ``pages`` may be a list of page strings or a list of line lists.
    """
    if pages and isinstance(pages[0], str):
        pages = [[page] for page in pages]

    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        (
            "<< /Type /Pages /Kids "
            f"[{' '.join(f'{3 + 3 * index} 0 R' for index in range(len(pages)))}] "
            f"/Count {len(pages)} >>"
        ),
    ]

    for index, lines in enumerate(pages):
        base = 3 + 3 * index
        commands = ["BT /F1 12 Tf 72 720 Td"]
        for line in lines:
            commands.append(f"({_escape_pdf_text(line)}) Tj 0 -18 Td")
        commands.append("ET")
        content = " ".join(commands)

        objects.append(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {base + 1} 0 R "
            f"/Resources << /Font << /F1 {base + 2} 0 R >> >> >>"
        )
        objects.append(
            f"<< /Length {len(content.encode('latin-1'))} >>\n"
            f"stream\n{content}\nendstream"
        )
        objects.append(
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
        )

    output = "%PDF-1.4\n"
    offsets = []

    for number, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output += f"{number} 0 obj\n{body}\nendobj\n"

    xref_offset = len(output)
    output += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"

    for offset in offsets:
        output += f"{offset:010d} 00000 n \n"

    output += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    )

    path.write_bytes(output.encode("latin-1"))
    return path


class FakeEmbeddings:
    """Cheap deterministic embeddings for unit tests (no model download)."""

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * 64
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            vector[sum(ord(character) for character in token) % 64] += 1.0
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture
def pdf_factory(tmp_path):
    def create(name: str = "document.pdf", pages=None) -> Path:
        pages = pages or [["This is a test PDF page."]]
        return make_pdf(tmp_path / name, pages)

    return create


@pytest.fixture
def temp_store(monkeypatch):
    """Replace ChromaDB and the embedding model with in-memory test doubles."""
    client = chromadb.EphemeralClient()
    collection_name = f"test_chunks_{uuid4().hex}"
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    embeddings = FakeEmbeddings()

    monkeypatch.setattr(vectorstore, "_get_collection", lambda: collection)
    monkeypatch.setattr(vectorstore, "_get_embeddings", lambda: embeddings)

    yield collection

    client.delete_collection(collection_name)


@pytest.fixture
def chunk_factory():
    def create(
        text: str = "sample chunk text",
        document_id: str = "doc-a",
        filename: str = "a.pdf",
        page: int = 1,
        chunk_index: int = 0,
        start_index: int = 0,
        total_pages: int | None = None,
    ) -> dict:
        return {
            "text": text,
            "document_id": document_id,
            "filename": filename,
            "page": page,
            "chunk_index": chunk_index,
            "start_index": start_index,
            "total_pages": page if total_pages is None else total_pages,
        }

    return create
