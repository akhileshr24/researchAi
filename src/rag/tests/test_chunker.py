"""Tests for text chunking."""

from src.rag.chunker import chunk_pages


def test_chunk_pages_preserves_metadata():
    pages = [
        {"page": 1, "text": "Alpha beta gamma."},
        {"page": 2, "text": "Delta epsilon zeta."},
    ]

    chunks = chunk_pages(pages, "guide.pdf", "doc-1")

    assert len(chunks) == 2
    assert chunks[0]["page"] == 1
    assert chunks[1]["page"] == 2
    assert [chunk["chunk_index"] for chunk in chunks] == [0, 1]
    assert all(chunk["filename"] == "guide.pdf" for chunk in chunks)
    assert all(chunk["document_id"] == "doc-1" for chunk in chunks)
    assert all(isinstance(chunk["start_index"], int) for chunk in chunks)


def test_chunk_pages_skips_blank_pages():
    pages = [
        {"page": 1, "text": "   "},
        {"page": 2, "text": "Useful content."},
    ]

    chunks = chunk_pages(pages, "guide.pdf", "doc-1")

    assert len(chunks) == 1
    assert chunks[0]["page"] == 2
    assert chunks[0]["text"] == "Useful content."


def test_chunk_pages_splits_long_text_with_overlap():
    pages = [{"page": 1, "text": "word " * 400}]

    chunks = chunk_pages(
        pages,
        "long.pdf",
        "doc-2",
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) > 1
    assert all(len(chunk["text"]) <= 100 for chunk in chunks)
    assert [chunk["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk["text"].strip() for chunk in chunks)


def test_chunk_pages_returns_empty_for_no_pages():
    assert chunk_pages([], "empty.pdf", "doc-3") == []
