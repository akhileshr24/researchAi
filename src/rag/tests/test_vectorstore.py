"""Tests for ChromaDB storage, search and document management."""

import pytest

from src.rag.vectorstore import (
    add_chunks,
    delete_document,
    get_all_chunks,
    get_document,
    list_documents,
    search,
)


def test_add_and_search_chunks(temp_store, chunk_factory):
    add_chunks([
        chunk_factory(
            text="Vector search uses embeddings to find similar text.",
            chunk_index=0,
        ),
        chunk_factory(
            text="BM25 ranks keyword matches with term frequencies.",
            chunk_index=1,
        ),
    ])

    results = search("vector search")

    assert results
    assert "Vector search" in results[0]["text"]
    assert results[0]["filename"] == "a.pdf"
    assert results[0]["page"] == 1
    assert results[0]["document_id"] == "doc-a"
    assert "score" in results[0]


def test_search_returns_empty_for_empty_store(temp_store):
    assert search("anything") == []


def test_search_returns_empty_for_empty_query(temp_store, chunk_factory):
    add_chunks([chunk_factory()])

    assert search("   ") == []


def test_search_filters_by_document(temp_store, chunk_factory):
    add_chunks([
        chunk_factory(text="vector search in document a", chunk_index=0),
    ])
    add_chunks([
        chunk_factory(
            text="vector search in document b",
            document_id="doc-b",
            filename="b.pdf",
            chunk_index=0,
        ),
    ])

    results = search("vector search", document_id="doc-b")

    assert results
    assert all(result["document_id"] == "doc-b" for result in results)


def test_reingesting_a_document_replaces_old_chunks(temp_store, chunk_factory):
    add_chunks([
        chunk_factory(text="first version", chunk_index=0),
        chunk_factory(text="first version second chunk", chunk_index=1),
    ])

    add_chunks([
        chunk_factory(text="replacement version", chunk_index=0),
    ])

    assert len(get_all_chunks()) == 1
    assert list_documents()[0]["total_chunks"] == 1


def test_list_documents_reports_counts(temp_store, chunk_factory):
    add_chunks([
        chunk_factory(text="page one", chunk_index=0, page=1),
        chunk_factory(text="page two", chunk_index=1, page=2),
    ])
    add_chunks([
        chunk_factory(
            text="other document",
            document_id="doc-b",
            filename="b.pdf",
            chunk_index=0,
        ),
    ])

    documents = list_documents()

    assert [document["filename"] for document in documents] == [
        "a.pdf",
        "b.pdf",
    ]
    assert documents[0]["total_chunks"] == 2
    assert documents[0]["total_pages"] == 2


def test_get_document_summary(temp_store, chunk_factory):
    add_chunks([
        chunk_factory(text="page one", chunk_index=0, page=1),
        chunk_factory(text="page two", chunk_index=1, page=2),
    ])

    document = get_document("doc-a")

    assert document == {
        "document_id": "doc-a",
        "filename": "a.pdf",
        "total_chunks": 2,
        "total_pages": 2,
    }
    assert get_document("missing") is None


def test_list_and_get_document_use_stored_total_pages(
    temp_store,
    chunk_factory,
):
    add_chunks([
        chunk_factory(
            text="Only text page",
            chunk_index=0,
            page=1,
            total_pages=2,
        ),
    ])

    assert list_documents()[0]["total_pages"] == 2
    assert get_document("doc-a")["total_pages"] == 2


def test_failed_upsert_keeps_existing_document(
    temp_store,
    chunk_factory,
    monkeypatch,
):
    add_chunks([chunk_factory(text="original content", chunk_index=0)])

    class FailingUpsert:
        def __init__(self, collection):
            self._collection = collection

        def __getattr__(self, name):
            return getattr(self._collection, name)

        def upsert(self, *args, **kwargs):
            raise RuntimeError("upsert failed")

    monkeypatch.setattr(
        "src.rag.vectorstore._get_collection",
        lambda: FailingUpsert(temp_store),
    )

    with pytest.raises(RuntimeError, match="upsert failed"):
        add_chunks([chunk_factory(text="new content", chunk_index=0)])

    remaining = get_all_chunks()

    assert len(remaining) == 1
    assert remaining[0]["text"] == "original content"


def test_delete_document(temp_store, chunk_factory):
    add_chunks([chunk_factory()])

    assert delete_document("doc-a") is True
    assert delete_document("doc-a") is False
    assert get_all_chunks() == []


def test_add_chunks_rejects_empty_and_mixed_documents(
    temp_store,
    chunk_factory,
):
    with pytest.raises(ValueError, match="empty list"):
        add_chunks([])

    with pytest.raises(ValueError, match="same document"):
        add_chunks([
            chunk_factory(document_id="doc-a", chunk_index=0),
            chunk_factory(document_id="doc-b", chunk_index=0),
        ])
