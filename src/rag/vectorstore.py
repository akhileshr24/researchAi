"""Persistent ChromaDB storage and vector search."""

from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb import Collection
from langchain_huggingface import HuggingFaceEmbeddings

from src.config.settings import (
    CHROMA_DIR,
    EMBEDDING_MODEL,
    MIN_RELEVANCE_SCORE,
    TOP_K,
)

COLLECTION_NAME = "rag_chunks"


@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _get_collection() -> Collection:
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _to_chunk(text: str, metadata: dict, score: float | None = None) -> dict:
    """Convert ChromaDB output into the metadata shape used by retrievers."""
    chunk = {
        "text": text,
        "document_id": metadata["document_id"],
        "filename": metadata["filename"],
        "page": metadata["page"],
        "chunk_index": metadata["chunk_index"],
        "start_index": metadata["start_index"],
        "total_pages": metadata.get("total_pages", metadata["page"]),
    }

    if score is not None:
        chunk["score"] = score

    return chunk


def add_chunks(chunks: list[dict]) -> int:
    """Store one document's chunks, replacing its previous chunks if any.

    Embeddings are computed and new chunks are upserted before stale chunks
    are deleted, so a failed embedding or upsert keeps the previous document.
    """
    if not chunks:
        raise ValueError("Cannot add an empty list of chunks")

    document_ids = {chunk["document_id"] for chunk in chunks}

    if len(document_ids) != 1:
        raise ValueError("All chunks must belong to the same document")

    document_id = document_ids.pop()
    collection = _get_collection()

    texts = [chunk["text"] for chunk in chunks]
    embeddings = _get_embeddings().embed_documents(texts)

    new_ids = [
        f"{document_id}-{chunk['chunk_index']}"
        for chunk in chunks
    ]
    old_ids = set(
        collection.get(
            where={"document_id": document_id},
            include=[],
        )["ids"]
    )

    collection.upsert(
        ids=new_ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {
                "document_id": document_id,
                "filename": chunk["filename"],
                "page": chunk["page"],
                "chunk_index": chunk["chunk_index"],
                "start_index": chunk["start_index"],
                "total_pages": chunk.get("total_pages", chunk["page"]),
            }
            for chunk in chunks
        ],
    )

    stale_ids = old_ids - set(new_ids)

    if stale_ids:
        collection.delete(ids=sorted(stale_ids))

    return len(chunks)


def search(
    query: str,
    top_k: int = TOP_K,
    document_id: str | None = None,
    min_score: float = MIN_RELEVANCE_SCORE,
) -> list[dict]:
    """Return the most similar chunks, optionally restricted to one document."""
    query = query.strip()

    if not query or top_k <= 0:
        return []

    collection = _get_collection()

    if collection.count() == 0:
        return []

    query_args = {
        "query_embeddings": [_get_embeddings().embed_query(query)],
        "n_results": top_k,
    }

    if document_id:
        query_args["where"] = {"document_id": document_id}

    results = collection.query(**query_args)
    documents = results.get("documents") or []

    if not documents or not documents[0]:
        return []

    matches = []

    for text, metadata, distance in zip(
        documents[0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        score = round(1 - distance, 4)

        if score < min_score:
            continue

        matches.append(_to_chunk(text, metadata, score))

    return matches


def get_all_chunks() -> list[dict]:
    """Load every stored chunk. Used to build the BM25 index."""
    collection = _get_collection()

    if collection.count() == 0:
        return []

    results = collection.get(include=["documents", "metadatas"])
    documents = results.get("documents") or []

    return [
        _to_chunk(text, metadata)
        for text, metadata in zip(documents, results["metadatas"])
    ]


def list_documents() -> list[dict]:
    """List ingested documents with their chunk and page counts."""
    documents = {}

    for chunk in get_all_chunks():
        document_id = chunk["document_id"]
        document = documents.setdefault(document_id, {
            "document_id": document_id,
            "filename": chunk["filename"],
            "total_chunks": 0,
            "total_pages": 0,
        })

        document["total_chunks"] += 1
        document["total_pages"] = max(
            document["total_pages"],
            chunk["total_pages"],
        )

    return sorted(
        documents.values(),
        key=lambda document: document["filename"].lower(),
    )


def get_document(document_id: str) -> dict | None:
    """Return one document's summary, or None when it is not stored."""
    collection = _get_collection()
    results = collection.get(
        where={"document_id": document_id},
        include=["metadatas"],
    )

    if not results["ids"]:
        return None

    metadatas = results["metadatas"]

    return {
        "document_id": document_id,
        "filename": metadatas[0]["filename"],
        "total_chunks": len(metadatas),
        "total_pages": max(
            metadata.get("total_pages", metadata["page"])
            for metadata in metadatas
        ),
    }


def delete_document(document_id: str) -> bool:
    """Delete a document. Return False when it does not exist."""
    collection = _get_collection()
    results = collection.get(
        where={"document_id": document_id},
        include=[],
    )

    if not results["ids"]:
        return False

    collection.delete(where={"document_id": document_id})
    return True
