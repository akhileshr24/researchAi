
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

from src.config.settings import CHROMA_DIR, EMBEDDING_MODEL, TOP_K, MIN_RELEVANCE_SCORE



# Initialize embedding model and ChromaDB
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

client = chromadb.PersistentClient(path=CHROMA_DIR)

collection = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)


# Store chunks


import hashlib


def add_chunks(chunks):
    if not chunks:
        return

    texts = [chunk["text"] for chunk in chunks]

    # Generate a stable document ID
    filename = chunks[0]["filename"]
    document_id = hashlib.sha256(
        filename.encode("utf-8")
    ).hexdigest()[:16]

    # Generate unique chunk IDs
    ids = [
        f"{document_id}-{chunk['page']}-{chunk['chunk_index']}"
        for chunk in chunks
    ]

    # Store additional metadata
    metadata = [
        {
            "page": chunk["page"],
            "filename": chunk["filename"],
            "document_id": document_id,
            "chunk_index": chunk["chunk_index"],
            "start_index": chunk["start_index"]
        }
        for chunk in chunks
    ]

    vectors = embeddings.embed_documents(texts)

    # Remove old chunks before re-indexing this document
    collection.delete(
        where={"document_id": document_id}
    )

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=vectors,
        metadatas=metadata
    )



# Search relevant chunks


def search(question, top_k=TOP_K):
    query_vector = embeddings.embed_query(question)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k
    )

    matches = []

    if not results["documents"] or not results["documents"][0]:
        return matches

    for i, text in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]
        score = round(1 - results["distances"][0][i], 3)

        # Ignore low-relevance chunks
        if score < MIN_RELEVANCE_SCORE:
            continue

        matches.append({
            "text": text,
            "page": metadata["page"],
            "filename": metadata["filename"],
            "chunk_index": metadata["chunk_index"],
            "start_index": metadata["start_index"],
            "score": score
        })

    return matches

# Get all stored chunks from ChromaDB
def get_all_chunks():
    results = collection.get(
        include=["documents", "metadatas"]
    )

    chunks = []

    for text, metadata in zip(
        results["documents"],
        results["metadatas"]
    ):
        chunks.append({
            "text": text,
            "page": metadata["page"],
            "filename": metadata["filename"],
            "chunk_index": metadata["chunk_index"],
            "start_index": metadata["start_index"]
        })

    return chunks
