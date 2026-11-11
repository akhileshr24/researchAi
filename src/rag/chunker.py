
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_pages(pages, filename):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True
    )

    chunks = []

    for page in pages:
        documents = splitter.create_documents([page["text"]])

        for index, document in enumerate(documents):
            chunks.append({
                "text": document.page_content,
                "page": page["page"],
                "filename": filename,
                "chunk_index": index,
                "start_index": document.metadata["start_index"]
            })

    return chunks
