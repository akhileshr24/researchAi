import os

import streamlit as st
import requests

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
GITHUB_URL = "https://github.com/akhileshr24"

st.set_page_config(
    page_title="ResearchAI",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# STYLING
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background: #F7F9FC;
    }

    .block-container {
        max-width: 1100px;
        padding: 2rem 2rem 4rem;
    }

    .section-title {
        color: #172033;
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .stTextInput input {
        background: white;
        border: 1px solid #DCE3ED;
        border-radius: 9px;
    }

    div.stButton > button {
        background: #2563EB;
        color: white;
        border: none;
        border-radius: 9px;
        height: 44px;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background: #1D4ED8;
        color: white;
    }

    .mode-description {
        color: #64748B;
        font-size: 14px;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🔎 ResearchAI")
st.caption("Your intelligent research and chat workspace")
st.divider()


# --------------------------------------------------
# MODE SELECTOR
# --------------------------------------------------

mode = st.segmented_control(
    "Select Mode",
    options=["Research", "Chat", "RAG"],
    default="Research",
    key="app_mode"
)


# --------------------------------------------------
# API ERROR HANDLING
# --------------------------------------------------

def handle_api_error(response):
    if response.status_code == 429:
        st.error("Rate limit exceeded. Please wait and try again.")

    elif response.status_code == 422:
        st.error("Invalid request. Please check your input.")

    elif response.status_code >= 500:
        st.error("Service temporarily unavailable. Please try again.")

    else:
        st.error(f"Request failed with status code {response.status_code}.")


# --------------------------------------------------
# RAG API HELPERS
# --------------------------------------------------

RAG_TIMEOUTS = {
    "upload": 600,  # ingestion chunks and embeds the whole PDF
    "query": 300,   # retrieval, reranking and LLM answer generation
    "manage": 60,   # document listing and deletion
}


def rag_api_call(method, endpoint, timeout, **kwargs):
    """Call a RAG endpoint, returning the response or None after an error."""
    try:
        return requests.request(
            method,
            f"{BASE_URL}{endpoint}",
            timeout=timeout,
            **kwargs
        )

    except requests.Timeout:
        st.error("The request timed out. Please try again.")

    except requests.ConnectionError:
        st.error("Cannot connect to the backend. Start FastAPI first.")

    except requests.RequestException:
        st.error("Unable to complete the request.")

    return None


def load_indexed_documents():
    """Return the indexed documents, or an empty list on failure."""
    response = rag_api_call(
        "GET",
        "/rag/documents",
        RAG_TIMEOUTS["manage"]
    )

    if response is None:
        return []

    if response.ok:
        return response.json()["documents"]

    handle_api_error(response)
    return []


def format_document_label(document):
    return (
        f"{document['filename']} "
        f"({document['total_pages']} pages, "
        f"{document['total_chunks']} chunks)"
    )


# --------------------------------------------------
# RESEARCH MODE
# --------------------------------------------------

if mode == "Research":

    st.markdown(
        '<div class="section-title">What would you like to research?</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="mode-description">'
        'Search the web and generate structured research reports.'
        '</p>',
        unsafe_allow_html=True
    )

    query = st.text_input(
        "Research topic",
        placeholder="e.g. Impact of AI agents on software engineering",
        max_chars=500,
        label_visibility="collapsed"
    )

    if st.button("🔍 Start Research", use_container_width=True):

        if len(query.strip()) < 3:
            st.warning("Enter at least 3 characters.")

        else:
            try:
                with st.spinner("Researching the web..."):

                    response = requests.post(
                        f"{BASE_URL}/research",
                        json={"query": query.strip()},
                        timeout=300
                    )

                if response.ok:
                    result = response.json()

                    st.session_state["report"] = result["report"]
                    st.session_state["sources"] = result["sources"]
                    st.session_state["query"] = result["query"]

                else:
                    handle_api_error(response)

            except requests.Timeout:
                st.error("Research request timed out.")

            except requests.ConnectionError:
                st.error("Cannot connect to the backend. Start FastAPI first.")

            except requests.RequestException:
                st.error("Unable to complete research.")

    # DISPLAY REPORT

    if "report" in st.session_state:

        st.divider()

        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown("### Research Report")
            st.caption(f"Topic: {st.session_state['query']}")

        with col2:
            st.metric(
                "Sources found",
                len(st.session_state["sources"])
            )

        with st.container(border=True):
            st.markdown(st.session_state["report"])

        # DOWNLOAD REPORT

        markdown_report = (
            f"# Research Report\n\n"
            f"**Topic:** {st.session_state['query']}\n\n"
            f"{st.session_state['report']}\n\n"
            f"## Sources\n\n"
        )

        for source in st.session_state["sources"]:
            markdown_report += (
                f"- [{source['title']}]({source['url']})\n"
            )

        st.download_button(
            "⬇ Download Markdown Report",
            data=markdown_report,
            file_name="research_report.md",
            mime="text/markdown"
        )

        # SOURCE LINKS

        st.divider()
        st.markdown("### Research Sources")

        for index, source in enumerate(
            st.session_state["sources"],
            start=1
        ):
            with st.container(border=True):
                st.markdown(
                    f"**{index}. [{source['title']}]({source['url']})**"
                )


# --------------------------------------------------
# CHAT MODE
# --------------------------------------------------

elif mode == "Chat":

    st.markdown("### AI Chat")
    st.markdown(
        '<p class="mode-description">'
        'Ask questions and continue your conversation with ResearchAI.'
        '</p>',
        unsafe_allow_html=True
    )

    # INITIALIZE CHAT HISTORY

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # CLEAR CHAT

    if st.button("🗑 Clear Chat"):
        st.session_state["chat_history"] = []
        st.rerun()

    # DISPLAY PREVIOUS MESSAGES

    for message in st.session_state["chat_history"]:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # CHAT INPUT

    user_message = st.chat_input(
        "Ask ResearchAI anything...",
        max_chars=1000
    )

    if user_message:

        user_message = user_message.strip()

        if not user_message:
            st.warning("Please enter a message.")

        else:

            # SEND PREVIOUS CONVERSATION CONTEXT

            history = st.session_state["chat_history"][-6:]

            try:
                with st.spinner("Thinking..."):

                    response = requests.post(
                        f"{BASE_URL}/chat",
                        json={
                            "message": user_message,
                            "history": history
                        },
                        timeout=90
                    )

                if response.ok:

                    result = response.json()
                    answer = result["response"]

                    # SAVE CONVERSATION

                    st.session_state["chat_history"].append({
                        "role": "user",
                        "content": user_message
                    })

                    st.session_state["chat_history"].append({
                        "role": "assistant",
                        "content": answer
                    })

                    st.rerun()

                else:
                    handle_api_error(response)

            except requests.Timeout:
                st.error("Chat request timed out.")

            except requests.ConnectionError:
                st.error("Cannot connect to the backend.")

            except requests.RequestException:
                st.error("Unable to complete chat request.")


# --------------------------------------------------
# RAG MODE - PDF RAG ASSISTANT
# --------------------------------------------------

elif mode == "RAG":

    st.markdown(
        '<div class="section-title">PDF RAG Assistant</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="mode-description">'
        'Upload PDF documents, manage the index and ask questions '
        'about their contents.'
        '</p>',
        unsafe_allow_html=True
    )

    # DOCUMENT MANAGEMENT

    st.markdown("#### 📚 Indexed Documents")

    documents = load_indexed_documents()

    if documents:

        for document in documents:
            with st.container(border=True):
                col1, col2 = st.columns([3, 2])

                with col1:
                    st.markdown(f"**📄 {document['filename']}**")
                    st.caption(f"ID: {document['document_id']}")

                with col2:
                    st.caption(
                        f"{document['total_pages']} pages · "
                        f"{document['total_chunks']} chunks"
                    )

        # DELETE DOCUMENT

        delete_options = {
            f"{document['filename']} · {document['document_id'][:8]}":
                document["document_id"]
            for document in documents
        }

        selected_label = st.selectbox(
            "Document to delete",
            list(delete_options.keys()),
            key="rag_delete_selection"
        )

        if st.button("🗑 Delete Selected Document"):
            with st.spinner("Deleting document..."):
                response = rag_api_call(
                    "DELETE",
                    f"/rag/documents/{delete_options[selected_label]}",
                    RAG_TIMEOUTS["manage"]
                )

            if response is not None:
                if response.ok:
                    st.success(f"Deleted: {selected_label}")
                    st.rerun()
                else:
                    handle_api_error(response)

    else:
        st.info(
            "No documents indexed yet. Upload a PDF below to get started."
        )

    st.divider()

    # PDF UPLOAD

    st.markdown("#### ⬆ Upload a PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF document",
        type=["pdf"],
        help="The PDF is chunked, embedded and stored in the vector index."
    )

    if uploaded_file is not None:
        st.caption(
            f"Selected: {uploaded_file.name} "
            f"({uploaded_file.size / 1024:.0f} KB)"
        )

        if st.button("⬆ Upload and Index", use_container_width=True):
            with st.spinner("Uploading and indexing PDF..."):
                response = rag_api_call(
                    "POST",
                    "/rag/upload",
                    RAG_TIMEOUTS["upload"],
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf"
                        )
                    }
                )

            if response is not None:
                if response.ok:
                    result = response.json()

                    if result["status"] == "skipped":
                        st.info(
                            f"ℹ {result['message']}. "
                            f"{result['filename']} is already indexed."
                        )
                    else:
                        st.success(
                            f"✅ Indexed {result['filename']} "
                            f"({result['total_pages']} pages, "
                            f"{result['total_chunks']} chunks)."
                        )

                    st.rerun()
                else:
                    handle_api_error(response)

    st.divider()

    # QUESTION ANSWERING

    st.markdown("#### 💬 Ask a Question")

    scope_options = ["All indexed documents"] + [
        format_document_label(document) for document in documents
    ]

    scope = st.selectbox(
        "Query scope",
        scope_options,
        key="rag_query_scope"
    )

    if scope == "All indexed documents":
        document_id = None
    else:
        document_id = next(
            document["document_id"]
            for document in documents
            if format_document_label(document) == scope
        )

    question = st.text_input(
        "Your question",
        placeholder="e.g. What is retrieval-augmented generation?",
        max_chars=1000,
        label_visibility="collapsed"
    )

    if st.button("🔍 Ask Question", use_container_width=True):

        if not question.strip():
            st.warning("Enter a question first.")

        else:
            payload = {"question": question.strip()}

            if document_id is not None:
                payload["document_id"] = document_id

            with st.spinner("Searching documents and generating answer..."):
                response = rag_api_call(
                    "POST",
                    "/rag/query",
                    RAG_TIMEOUTS["query"],
                    json=payload
                )

            if response is not None:
                if response.ok:
                    st.session_state["rag_result"] = response.json()
                else:
                    handle_api_error(response)

    # ANSWER DISPLAY

    if "rag_result" in st.session_state:

        result = st.session_state["rag_result"]

        st.divider()

        st.markdown("### Answer")

        with st.container(border=True):
            st.markdown(result["answer"])

        # SOURCE CITATIONS

        st.markdown("### Source Citations")

        if result["sources"]:

            for index, source in enumerate(result["sources"], start=1):
                with st.container(border=True):
                    st.markdown(
                        f"**{index}. {source['filename']}** — "
                        f"page {source['page']}"
                    )
                    st.caption(
                        f"Chunk index: {source.get('chunk_index', 'n/a')}"
                    )

        else:
            st.caption(
                "No sources found for this question in the indexed documents."
            )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    f"""
    <div style="
        text-align: center;
        color: #94A3B8;
        font-size: 13px;
        padding: 15px 0;
    ">
        Built by
        <a href="{GITHUB_URL}"
           target="_blank"
           style="
               color: #2563EB;
               text-decoration: none;
               font-weight: 600;
           ">
           Akhilesh R
        </a>
        <br>
        AI Research Assistant · Research · RAG · Chat
    </div>
    """,
    unsafe_allow_html=True
)