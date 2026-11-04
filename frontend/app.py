import streamlit as st
import requests

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_URL = "http://127.0.0.1:8000"
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
# RAG MODE - PLACEHOLDER
# --------------------------------------------------

elif mode == "RAG":

    st.markdown("### 📄 Document Intelligence")

    st.markdown(
        '<p class="mode-description">'
        'Upload PDF documents and ask questions about their contents.'
        '</p>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("#### Upload your PDF")

    st.file_uploader(
        "Choose a PDF document",
        type=["pdf"],
        disabled=True,
        help="PDF processing will be enabled in the next development phase."
    )

    st.info(
        "🚧 RAG functionality is coming soon. "
        "PDF processing and document question answering "
        "are not available yet."
    )

    st.markdown("#### Planned features")

    features = [
        "PDF document upload",
        "Text extraction and chunking",
        "Hugging Face embeddings",
        "ChromaDB vector storage",
        "Semantic document retrieval",
        "AI answers with document references"
    ]

    for feature in features:
        st.markdown(f"- {feature}")


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