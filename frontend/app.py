
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/research"
GITHUB_URL = "https://github.com/akhileshr24"

# ---------- PAGE CONFIG ----------

st.set_page_config(
    page_title="ResearchAI",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------- CUSTOM CSS ----------

st.markdown("""
<style>
    .stApp {
        background: #F7F9FC;
    }

    .block-container {
        max-width: 1100px;
        padding: 2rem 2rem 4rem;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    .header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .brand {
        font-size: 27px;
        font-weight: 750;
        color: #172033;
        letter-spacing: -1px;
    }

    .github-link {
        display: flex;
        align-items: center;
        gap: 8px;
        text-decoration: none !important;
        color: #24292f !important;
        font-weight: 600;
        font-size: 15px;
        padding: 8px 13px;
        border: 1px solid #E2E8F0;
        border-radius: 9px;
        background: white;
    }

    .github-link:hover {
        border-color: #2563EB;
        color: #2563EB !important;
    }

    .subtitle {
        color: #64748B;
        font-size: 14px;
        margin-top: 5px;
    }

    .section-title {
        color: #172033;
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .stTextInput input {
        background: #FFFFFF;
        border: 1px solid #DCE3ED;
        border-radius: 9px;
        padding: 12px;
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
        border: none;
    }

    a {
        color: #2563EB;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ---------- HEADER ----------

st.title("🔎 ResearchAI")
st.caption("Your intelligent web research workspace")

st.divider()


# ---------- RESEARCH INPUT ----------

st.markdown(
    '<div class="section-title">What would you like to research?</div>',
    unsafe_allow_html=True
)

query = st.text_input(
    "Research topic",
    placeholder="e.g. Impact of AI agents on backend engineering by 2030",
    label_visibility="collapsed"
)

research_button = st.button(
    "🔍  Start Research",
    use_container_width=True
)


# ---------- RESEARCH REQUEST ----------

if research_button:

    if not query.strip():
        st.warning("Please enter a research topic.")

    else:
        try:
            with st.spinner(
                "Searching the web and preparing your research report..."
            ):
                response = requests.post(
                    API_URL,
                    json={"query": query.strip()},
                    timeout=300
                )

                response.raise_for_status()
                result = response.json()

                st.session_state["report"] = result["report"]
                st.session_state["sources"] = result["sources"]
                st.session_state["query"] = result["query"]

        except requests.RequestException as error:
            st.error(f"Unable to complete research: {error}")


# ---------- REPORT DISPLAY ----------

if "report" in st.session_state:

    st.divider()

    st.success("Research completed successfully!")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("### Research Report")
        st.caption(f"Topic: {st.session_state['query']}")

    with col2:
        st.markdown(
            f"**{len(st.session_state['sources'])}** sources found"
        )

    with st.container(border=True):
        st.markdown(st.session_state["report"])

    # ---------- MARKDOWN DOWNLOAD ----------

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
        label="⬇ Download Markdown Report",
        data=markdown_report,
        file_name="research_report.md",
        mime="text/markdown"
    )

    # ---------- SOURCES ----------

    st.divider()

    st.markdown("### Research Sources")
    st.caption("Original webpages used during research.")

    for index, source in enumerate(
        st.session_state["sources"], start=1
    ):
        with st.container(border=True):
            st.markdown(
                f"**{index}. [{source['title']}]({source['url']})**"
            )

    st.caption(
        "Verify important research claims against original sources."
    )


# ---------- FOOTER ----------

st.divider()

st.markdown(
    """
    <div style="
        text-align: center;
        color: #94A3B8;
        font-size: 13px;
        padding: 15px 0;
    ">
        Built with ❤️ by
        <a href="https://github.com/akhileshr24"
           target="_blank"
           style="
               color: #2563EB;
               text-decoration: none;
               font-weight: 600;
           ">
           Akhilesh R
        </a>
        <br>
        AI Research Assistant · Powered by LangGraph and FastAPI
    </div>
    """,
    unsafe_allow_html=True
)