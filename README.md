# ResearchAI

**An AI-powered research and conversational assistant built with Python, FastAPI, LangGraph, Groq and Streamlit.**

ResearchAI helps users explore topics through web research, generate structured reports and interact with an AI assistant through a conversational interface.

## Application Preview

### Home
![ResearchAI Home](docs/assets/home.png)

### Research Mode
![ResearchAI Research](docs/assets/research_output.png)

### Chat Mode
![ResearchAI Chat](docs/assets/chat_output.png)

[View Sample Research Report (PDF)](docs/assets/Report.pdf)

## Features

### Web Research
- Web search powered by Tavily.
- Webpage content extraction using Trafilatura.
- AI-generated research reports using Groq.
- Research source links.
- Markdown report download.

### AI Chat
- Conversational AI powered by Groq.
- Context-aware conversation using chat history.
- Clear chat functionality.
- Request validation and error handling.

### Document Intelligence
**Coming Soon**

Planned PDF-based question answering using:
- PDF text extraction.
- Text chunking.
- Hugging Face embeddings.
- ChromaDB vector storage.
- Semantic retrieval.
- Context-based answers with document references.

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Frontend | Streamlit |
| Backend | FastAPI |
| AI Workflow | LangGraph |
| LLM | Groq |
| Web Search | Tavily |
| Web Extraction | Trafilatura |
| Validation | Pydantic |
| Rate Limiting | SlowAPI |
| Package Management | uv |

## Architecture

```text
                  USER
                    |
                    v
               STREAMLIT UI
                    |
                    v
               FASTAPI API
                    |
             +------+------+
             |             |
             v             v
          /chat         /research
             |             |
             v             v
           GROQ         LANGGRAPH
                           |
                           v
                     TAVILY SEARCH
                           |
                           v
                      TRAFILATURA
                           |
                           v
                     EXTRACT CONTENT
                           |
                           v
                         GROQ
                           |
                           v
                   RESEARCH REPORT
                           |
                           v
                      STREAMLIT UI
```

## Project Structure

```text
ResearchAI/
│
├── docs/
│   └── assets/
│       ├── home.png
│       ├── research_output.png
│       ├── chat_output.png
│       └── Report.pdf
│
├── frontend/
│   └── app.py
│
├── src/
│   ├── agent/
│   │   └── graph.py
│   ├── api/
│   │   └── api.py
│   ├── config/
│   │   └── settings.py
│   ├── llm/
│   │   └── providers.py
│   ├── tools/
│   │   ├── search.py
│   │   └── scraper.py
│   └── main.py
│
├── tests/
├── .env.example
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.13 or a version supported by the project configuration.
- [uv](https://docs.astral.sh/uv/)
- [Groq API Key](https://console.groq.com/)
- [Tavily API Key](https://tavily.com/)

### 1. Clone the repository

```bash
git clone https://github.com/akhileshr24/researchAi.git
cd researchAi
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Configure environment variables

Create a `.env` file in the project root using `.env.example` as a reference.

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b

TAVILY_API_KEY=your_tavily_api_key

LANGCHAIN_TRACING_V2=false
LANGSMITH_TRACING=false
```

**Important:** Never commit your `.env` file or expose API keys.

### 4. Start the backend

```bash
uv run uvicorn src.main:app --reload
```

Backend URL:

`http://127.0.0.1:8000`

Interactive API documentation:

`http://127.0.0.1:8000/docs`

### 5. Start the frontend

Open another terminal:

```bash
uv run streamlit run frontend/app.py
```

Application URL:

`http://localhost:8501`

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API health response |
| POST | `/chat` | AI conversational endpoint |
| POST | `/research` | Web research workflow |

### Chat Request

```json
{
  "message": "Explain AI agents",
  "history": []
}
```

### Research Request

```json
{
  "query": "The evolution of AI agents"
}
```

## Security and Reliability

- API keys managed through environment variables.
- Request validation using Pydantic.
- Safe API error responses.
- Application logging.
- Rate limiting using SlowAPI.
- Sensitive files excluded through `.gitignore`.

## Current Scope

Research and Chat are implemented and available.

PDF-based RAG functionality is planned for a future development phase and is not currently available.

Research results depend on external search results and webpage accessibility. AI-generated information should be independently verified.

## Author

**Akhilesh R**

[GitHub Profile](https://github.com/akhileshr24) · [ResearchAI Repository](https://github.com/akhileshr24/researchAi)

---

*Built with Python, FastAPI, LangGraph, Groq and Streamlit.*