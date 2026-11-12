from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ---------- RAG ----------
CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MIN_RELEVANCE_SCORE = 0.40
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "0.0"))
TOP_K = 4
RRF_K = 60
RERANK_CANDIDATES = 20
