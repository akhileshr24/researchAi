from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config.settings import *

def get_llm(provider="groq"):

    if provider == "groq":
        return ChatGroq(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY
        )

    elif provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY
        )

    elif provider == "nvidia":
        return ChatOpenAI(
            model=NVIDIA_MODEL,
            api_key=NVIDIA_API_KEY,
            base_url=NVIDIA_BASE_URL
        )

    raise ValueError("Invalid LLM provider")