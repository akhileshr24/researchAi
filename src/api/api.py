
import logging
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)

from src.agent.graph import research_graph
from src.api.rag_routes import router as rag_router
from src.api.rate_limit import limiter
from src.llm.providers import get_llm


# LOGGING
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# FASTAPI APP
app = FastAPI(
    title="ResearchAI",
    description="AI-powered research assistant",
    version="1.0.0",
)


# RATE LIMITING
app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler,
)

# RAG ENDPOINTS
app.include_router(rag_router)


# REQUEST SCHEMAS

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=1000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=6,
    )

    @field_validator("message")
    @classmethod
    def clean_message(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Message cannot be empty")

        return value


class ResearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500)

    @field_validator("query")
    @classmethod
    def clean_query(cls, value: str) -> str:
        value = value.strip()

        if len(value) < 3:
            raise ValueError(
                "Query must contain at least 3 characters"
            )

        return value


# HOME ENDPOINT

@app.get("/")
def home():
    return {"message": "ResearchAI API is running"}


# CHAT ENDPOINT

@app.post("/chat")
@limiter.limit("10/minute")
def chat(request: Request, body: ChatRequest):
    try:
        messages = [
            SystemMessage(
                content=(
                    "You are ResearchAI, a helpful "
                    "and concise AI assistant."
                )
            )
        ]

        for item in body.history:
            if item.role == "user":
                messages.append(
                    HumanMessage(content=item.content)
                )
            else:
                messages.append(
                    AIMessage(content=item.content)
                )

        messages.append(
            HumanMessage(content=body.message)
        )

        llm = get_llm().bind(max_tokens=500)
        response = llm.invoke(messages)

        return {
            "message": body.message,
            "response": response.content,
        }

    except Exception:
        logger.exception("Chat request failed")

        raise HTTPException(
            status_code=500,
            detail="Chat failed. Please try again later.",
        )


# RESEARCH ENDPOINT

@app.post("/research")
@limiter.limit("3/minute")
def research(request: Request, body: ResearchRequest):
    try:
        result = research_graph.invoke({
            "query": body.query,
            "sources": [],
            "content": [],
            "answer": "",
        })

        return {
            "query": body.query,
            "report": result["answer"],
            "sources": result["content"],
        }

    except Exception:
        logger.exception("Research workflow failed")

        raise HTTPException(
            status_code=500,
            detail="Research failed. Please try again later.",
        )
