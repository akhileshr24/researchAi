from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.agent.graph import research_graph

app = FastAPI(title="AI Research Assistant")


class ResearchRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {"message": "AI Research Assistant API is running"}


@app.post("/research")
def research(request: ResearchRequest):
    try:
        result = research_graph.invoke({
            "query": request.query,
            "sources": [],
            "content": [],
            "answer": ""
        })

        return {
            "query": request.query,
            "report": result["answer"],
            "sources": result["content"]
        }

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))