from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage

from src.llm.providers import get_llm
from src.tools.search import web_search
from src.tools.scraper import scrape_page


class ResearchState(TypedDict):
    query: str
    sources: list
    content: list
    answer: str


def search(state):
    results = web_search(state["query"])
    return {"sources": results["results"]}


def extract(state):
    content = []

    for source in state["sources"][:5]:
        text = scrape_page(source["url"])

        if text:
            content.append({
                "title": source["title"],
                "url": source["url"],
                "content": text[:5000]
            })

    return {"content": content}


def research(state):
    if not state["content"]:
        return {"answer": "No webpage content could be extracted. Please try another topic."}

    llm = get_llm()

    sources = "\n\n".join(
        f"Title: {item['title']}\nURL: {item['url']}\nContent: {item['content']}"
        for item in state["content"]
    )

    response = llm.invoke([
        SystemMessage(content="""
        You are a research assistant.
        Write a clear, structured research report.
        Use only the provided webpage content.
        Include source URLs.
        Do not invent facts.
        """),
        HumanMessage(content=f"Research topic: {state['query']}\n\nSources:\n{sources}")
    ])

    return {"answer": response.content}


builder = StateGraph(ResearchState)

builder.add_node("search", search)
builder.add_node("extract", extract)
builder.add_node("research", research)

builder.add_edge(START, "search")
builder.add_edge("search", "extract")
builder.add_edge("extract", "research")
builder.add_edge("research", END)

research_graph = builder.compile()