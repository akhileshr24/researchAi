from langchain_tavily import TavilySearch
from src.config.settings import TAVILY_API_KEY


def web_search(query):
    tool = TavilySearch(
        max_results=5,
        tavily_api_key=TAVILY_API_KEY
    )

    return tool.invoke({"query": query})