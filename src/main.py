from src.agent.graph import research_graph

query = input("Enter research topic: ")

result = research_graph.invoke({
    "query": query,
    "sources": [],
    "content": [],
    "answer": ""
})

print("\n===== RESEARCH REPORT =====\n")
print(result["answer"])

print("\n===== SOURCES =====\n")

for source in result["content"]:
    print(source["title"])
    print(source["url"])
    print()