
from src.rag.qa import ask_question


result = ask_question("What is machine learning?")
print(result)

print("\nANSWER:")
print(result["answer"])

print("\nSOURCES:")
for source in result["sources"]:
    print(f"{source['filename']} - Page {source['page']}")
