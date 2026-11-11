
from src.rag.vectorstore import search
from src.rag.tests.evaluation_cases import evaluation_cases


total = len(evaluation_cases)
passed = 0

for case in evaluation_cases:
    question = case["question"]
    expected = case["expected"]

    # Retrieve relevant chunks
    results = search(question)

    # Determine actual retrieval result
    actual = "relevant" if results else "irrelevant"

    # Compare expected and actual
    status = "PASS" if actual == expected else "FAIL"

    if status == "PASS":
        passed += 1

    print(f"\nQuestion: {question}")
    print(f"Expected: {expected}")
    print(f"Actual:   {actual}")
    print(f"Chunks retrieved: {len(results)}")
    print(f"Result: {status}")

# Evaluation summary
print("\n========== EVALUATION SUMMARY ==========")
print(f"Total tests: {total}")
print(f"Passed: {passed}")
print(f"Failed: {total - passed}")
print(f"Accuracy: {(passed / total) * 100:.2f}%")
