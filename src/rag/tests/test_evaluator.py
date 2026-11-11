
from src.rag.tests.retrieval_evaluator import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
)


# Simulated retrieval results
retrieved = [
    (3, 0),
    (2, 0),
    (3, 1),
    (4, 0),
]

# Human-annotated relevant chunks
relevance_grades = {
    (3, 0): 2,
    (3, 1): 2,
    (2, 1): 1,
}

# Convert relevance grades into a set
relevant = set(relevance_grades.keys())

k = 4

print("Precision@4:", precision_at_k(retrieved, relevant, k))
print("Recall@4:", recall_at_k(retrieved, relevant, k))
print("MRR:", reciprocal_rank(retrieved, relevant))
print("nDCG@4:", ndcg_at_k(retrieved, relevance_grades, k))
