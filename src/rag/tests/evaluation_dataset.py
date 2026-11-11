
evaluation_dataset = [
    # Machine Learning
    {
        "question": "What is machine learning?",
        "answerable": True,
        "relevance_grades": {
            (3, 0): 2,
            (2, 1): 1,
        },
    },
    {
        "question": "How does supervised learning work?",
        "answerable": True,
        "relevance_grades": {
            (3, 0): 2,
            (3, 1): 2,
        },
    },
    {
        "question": "What is a train-test split?",
        "answerable": True,
        "relevance_grades": {
            (3, 1): 2,
        },
    },
    {
        "question": "How do we evaluate classification models?",
        "answerable": True,
        "relevance_grades": {
            (3, 2): 2,
        },
    },

    # Deep Learning and LLMs
    {
        "question": "What is deep learning?",
        "answerable": True,
        "relevance_grades": {
            (4, 0): 2,
            (2, 1): 1,
        },
    },
    {
        "question": "What is a large language model?",
        "answerable": True,
        "relevance_grades": {
            (4, 0): 2,
        },
    },
    {
        "question": "What are tokens and context windows?",
        "answerable": True,
        "relevance_grades": {
            (4, 1): 2,
        },
    },
    {
        "question": "Why do LLMs hallucinate?",
        "answerable": True,
        "relevance_grades": {
            (4, 1): 2,
            (4, 2): 1,
        },
    },

    # RAG
    {
        "question": "What is retrieval augmented generation?",
        "answerable": True,
        "relevance_grades": {
            (5, 0): 2,
        },
    },
    {
        "question": "What happens during the RAG pipeline?",
        "answerable": True,
        "relevance_grades": {
            (5, 0): 2,
        },
    },
    {
        "question": "Why does RAG store original text and embeddings?",
        "answerable": True,
        "relevance_grades": {
            (5, 1): 2,
        },
    },
    {
        "question": "What should we consider when building production RAG systems?",
        "answerable": True,
        "relevance_grades": {
            (5, 2): 2,
            (5, 1): 1,
        },
    },

    # Artificial Intelligence
    {
        "question": "What is artificial intelligence?",
        "answerable": True,
        "relevance_grades": {
            (2, 0): 2,
            (2, 1): 1,
        },
    },

    # Unanswerable questions
    {
        "question": "What is quantum computing?",
        "answerable": False,
        "relevance_grades": {},
    },
    {
        "question": "Who won the FIFA World Cup in 2022?",
        "answerable": False,
        "relevance_grades": {},
    },
    {
        "question": "What is the capital of Japan?",
        "answerable": False,
        "relevance_grades": {},
    },
]


# Compatibility with retrieval_evaluator.py
ground_truth = {
    case["question"]: case["relevance_grades"]
    for case in evaluation_dataset
}
