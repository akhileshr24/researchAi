
# Human-annotated relevance labels.
# Key: (page_number, chunk_index)
# Grade: 0 = irrelevant, 1 = partially relevant, 2 = highly relevant

ground_truth = {
    "What is machine learning?": {
        (3, 0): 2,
        (2, 1): 2,
        (1, 0): 1,
    },

    "What is deep learning?": {
        (4, 0): 2,
        (2, 1): 1,
    },

    "What is artificial intelligence?": {
        (2, 0): 2,
        (2, 1): 1,
        (1, 0): 1,
    },

    "What is quantum computing?": {},

    "Explain how machine learning models learn from data.": {
        (3, 0): 2,
        (3, 1): 2,
    },

    "Who won the FIFA World Cup in 2022?": {},
}
