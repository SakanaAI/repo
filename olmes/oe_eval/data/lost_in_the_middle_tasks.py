PROMPTS = dict()

PROMPTS['qa_with_query_aware_contextualization'] = """
Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant).

Question: {question}

{search_results}

Question: {question}
Answer:
""".strip()

PROMPTS['qa'] = """
Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant).

{search_results}

Question: {question}
Answer:
""".strip()

LOST_IN_THE_MIDDLE_TASKS = [
    {"num_docs": 10, "gold_at": [0, 4, 9]},
    {"num_docs": 20, "gold_at": [0, 4, 9, 14, 19]},
    {"num_docs": 30, "gold_at": [0, 4, 9, 14, 19, 24, 29]}
]