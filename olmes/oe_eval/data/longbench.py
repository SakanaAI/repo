PROMPT = """
Write a high-quality answer for the given query using only the provided context information.

## Context
{context}

Query: {question}
Answer:
""".strip()

LONG_BENCH_TASKS = ["narrativeqa", "qasper", "multifieldqa_en", "2wikimqa", "musique", \
            "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht"]

LONG_BENCH_TASK_TYPES = {
    "multidoc_qa": ["2wikimqa", "musique", "dureader"],
    "singledoc_qa": ["multifieldqa_en", "narrativeqa", "qasper"],
    "summarization": ["gov_report", "qmsum", "multi_news", "vcsum"],
    "fewshot": ["trec", "triviaqa", "samsum", "lsht"]
}