
import re
from typing import List, Union

from oe_eval.components.instances import RequestInstance
from oe_eval.components.requests import RequestType
from oe_eval.metrics.metric import ExactMatch, MCAccuracy, GeneralMatch, RougeMetric
from oe_eval.tasks.base_task import Task
from oe_eval.tasks.utils import apply_prompt_template, map_indexed
from oe_eval.utils import get_dict_with_defaults
from oe_eval.data.longbench import LONG_BENCH_TASKS, PROMPT, LONG_BENCH_TASK_TYPES
import string
from typing import List

import regex


def create_core_long_bench_tasks() -> dict:
    tasks = dict()
    for task_name in LONG_BENCH_TASKS:
        tasks[f"longbench_{task_name}"] = create_long_bench_task(task_name) 
    return tasks


def create_long_bench_task(task_name):
    if task_name in LONG_BENCH_TASK_TYPES['summarization']:
        max_gen_toks = 512
        primary_metric = "rougeL_f1"
    elif task_name in ["trec", "lsht"]:
        max_gen_toks = 32
        primary_metric = "substring_exact_match"
    else:
        max_gen_toks = 32
        primary_metric = "f1"

    class LongBench(GenericLongBench):
        TASK_CONFIG_DEFAULTS = get_dict_with_defaults(
            {
                "dataset_name": task_name,
                'primary_metric': primary_metric,
                "generation_kwargs": {
                    "max_gen_toks": max_gen_toks,
                }
            },
            GenericLongBench.TASK_CONFIG_DEFAULTS,
        )

    return LongBench


def normalize_answer(s: str) -> str:
    """Normalization from the SQuAD evaluation script.

    See https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
    """

    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


class GenericLongBench(Task):
    VERSION = 0
    REQUEST_TYPE = RequestType.GENERATE_UNTIL
    TASK_CONFIG_DEFAULTS = {
        "dataset_path": "huayangli/long_bench",
        "primary_metric": "substring_exact_match",
        "split": "test",
        "num_shots": 0,
        "generation_kwargs": {
            "max_gen_toks": 32,
            "temperature": 0.0,
            "do_sample": False,
            "stop_sequences": ["\n\n", "\n"], # stop new line
        }
    }

    def make_metrics(self):
        metric_kwargs = self.task_config["metric_kwargs"]
        dataset_name = self.task_config["dataset_name"]
        if dataset_name in ["gov_report", "qmsum", "multi_news", "vcsum", "samsum"]:
            self._metrics = [
                RougeMetric(
                    **metric_kwargs,
                )
            ]
        elif dataset_name in ["narrativeqa", "qasper", "multifieldqa_en", "2wikimqa", "musique", "triviaqa"]:
            self._metrics = [
                GeneralMatch(
                    extract_pred_fn=self._extract_answer,
                    return_f1=True,  # The dyck_languages task has punctuation in answers
                    **metric_kwargs,
                )
            ]
        elif dataset_name in ["trec", "lsht"]:
            self._metrics = [
                ExactMatch(
                    extract_pred_fn=self._extract_answer,
                    metric_names=['substring_exact_match'],
                    ignore_case=True,
                    ignore_punctuation=True,  # The dyck_languages task has punctuation in answers
                    **metric_kwargs,
                )
            ]
        return self._metrics

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        dataset_name = self.task_config["dataset_name"]
        return [self._process_doc(doc, dataset_name) for doc in self.dataset["test"]]

    def _process_doc(self, doc, dataset_name):
        if dataset_name in ["gov_report", "multi_news"]:
            question = "Summarize the given context into a paragraph."
        else:
            question = doc['input']
        
        if dataset_name in LONG_BENCH_TASK_TYPES['fewshot']:
            query = doc['context'] + "\n" + question
        else:
            query = PROMPT.format(
                context=doc['context'], question=question
            )
        if dataset_name in LONG_BENCH_TASK_TYPES['summarization']:
            answer = doc['answers'][0]
        else:
            answer = [normalize_answer(ans) for ans in doc['answers']]

        out_doc = {
            "query": query,
            "answer": answer,
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]

    def doc_to_target(self, doc):
        return " " + doc["answer"] if isinstance(doc['answer'], str) else doc["answer"][0]

    def construct_requests(
        self, doc: dict, ctx: Union[str, list, dict], doc_id: int
    ) -> List[RequestInstance]:
        return self.construct_basic_generation_requests(doc, ctx, doc_id, label=doc["answer"])

    def _extract_answer(self, continuation: str):
        answer = normalize_answer(continuation)
        return answer