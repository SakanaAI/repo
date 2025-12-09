
import re
from typing import List, Union

from oe_eval.components.instances import RequestInstance
from oe_eval.components.requests import RequestType
from oe_eval.data.lost_in_the_middle_tasks import PROMPTS, LOST_IN_THE_MIDDLE_TASKS
from oe_eval.metrics.metric import ExactMatch, MCAccuracy
from oe_eval.tasks.base_task import Task
from oe_eval.tasks.utils import apply_prompt_template, map_indexed
from oe_eval.utils import get_dict_with_defaults
import string
from typing import List

import regex


def create_core_lost_in_the_middle_tasks() -> dict:
    tasks = dict()
    for task_info in LOST_IN_THE_MIDDLE_TASKS:
        num_docs = task_info['num_docs']
        positions = task_info['gold_at']
        for pos in positions:
            tasks[f"lost_in_the_middle_{num_docs}_at_{pos}"] = create_lost_in_the_middle_task(num_docs, pos) 
    return tasks


def create_lost_in_the_middle_task(num_docs, pos):
    class LostInTheMiddle(GenericLostInTheMiddle):
        TASK_CONFIG_DEFAULTS = get_dict_with_defaults(
            {
                "dataset_name": f"nq-open-{num_docs}_total_documents_gold_at_{pos}",
                "context_kwargs": {
                    "prompt_key": "qa"
                }
            },
            GenericLostInTheMiddle.TASK_CONFIG_DEFAULTS,
        )

    return LostInTheMiddle


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


class GenericLostInTheMiddle(Task):
    VERSION = 0
    REQUEST_TYPE = RequestType.GENERATE_UNTIL
    TASK_CONFIG_DEFAULTS = {
        "dataset_path": "huayangli/lost_in_the_middle",
        "native_id_field": "doc_id",
        "primary_metric": "substring_exact_match",
        "split": "test",
        "num_shots": 0,
        "generation_kwargs": {
            "max_gen_toks": 512,
            "temperature": 0.0,
            "do_sample": False,
            "stop_sequences": ["\n\n"], # stop new line
        }
    }

    def make_metrics(self):
        metric_kwargs = self.task_config["metric_kwargs"]
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
        prompt_key = self.task_config["context_kwargs"]["prompt_key"]
        return [self._process_doc(doc, PROMPTS[prompt_key]) for doc in self.dataset["test"]]

    def _process_doc(self, doc, prompt_template):
        question = doc['question']
        ctxs = doc['ctxs']
        formatted_documents = []
        num_docs = len(ctxs['id'])
        for idx in range(num_docs):
            formatted_documents.append(f"Document [{idx+1}](Title: {ctxs['title'][idx]}) {ctxs['text'][idx]}")
        query = prompt_template.format(question=question, search_results="\n\n".join(formatted_documents))
        answer = [normalize_answer(ans) for ans in doc['answers']]
        out_doc = {
            "query": query,
            "answer": answer,
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]

    def doc_to_target(self, doc):
        return " " + doc["answer"][0]

    def construct_requests(
        self, doc: dict, ctx: Union[str, list, dict], doc_id: int
    ) -> List[RequestInstance]:
        return self.construct_basic_generation_requests(doc, ctx, doc_id, label=doc["answer"])

    def _extract_answer(self, continuation: str):
        answer = normalize_answer(continuation)
        return answer