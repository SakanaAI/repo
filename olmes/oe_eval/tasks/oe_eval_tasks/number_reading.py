
import re
from typing import List, Union

from oe_eval.components.instances import RequestInstance
from oe_eval.components.requests import RequestType
from oe_eval.data.nlgraph_tasks import NLGRAPH_TASKS, NLGRAPH_LEVEL
from oe_eval.metrics.metric import ExactMatch, MCAccuracy
from oe_eval.tasks.base_task import Task
from oe_eval.tasks.utils import apply_prompt_template, map_indexed
from oe_eval.utils import get_dict_with_defaults


def create_core_number_reading_tasks() -> dict:
    return {f"number_reading_{level}": create_number_reading_task(level) for level in ["easy", "middle", "hard"]}


def create_number_reading_task(level):
    class NumberReading(GenericNumberReading):
        TASK_CONFIG_DEFAULTS = get_dict_with_defaults(
            {
                "dataset_name": level
            },
            GenericNumberReading.TASK_CONFIG_DEFAULTS,
        )

    return NumberReading

class GenericNumberReading(Task):
    VERSION = 0
    REQUEST_TYPE = RequestType.GENERATE_UNTIL
    TASK_CONFIG_DEFAULTS = {
        "dataset_path": "huayangli/number_reading",
        "primary_metric": "exact_match",
        "split": "test",
        "num_shots": 0,
        "generation_kwargs": {
            "max_gen_toks": 128,
            "temperature": 0.0,
            "do_sample": False,
            "stop_sequences": ["\n"],
        }
    }

    def make_metrics(self):
        metric_kwargs = self.task_config["metric_kwargs"]
        self._metrics = [
            ExactMatch(
                extract_pred_fn=self._extract_answer,
                ignore_case=True,
                ignore_punctuation=True,
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

    def training_docs(self):
        return [self._process_doc(doc, doc_id) for doc_id, doc in enumerate(self.dataset['train'])]

    def test_docs(self):
        return [self._process_doc(doc, doc_id) for doc_id, doc in enumerate(self.dataset['test'])]

    def _process_doc(self, doc, doc_id):
        query = doc["prompt"]
        out_doc = {
            "index": doc_id,
            "query": query,
            "answer": doc["golden_reading"][0].lower(),
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]

    def doc_to_target(self, doc):
        return " " + doc["answer"]

    def construct_requests(
        self, doc: dict, ctx: Union[str, list, dict], doc_id: int
    ) -> List[RequestInstance]:
        return self.construct_basic_generation_requests(doc, ctx, doc_id, label=doc["answer"])

    def _extract_answer(self, continuation: str):
        pred = continuation.lower().strip()
        # pattern = r"<reading>(.*?)</reading>"
        # # re.DOTALL makes '.' match newlines too
        # matches = re.findall(pattern, pred, re.DOTALL)
        # ans = matches[-1] if matches else ""
        # import pdb; pdb.set_trace()
        return pred