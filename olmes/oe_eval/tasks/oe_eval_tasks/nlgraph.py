
import re
from typing import List, Union

from oe_eval.components.instances import RequestInstance
from oe_eval.components.requests import RequestType
from oe_eval.data.nlgraph_tasks import NLGRAPH_TASKS, NLGRAPH_LEVEL
from oe_eval.metrics.metric import ExactMatch, MCAccuracy
from oe_eval.tasks.base_task import Task
from oe_eval.tasks.utils import apply_prompt_template, map_indexed
from oe_eval.utils import get_dict_with_defaults

def create_core_nlgraph_tasks() -> dict:
    return {f"nlgraph_{task_type}_{level}": create_nlgraph_task(task_type, level) for task_type in NLGRAPH_TASKS for level in NLGRAPH_LEVEL}


def create_nlgraph_task(task_type, level):
    class NLGraph(GenericNLGraph):
        TASK_CONFIG_DEFAULTS = get_dict_with_defaults(
            {
                "dataset_name": task_type,
                "context_kwargs": {"description": task_type},
                "context_kwargs": {
                    "level": level
                }
            },
            GenericNLGraph.TASK_CONFIG_DEFAULTS,
        )

    return NLGraph




class GenericNLGraph(Task):
    VERSION = 0
    REQUEST_TYPE = RequestType.GENERATE_UNTIL
    TASK_CONFIG_DEFAULTS = {
        "dataset_path": "huayangli/nlgraph",
        "native_id_field": "doc_id",
        "primary_metric": "exact_match",
        "split": "train",
        "num_shots": 0,
        "generation_kwargs": {
            "max_gen_toks": 512,
            "temperature": 0.0,
            "do_sample": False,
            "stop_sequences": ["\n\n", "<|eot_id|>", "</s>", "<|im_end|>"],
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
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        level = self.task_config["context_kwargs"]["level"]
        return [self._process_doc(doc) for doc in self.dataset["train"] if doc['difficulty'] == level]

    def test_docs(self):
        level = self.task_config["context_kwargs"]["level"]
        return [self._process_doc(doc) for doc in self.dataset["test"] if doc['difficulty'] == level]

    def _process_doc(self, doc):
        doc_id = int(doc["doc_id"])
        query = doc["question"]
        answer = self._extract_answer(doc["answer"])
        out_doc = {
            "index": doc_id,
            "full_answer": doc["answer"],
            "query": query,
            "answer": answer,
            "level": doc['difficulty'],
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
        task_name = self.task_config["dataset_name"]
        pred = continuation.lower()
        answer = ""
        if task_name == "connectivity":
            answer = "yes" if "answer is yes" in pred else "no"
        elif task_name == "cycle":
            answer = "yes" if "there is a cycle in this graph" in pred else "no"
        elif task_name == "flow":
            match = re.search(r"maximum flow from node (\d+) to node (\d+) is (\d+)", pred)
            if match:
                answer = match.group(3)
        elif task_name == "GNN":
            pattern = r"node\s+(\d+):\s*\[(\d+),\s*(\d+)\]"
            matches = re.findall(pattern, pred.split("So the answer is:")[-1])
            # Convert to dictionary
            answer = "\n".join([f"node {node}: [{x}, {y}]" for node, x, y in matches])
        elif task_name == "hamilton":
            match = re.search(r"(\d+(?:,\d+)*)", pred)
            if match:
                answer = match.group(1)
        elif task_name == "matching":
            pattern = r"applicant\s+(\d+):\s+job\s+(\d+)"
            assignments = re.findall(pattern, pred)
            answer = ",".join([f"{app}:{job}" for app, job in assignments])
        elif task_name == "shortest_path":
            match = re.search(r'shortest path from node \d+ to node \d+ is ([\d,]+)', pred)
            if match:
                answer = match.group(1)
        elif task_name == "topology":
            match = re.search(r'the solution is: ([\d,]+)\.', pred)
            if match:
                answer = match.group(1)
        else:
            raise NotImplementedError
   
        return answer