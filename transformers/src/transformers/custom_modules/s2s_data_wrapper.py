PROMPT_DICT = {
    "copy": "Copy the following words: {src} .",
    "addition": "Compute: {src} ? The answer is",
    "reverse": "Reverse the following words: {src} .",
    "scan": "Generate the sequence based on given operations: {src} . The answer is",
    "summation": "Compute: {src} ? The answer is",
    "pcfg": "Generate the sequence based on given operations: {src} . The answer is",
    "graph_connect": "Query: {query} . Graph: {src} . The answer is",
    "graph_path": "Query: {query} . Graph: {src} . Found path:",
    "graph_adj": "Query: {query} . Graph: {src} . Found adjacent nodes:",
    "dyck_classify": "Is the given input a valid Dyck string ? Input: {src} . The answer is",
    "dyck_complete": "Complete the given Dyck string . Input: {src} . Completion:"
}

def data_wrapper(item, task_name, input_key="prompt", output_key="response", cat_key="cat"):
    assert task_name in PROMPT_DICT
    if task_name == "copy":
        src = PROMPT_DICT[task_name].format(src=item["src_seq"])
        tgt = item["tgt_seq"]
        return {input_key: src, output_key: tgt, cat_key: item["cat"]}
    elif task_name == "addition":
        raw_src = [ch for ch in str(item["a"])] + ['+'] + [ch for ch in str(item["b"])]
        raw_src = " ".join(raw_src)
        src = PROMPT_DICT[task_name].format(src=raw_src)
        tgt = " ".join([ch for ch in str(item["sum"])])
        return {input_key: src, output_key: tgt, cat_key: item["cat"]}
    elif task_name == "reverse":
        src = PROMPT_DICT[task_name].format(src=item["src_seq"])
        tgt = item["tgt_seq"]
        return {input_key: src, output_key: tgt, cat_key: item["cat"]}
    elif task_name == "scan":
        src = PROMPT_DICT[task_name].format(src=item["source"])
        tgt = item["target"]
        return {input_key: src, output_key: tgt, cat_key: item["category"]}
    elif task_name == "summation":
        raw_src = f"( {' + '.join([str(it) for it in item['values']])} ) % 10"
        src = PROMPT_DICT[task_name].format(src=raw_src)
        tgt = str(int(item["answer"] % 10))
        return {input_key: src, output_key: tgt, cat_key: item["cat"]}
    elif task_name == "pcfg":
        src = PROMPT_DICT[task_name].format(src=item["source"])
        tgt = item["target"]
        return {input_key: src, output_key: tgt, cat_key: item["category"]}
    elif task_name == "graph_connect":
        src = PROMPT_DICT[task_name].format(src=item["graph"], query=item["query"])
        tgt = item["connected"]
        return {input_key: src, output_key: tgt, cat_key: item["cat"]}
    elif task_name == "graph_path":
        src = PROMPT_DICT[task_name].format(src=item["graph"], query=item["query"])
        tgt = item["paths"][0]
        return {input_key: src, output_key: tgt, cat_key: item["cat"]}
    elif task_name == "graph_adj":
        src = PROMPT_DICT[task_name].format(src=item["graph"], query=item["query"])
        tgt = item["adj_nodes"]
        return {input_key: src, output_key: tgt, cat_key: item["cat"]}
    elif task_name == "dyck_complete":
        src = PROMPT_DICT[task_name].format(src=item["src_seq"])
        tgt = item["tgt_seq"]
        return {input_key: src, output_key: tgt, cat_key: item["cat"]}
    elif task_name == "dyck_classify":
        src = PROMPT_DICT[task_name].format(src=item["expr"])
        tgt = item["tag"]
        return {input_key: src, output_key: tgt, cat_key: item["cat"]}
    else:
        raise NotImplementedError