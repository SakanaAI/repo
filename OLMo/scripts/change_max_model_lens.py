import os
import json

# path to your Hugging Face model directory
MODEL_DIR = "/home/huayang_sakana_ai/workspace/OLMo/hf_ckpts/"

def update_max_position_embeddings(directory, old_value=4096, new_value=65536):
    for root, _, files in os.walk(directory):
        for file in files:
            if file == "config.json":
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    if "max_position_embeddings" in data and data["max_position_embeddings"] == old_value:
                        print(f"Updating {file_path}")
                        data["max_position_embeddings"] = new_value
                        with open(file_path, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"Skipping {file_path}, error: {e}")


update_max_position_embeddings(MODEL_DIR)