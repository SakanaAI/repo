import os
import yaml
import subprocess

CONFIG_PATH = "./configs/official-0425/OLMo2-1B-stage2-seed42.yaml"
DEST_DIR = "/home/huayang_sakana_ai/workspace/olmo_data"

def main():
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    paths = cfg.get("data", {}).get("paths", [])
    for url in paths:
        if not url.startswith("http"):
            continue

        rel_path = url.split("://", 1)[1].split("/", 1)[1]
        dest_path = os.path.join(DEST_DIR, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        print(f"⬇️ Downloading {url}")
        subprocess.run(
            ["wget", "-c", "-O", dest_path, url],
            check=True
        )

if __name__ == "__main__":
    main()
