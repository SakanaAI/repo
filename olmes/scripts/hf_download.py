from huggingface_hub import snapshot_download

# Define the repository ID and local directory
repo_id = "SakanaAI/RePo-OLMo2-1B-stage2-L5"
local_dir = "../OLMo/hf_ckpts/OLMo2-1B-stage2-seed42-SEXMH-L5/step23852-unsharded/"

print(f"Downloading {repo_id} to {local_dir}...")

# Download the repository
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # Set to False to download actual files, not symlinks
    revision="main"                # Optional: specify branch (default is main)
)

print("Download complete.")
