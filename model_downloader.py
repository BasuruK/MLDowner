import os
from huggingface_hub import snapshot_download

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Get user input for repository ID and local directory
repo_id = input("Enter the Hugging Face repository ID (e.g., unsloth/Qwen3-14B): ").strip()
local_dir = input("Enter the local directory path to save the model: ").strip()

# Validate inputs
if not repo_id:
    print("Error: Repository ID cannot be empty.")
    exit(1)

if not local_dir:
    print("Error: Local directory path cannot be empty.")
    exit(1)

print(f"Downloading model from '{repo_id}' to '{local_dir}'...")

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    allow_patterns=["*.safetensors", "*config.json"]
)
