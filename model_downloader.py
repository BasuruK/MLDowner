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

# For fine-tuning/retraining, we need to download all essential files
# including model weights, configs, tokenizer files, and metadata
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    allow_patterns=[
        # Model weights (both safetensors and bin formats)
        "*.safetensors",
        "*.bin",
        
        # Configuration files
        "*config.json",
        "generation_config.json",
        
        # Tokenizer files (essential for fine-tuning)
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",  # For some models like SentencePiece
        
        # Additional metadata files
        "README.md",
        ".gitattributes",
        "model.safetensors.index.json",  # For sharded models
        
        # Training-related files (if present)
        "training_args.bin",
        "trainer_state.json",
        
        # Other potential files
        "pytorch_model.bin.index.json",
        "preprocessor_config.json"
    ]
)

print(f"Download completed! Model saved to: {local_dir}")
print("\nFor fine-tuning with Unsloth, you can now use:")
print(f"from unsloth import FastLanguageModel")
print(f"model, tokenizer = FastLanguageModel.from_pretrained('{local_dir}')")
