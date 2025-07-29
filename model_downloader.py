import os
import glob
from huggingface_hub import snapshot_download

try:
    from safetensors import safe_open
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("Warning: safetensors library not found. Install with: pip install safetensors")
    print("Safetensors validation will be skipped.")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not found. Install with: pip install torch")
    print("Safetensors validation will be skipped.")

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

# Check if directory already exists and has files
if os.path.exists(local_dir):
    existing_files = os.listdir(local_dir)
    if existing_files:
        print(f"Directory '{local_dir}' already exists with {len(existing_files)} files.")
        print("The download will check for updates and only download modified/new files.")
    else:
        print(f"Directory '{local_dir}' exists but is empty.")
else:
    print(f"Creating new directory: '{local_dir}'")

print(f"Downloading model from '{repo_id}' to '{local_dir}'...")

# For fine-tuning/retraining, we need to download all essential files
# including model weights, configs, tokenizer files, and metadata
# The function automatically resumes downloads and only updates changed files
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    force_download=False,  # Don't re-download if file already exists and is valid
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

# Validate safetensors files (only if both safetensors and torch are available)
if SAFETENSORS_AVAILABLE and TORCH_AVAILABLE:
    print("\nValidating safetensors files...")
    safetensors_files = glob.glob(os.path.join(local_dir, "*.safetensors"))
    
    if not safetensors_files:
        print("No safetensors files found to validate.")
    else:
        valid_files = 0
        invalid_files = 0
        
        for file_path in safetensors_files:
            try:
                # Try to open and read metadata from the safetensors file
                # Using GPU device for faster validation if available
                with safe_open(file_path, framework="pt", device="cuda") as f:
                    # Get metadata and tensor names to verify file structure
                    metadata = f.metadata()
                    tensor_names = f.keys()
                    
                    print(f"✅ Valid: {os.path.basename(file_path)} ({len(tensor_names)} tensors)")
                    valid_files += 1
                    
            except Exception as e:
                # If GPU validation fails, try CPU as fallback
                try:
                    with safe_open(file_path, framework="pt", device="cpu") as f:
                        metadata = f.metadata()
                        tensor_names = f.keys()
                        
                        print(f"✅ Valid: {os.path.basename(file_path)} ({len(tensor_names)} tensors) [CPU fallback]")
                        valid_files += 1
                except Exception as cpu_error:
                    print(f"❌ Invalid: {os.path.basename(file_path)} - Error: {str(cpu_error)}")
                    invalid_files += 1
        
        print(f"\nValidation Results:")
        print(f"  Valid files: {valid_files}")
        print(f"  Invalid files: {invalid_files}")
        
        if invalid_files > 0:
            print(f"⚠️  Warning: {invalid_files} safetensors file(s) appear to be corrupted!")
            print("You may want to delete the corrupted files and re-run the download.")
        else:
            print("✅ All safetensors files are valid!")
else:
    print("\nSkipping safetensors validation:")
    if not SAFETENSORS_AVAILABLE:
        print("  - safetensors library not available")
    if not TORCH_AVAILABLE:
        print("  - PyTorch not available")
    print("Install missing dependencies to enable validation:")
    print("  pip install safetensors torch")

print("\nFor fine-tuning with Unsloth, you can now use:")
print(f"from unsloth import FastLanguageModel")
print(f"model, tokenizer = FastLanguageModel.from_pretrained('{local_dir}')")
