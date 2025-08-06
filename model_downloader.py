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

def find_model_directories(base_path):
    """Find directories that likely contain ML models (one level deep only)."""
    model_dirs = []
    
    # Check if the base directory itself has model files
    safetensors_files = glob.glob(os.path.join(base_path, "*.safetensors"))
    bin_files = glob.glob(os.path.join(base_path, "*.bin"))
    config_files = glob.glob(os.path.join(base_path, "*config.json"))
    
    if safetensors_files or bin_files or config_files:
        model_dirs.append(base_path)
    
    # Check only immediate subdirectories (one level deep)
    try:
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            
            # Skip if it's not a directory
            if not os.path.isdir(item_path):
                continue
            
            # Skip if we already added the base directory and this is the same
            if item_path == base_path:
                continue
                
            # Check for model files in this immediate subdirectory
            safetensors_files = glob.glob(os.path.join(item_path, "*.safetensors"))
            bin_files = glob.glob(os.path.join(item_path, "*.bin"))
            config_files = glob.glob(os.path.join(item_path, "*config.json"))
            
            if safetensors_files or bin_files or config_files:
                model_dirs.append(item_path)
                
    except PermissionError:
        print(f"Warning: Permission denied accessing some directories in {base_path}")
    
    return model_dirs

def validate_directory_models(directory_path):
    """Validate safetensors files in a specific directory."""
    if not (SAFETENSORS_AVAILABLE and TORCH_AVAILABLE):
        return 0, 0, []
        
    safetensors_files = glob.glob(os.path.join(directory_path, "*.safetensors"))
    
    if not safetensors_files:
        return 0, 0, []
    
    valid_files = 0
    invalid_files = 0
    file_details = []
    
    for file_path in safetensors_files:
        try:
            # Try to open and read metadata from the safetensors file
            # Using GPU device for faster validation if available
            with safe_open(file_path, framework="pt", device="cuda") as f:
                # Get metadata and tensor names to verify file structure
                metadata = f.metadata()
                tensor_names = f.keys()
                
                # Test loading a tensor to ensure data integrity
                if tensor_names:
                    tensor = f.get_tensor(next(iter(tensor_names)))
                
                file_details.append((os.path.basename(file_path), len(tensor_names), "Valid"))
                valid_files += 1
                
        except Exception as e:
            # If GPU validation fails, try CPU as fallback
            try:
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    metadata = f.metadata()
                    tensor_names = f.keys()
                    
                    # Test loading a tensor to ensure data integrity
                    if tensor_names:
                        tensor = f.get_tensor(next(iter(tensor_names)))
                    
                    file_details.append((os.path.basename(file_path), len(tensor_names), "Valid (CPU)"))
                    valid_files += 1
            except Exception as cpu_error:
                file_details.append((os.path.basename(file_path), 0, f"Invalid: {str(cpu_error)}"))
                invalid_files += 1
    
    return valid_files, invalid_files, file_details

# Ask user to choose operation mode
print("Choose an operation:")
print("1. Download models from Hugging Face")
print("2. Verify existing local model in specific directory")
print("3. Verify all models in directory (including subdirectories)")
choice = input("Enter your choice (1, 2, or 3): ").strip()

if choice not in ['1', '2', '3']:
    print("Error: Invalid choice. Please enter 1, 2, or 3.")
    exit(1)

if choice == '1':
    # Download mode
    repo_id = input("Enter the Hugging Face repository ID (e.g., unsloth/Qwen3-14B): ").strip()
    local_dir = input("Enter the local directory path to save the model: ").strip()
    
    # Validate inputs
    if not repo_id:
        print("Error: Repository ID cannot be empty.")
        exit(1)
    
    if not local_dir:
        print("Error: Local directory path cannot be empty.")
        exit(1)
elif choice == '2':
    # Verify single model mode
    local_dir = input("Enter the local directory path containing the model to verify: ").strip()
    
    if not local_dir:
        print("Error: Local directory path cannot be empty.")
        exit(1)
    
    if not os.path.exists(local_dir):
        print(f"Error: Directory '{local_dir}' does not exist.")
        exit(1)
else:
    # Verify all models in directory mode (choice == '3')
    base_dir = input("Enter the base directory path to search for models (or press Enter for current directory): ").strip()
    
    if not base_dir:
        base_dir = os.getcwd()
        print(f"Using current directory: {base_dir}")
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory '{base_dir}' does not exist.")
        exit(1)

# Only proceed with download if user chose option 1
if choice == '1':
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
elif choice == '2':
    print(f"Verifying model in: {local_dir}")
else:
    print(f"Searching for models in: {base_dir}")

# Handle validation based on choice
if choice == '3':
    # Verify all models in directory and subdirectories
    model_directories = find_model_directories(base_dir)
    
    if not model_directories:
        print("No model directories found (no .safetensors, .bin, or config.json files detected).")
    else:
        print(f"\nFound {len(model_directories)} potential model directories:")
        
        total_valid = 0
        total_invalid = 0
        
        for i, model_dir in enumerate(model_directories, 1):
            relative_path = os.path.relpath(model_dir, base_dir)
            if relative_path == '.':
                relative_path = os.path.basename(base_dir)
            
            print(f"\n[{i}/{len(model_directories)}] Checking: {relative_path}")
            
            if SAFETENSORS_AVAILABLE and TORCH_AVAILABLE:
                valid_files, invalid_files, file_details = validate_directory_models(model_dir)
                total_valid += valid_files
                total_invalid += invalid_files
                
                if file_details:
                    for filename, tensor_count, status in file_details:
                        if "Valid" in status:
                            print(f"  ✅ {filename} ({tensor_count} tensors) - {status}")
                        else:
                            print(f"  ❌ {filename} - {status}")
                else:
                    print(f"  ℹ️  No safetensors files found")
            else:
                # Just list the files without validation
                safetensors_files = glob.glob(os.path.join(model_dir, "*.safetensors"))
                bin_files = glob.glob(os.path.join(model_dir, "*.bin"))
                config_files = glob.glob(os.path.join(model_dir, "*config.json"))
                
                print(f"  📁 Contains: {len(safetensors_files)} safetensors, {len(bin_files)} bin, {len(config_files)} config files")
        
        if SAFETENSORS_AVAILABLE and TORCH_AVAILABLE:
            print(f"\n📊 Overall Results:")
            print(f"  Total valid safetensors files: {total_valid}")
            print(f"  Total invalid safetensors files: {total_invalid}")
            print(f"  Model directories checked: {len(model_directories)}")
            
            if total_invalid > 0:
                print(f"⚠️  Warning: {total_invalid} safetensors file(s) appear to be corrupted!")
            else:
                print("✅ All found safetensors files are valid!")

else:
    # Original single directory validation (choices 1 and 2)
    target_dir = local_dir if choice in ['1', '2'] else base_dir
    
    # Validate safetensors files (only if both safetensors and torch are available)
    if SAFETENSORS_AVAILABLE and TORCH_AVAILABLE:
        print("\nValidating safetensors files...")
        valid_files, invalid_files, file_details = validate_directory_models(target_dir)
        
        if not file_details:
            print("No safetensors files found to validate.")
        else:
            for filename, tensor_count, status in file_details:
                if "Valid" in status:
                    print(f"✅ {filename} ({tensor_count} tensors) - {status}")
                else:
                    print(f"❌ {filename} - {status}")
            
            print(f"\nValidation Results:")
            print(f"  Valid files: {valid_files}")
            print(f"  Invalid files: {invalid_files}")
            
            if invalid_files > 0:
                print(f"⚠️  Warning: {invalid_files} safetensors file(s) appear to be corrupted!")
                print("You may want to delete the corrupted files and re-run the download.")
            else:
                print("✅ All safetensors files are valid!")

# Show dependency warning if libraries are missing
if not (SAFETENSORS_AVAILABLE and TORCH_AVAILABLE):
    print("\nSkipping safetensors validation:")
    if not SAFETENSORS_AVAILABLE:
        print("  - safetensors library not available")
    if not TORCH_AVAILABLE:
        print("  - PyTorch not available")
    print("Install missing dependencies to enable validation:")
    print("  pip install safetensors torch")

# Show usage instructions based on operation mode
if choice == '2':
    print(f"\nModel verification completed for: {local_dir}")
elif choice == '3':
    print(f"\nBatch model verification completed for: {base_dir}")
    if 'model_directories' in locals() and model_directories:
        print("Valid model directories found:")
        for model_dir in model_directories:
            relative_path = os.path.relpath(model_dir, base_dir)
            if relative_path == '.':
                relative_path = os.path.basename(base_dir)
            print(f"  📁 {relative_path}")
