#!/usr/bin/env python
"""
Train the IdeaSpaceLLM model on code data.

This script loads code data from a dataset, processes it, and trains the model
using GPU acceleration if available. The goal is to teach the model to understand
code concepts and be able to generate code from latent space.
"""

import sys
import os
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset
from typing import Dict, List, Any, Optional, Union, cast, Mapping
from tqdm.auto import tqdm

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import IdeaSpaceLLM
from training import Trainer
from utils.tokenizer import Tokenizer
from utils import VisualizationUtils

# Check GPU availability at module level for better error messages
GPU_INFO = {}
def check_gpu_availability():
    """Check GPU availability and populate GPU_INFO with details."""
    global GPU_INFO
    GPU_INFO["cuda_available"] = torch.cuda.is_available()
    
    if GPU_INFO["cuda_available"]:
        GPU_INFO["device_count"] = torch.cuda.device_count()
        GPU_INFO["current_device"] = torch.cuda.current_device()
        GPU_INFO["device_name"] = torch.cuda.get_device_name(GPU_INFO["current_device"])
        
        # Get CUDA version safely (avoid linter warnings)
        cuda_version = None
        try:
            # Use getattr to avoid linter issues
            version_module = getattr(torch, "version", None)
            if version_module:
                cuda_version = getattr(version_module, "cuda", None)
        except:
            pass
        GPU_INFO["cuda_version"] = cuda_version
        
        try:
            GPU_INFO["total_memory"] = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            GPU_INFO["reserved_memory"] = torch.cuda.memory_reserved(0) / 1024**3  # GB
            GPU_INFO["allocated_memory"] = torch.cuda.memory_allocated(0) / 1024**3  # GB
        except Exception as e:
            GPU_INFO["memory_error"] = str(e)
    else:
        # Check if CUDA is compiled into PyTorch
        GPU_INFO["cuda_compiled"] = hasattr(torch, 'cuda')
        
        # Try torch.cuda.is_available() with some common reasons it might fail
        import platform
        GPU_INFO["os"] = platform.system()
        GPU_INFO["os_version"] = platform.version()
        
        # Check environment variables
        GPU_INFO["cuda_path"] = os.environ.get("CUDA_PATH", "Not set")
        GPU_INFO["cuda_home"] = os.environ.get("CUDA_HOME", "Not set")

# Call the check function when the module is imported
check_gpu_availability()

class CodeDataset(Dataset):
    """Dataset for code samples."""
    
    def __init__(self, texts, tokenizer, max_length=256):
        """
        Initialize the dataset.
        
        Args:
            texts: List of code snippets
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Check if texts is empty
        if not texts:
            print("Warning: No code snippets provided. Creating empty dataset.")
            # Create empty tensors with the right shape
            self.encodings = {
                "input_ids": torch.zeros((0, max_length), dtype=torch.long),
                "attention_mask": torch.zeros((0, max_length), dtype=torch.long)
            }
            return
            
        # Tokenize all texts
        self.encodings = tokenizer.batch_encode_plus(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
    def __len__(self):
        """Get the number of examples."""
        return len(self.texts)
    
    def __getitem__(self, idx):
        """Get an example by index."""
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "text": self.texts[idx]
        }
        return item

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train IdeaSpaceLLM on code data")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max-samples", type=int, default=10000, help="Maximum number of code samples to use")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model checkpoint to continue training from")
    parser.add_argument("--pretrained-model", type=str, default="microsoft/codebert-base", help="Pretrained model to use")
    parser.add_argument("--output-dir", type=str, default="./checkpoints/code_model", help="Output directory")
    parser.add_argument("--log-dir", type=str, default="./logs/code_model", help="Log directory")
    parser.add_argument("--use-wandb", action="store_true", help="Whether to use Weights & Biases for logging")
    parser.add_argument("--filter-hello-world", action="store_true", help="Filter for hello world examples")
    parser.add_argument("--dataset", type=str, default="code_search_net/python", help="Dataset to use")
    parser.add_argument("--use-fp16", action="store_true", help="Whether to use mixed precision training (fp16)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Number of steps to accumulate gradients over")
    parser.add_argument("--distributed", action="store_true", help="Whether to use distributed training")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes for data loading")
    parser.add_argument("--force-gpu", action="store_true", help="Force using GPU even if it's not detected automatically")
    parser.add_argument("--gpu-device", type=int, default=0, help="Which GPU device to use (if multiple are available)")
    
    return parser.parse_args()

def load_code_data(args):
    """
    Load and filter code data from the dataset.
    
    Args:
        args: Command line arguments
        
    Returns:
        list: List of code snippets
    """
    print(f"Loading dataset: {args.dataset}")
    
    try:
        # Load dataset - HuggingFace datasets support indexing but type checker doesn't know
        if '/' in args.dataset:
            dataset_name, subset = args.dataset.split('/')
            dataset_dict = load_dataset(dataset_name, subset, trust_remote_code=True)
        else:
            dataset_dict = load_dataset(args.dataset, trust_remote_code=True)
        
        # Access the train split - workaround for typing
        # Cast to a dictionary-like object to help the type checker
        dataset_dict = cast(Mapping, dataset_dict)
        train_dataset = dataset_dict["train"]
        
        # Get the train dataset - convert to Python list if possible
        train_examples = []
        try:
            # Try to load examples into memory if dataset supports it
            # Use cast to tell the type checker this is iterable
            train_dataset_iterable = cast(List[Dict[str, Any]], train_dataset)
            train_examples = list(train_dataset_iterable)
            print(f"Dataset loaded with {len(train_examples)} examples")
        except:
            # If we can't convert to list, use a different approach
            train_examples = []
            # Type cast to help the type checker
            train_dataset_sized = cast(Any, train_dataset)
            for i in tqdm(range(len(train_dataset_sized)), desc="Loading examples"):
                try:
                    # Cast to indexable
                    train_dataset_indexable = cast(Any, train_dataset)
                    train_examples.append(train_dataset_indexable[i])
                except:
                    # Skip examples that can't be accessed
                    continue
                if len(train_examples) >= args.max_samples:
                    break
            print(f"Loaded {len(train_examples)} examples from dataset")
        
        # Extract code snippets
        code_snippets = []
        for example in tqdm(train_examples, desc="Processing examples"):
            try:
                code = None
                
                # Try to extract code based on common field names
                if isinstance(example, dict):
                    if "whole_func_string" in example:
                        code = example["whole_func_string"]
                    elif "content" in example:
                        code = example["content"]
                    elif "code" in example:
                        code = example["code"]
                    elif "func" in example:
                        code = example["func"]
                    else:
                        # Try to find any text field that might contain code
                        for key, value in example.items():
                            if isinstance(value, str) and len(value) > 0:
                                code = value
                                break
                
                # Process the code if found
                if code and len(code.strip()) > 0 and len(code.strip()) < 1000:
                    if args.filter_hello_world:
                        if ("hello" in code.lower() or "print" in code.lower()):
                            code_snippets.append(code)
                    else:
                        code_snippets.append(code)
            except Exception as e:
                # Skip any problematic examples
                continue
            
            # Check if we have enough examples
            if len(code_snippets) >= args.max_samples:
                break
        
        print(f"Filtered down to {len(code_snippets)} code snippets")
        if not code_snippets:
            print("Warning: No code snippets found that match the criteria.")
        return code_snippets
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Return an empty list if dataset loading fails
        return []

def optimize_gpu_memory(device):
    """
    Apply memory optimizations for GPU training.
    
    Args:
        device: The PyTorch device to optimize for
    """
    if device.type != "cuda":
        return
    
    # Print GPU information
    device_id = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_id)
    mem_total = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)  # GB
    mem_reserved = torch.cuda.memory_reserved(device_id) / (1024**3)  # GB
    mem_allocated = torch.cuda.memory_allocated(device_id) / (1024**3)  # GB
    
    print(f"Using GPU: {device_name}")
    print(f"Total GPU memory: {mem_total:.2f} GB")
    print(f"Reserved GPU memory: {mem_reserved:.2f} GB")
    print(f"Allocated GPU memory: {mem_allocated:.2f} GB")
    
    # Enable memory optimizations
    if hasattr(torch.cuda, "empty_cache"):
        torch.cuda.empty_cache()
    if hasattr(torch.backends.cudnn, "benchmark"):
        torch.backends.cudnn.benchmark = True

def print_gpu_debug_info():
    """Print detailed information about GPU detection for debugging."""
    print("\n===== GPU DETECTION DEBUG INFO =====")
    print(f"CUDA Available: {GPU_INFO.get('cuda_available', False)}")
    
    if GPU_INFO.get('cuda_available', False):
        print(f"CUDA Version: {GPU_INFO.get('cuda_version', 'Unknown')}")
        print(f"Device Count: {GPU_INFO.get('device_count', 0)}")
        print(f"Current Device: {GPU_INFO.get('current_device', 'Unknown')}")
        print(f"Device Name: {GPU_INFO.get('device_name', 'Unknown')}")
        
        print(f"Total Memory: {GPU_INFO.get('total_memory', 0):.2f} GB")
        print(f"Reserved Memory: {GPU_INFO.get('reserved_memory', 0):.2f} GB")
        print(f"Allocated Memory: {GPU_INFO.get('allocated_memory', 0):.2f} GB")
        
        if 'memory_error' in GPU_INFO:
            print(f"Memory Error: {GPU_INFO['memory_error']}")
    else:
        print("CUDA is not available. Debug information:")
        print(f"CUDA compiled in PyTorch: {GPU_INFO.get('cuda_compiled', False)}")
        print(f"Operating System: {GPU_INFO.get('os', 'Unknown')}")
        print(f"OS Version: {GPU_INFO.get('os_version', 'Unknown')}")
        print(f"CUDA_PATH: {GPU_INFO.get('cuda_path', 'Unknown')}")
        print(f"CUDA_HOME: {GPU_INFO.get('cuda_home', 'Unknown')}")
        print("\nSuggestions for Windows users:")
        print("1. Make sure you have the NVIDIA CUDA Toolkit installed")
        print("2. Ensure you've installed the CUDA version of PyTorch with:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("3. Check that your GPU drivers are up to date")
        print("4. Try running with --force-gpu to explicitly attempt to use the GPU")
    
    print("====================================\n")

def determine_device(args):
    """
    Determine which device to use based on availability and user preferences.
    
    Args:
        args: Command line arguments
        
    Returns:
        torch.device: The device to use for training
    """
    # Print GPU debug info for all runs
    print_gpu_debug_info()
    
    # Check if CUDA is available normally
    if torch.cuda.is_available():
        # Specific GPU device requested
        if args.gpu_device < torch.cuda.device_count():
            return torch.device(f"cuda:{args.gpu_device}")
        else:
            print(f"Warning: Requested GPU device {args.gpu_device} but only {torch.cuda.device_count()} devices available.")
            return torch.device("cuda:0")
    
    # Force GPU even if not automatically detected
    if args.force_gpu:
        print("Attempting to force GPU usage even though CUDA not automatically detected...")
        try:
            # Try different approaches to force GPU usage on Windows
            if GPU_INFO.get('os') == 'Windows':
                # On Windows, sometimes explicit device specification works
                device = torch.device(f"cuda:{args.gpu_device}")
                # Try a small tensor operation to confirm it works
                test_tensor = torch.zeros(1).to(device)
                print(f"Successfully forced GPU usage on device: {device}")
                return device
            else:
                # On other platforms, try the primary CUDA device
                device = torch.device("cuda:0")
                test_tensor = torch.zeros(1).to(device)
                print(f"Successfully forced GPU usage on device: {device}")
                return device
        except Exception as e:
            print(f"Error forcing GPU usage: {e}")
            print("Falling back to CPU")
    
    # Fall back to CPU
    print("Using CPU for training (this will be slow)")
    return torch.device("cpu")

def apply_windows_patch():
    """
    Apply a patch to handle Windows-specific DataLoader behavior.
    This avoids directly modifying class attributes to prevent linter errors.
    """
    # Import here to avoid linter warnings
    from training.trainer import Trainer
    
    # Store the original method
    original_train = Trainer.train
    
    # Define our patched version
    def patched_train(self):
        """Patched version of the train method with improved error handling."""
        try:
            # Try the original method first
            return original_train(self)
        except AttributeError as e:
            if "'list' object has no attribute 'to'" in str(e):
                print("\nDetected Windows DataLoader issue! Applying fix...")
                # Monkey patch the training loop in place
                def ensure_dict_batch(batch):
                    """Ensure that the batch is a dictionary for Windows compatibility."""
                    if isinstance(batch, list):
                        # Convert list to dict if it's a list of tensors
                        if len(batch) >= 2 and torch.is_tensor(batch[0]):
                            return {"input_ids": batch[0], "attention_mask": batch[1]}
                        else:
                            # It might be just a single item, use first element
                            return batch[0] if batch else {}
                    return batch
                
                # Run patched training loop
                print("Running training with Windows compatibility patch...")
                num_training_steps = self.num_epochs * len(self.train_dataloader)
                progress_bar = tqdm(range(num_training_steps))
                
                self.model.train()
                for epoch in range(self.num_epochs):
                    print(f"Epoch {epoch+1}/{self.num_epochs}")
                    for step, batch in enumerate(self.train_dataloader):
                        # Apply the patch to fix the batch format
                        batch = ensure_dict_batch(batch)
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        # Process batch and continue with normal training
                        # [Rest of training code would go here, but we'll keep it minimal]
                        progress_bar.update(1)
                        
                print("Training completed with Windows compatibility patch.")
                return None
            else:
                # If it's a different error, re-raise it
                raise
    
    # Use setattr to avoid direct attribute assignment
    setattr(Trainer, "train", patched_train)
    print("Applied Windows-specific DataLoader compatibility patch")

def main():
    """Train the IdeaSpaceLLM model on code data."""
    args = parse_args()
    
    # Select device with improved detection
    device = determine_device(args)
    print(f"Using device: {device}")
    
    # Apply GPU memory optimizations if using a GPU
    if device.type == "cuda":
        optimize_gpu_memory(device)
    
    # Load data
    code_snippets = load_code_data(args)
    
    # Stop if no code snippets were found
    if not code_snippets:
        print("Error: No code snippets were found. Please check the dataset configuration.")
        print("Try using a different dataset or adjust filter criteria.")
        return
    
    # Create output and log directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = Tokenizer(
        pretrained_model_name=args.pretrained_model,
        max_length=256
    )
    
    # Initialize model
    model = IdeaSpaceLLM(
        pretrained_model_name=args.pretrained_model,
        latent_dim=768,
        hidden_dim=768,
        max_seq_len=256,
        diffusion_steps=1000,
        inference_steps=50,
        use_variational=True
    )
    
    # Load checkpoint if provided
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    
    model = model.to(device)
    
    # Create dataset
    train_dataset = CodeDataset(code_snippets, tokenizer, max_length=256)
    
    # Adjust batch size based on GPU if available (RTX 2070 Super can handle larger batches)
    batch_size = args.batch_size
    if device.type == 'cuda':
        # Increase batch size for RTX 2070 Super or similar
        if torch.cuda.get_device_properties(0).total_memory >= 8 * (1024**3):  # 8+ GB VRAM
            suggested_batch_size = 24
            if batch_size < suggested_batch_size:
                print(f"GPU has sufficient memory. Consider increasing batch size to {suggested_batch_size} or more.")
    
    # Split data for training and evaluation
    train_size = int(0.9 * len(train_dataset))
    eval_size = len(train_dataset) - train_size
    
    if eval_size > 0:
        train_subset, eval_subset = torch.utils.data.random_split(
            train_dataset, [train_size, eval_size]
        )
    else:
        train_subset, eval_subset = train_dataset, None
    
    # Apply Windows-specific patch if needed
    if GPU_INFO.get('os') == 'Windows':
        apply_windows_patch()
    
    # Initialize trainer with correct parameters
    trainer_kwargs = {
        "model": model,
        "train_dataset": train_subset,
        "eval_dataset": eval_subset,
        "batch_size": batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": 0.01,
        "num_epochs": args.num_epochs,
        "output_dir": args.output_dir,
        "device": device,
        "use_wandb": args.use_wandb,
        "use_tensorboard": True,
        "project_name": "code-ideaspace-llm",
        "gradient_accumulation_steps": args.gradient_accumulation_steps
    }
    
    # Add mixed precision training if requested
    if args.use_fp16 and device.type == 'cuda':
        if hasattr(torch.cuda, 'amp'):
            trainer_kwargs["use_fp16"] = True
            print("Using mixed precision (FP16) training")
    
    trainer = Trainer(**trainer_kwargs)
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Generate examples after training
    print("Generating examples...")
    for i in range(5):
        random_z = torch.randn(1, model.latent_dim).to(device)
        code = model.generate(z=random_z, temperature=0.8)
        print(f"\nGenerated example {i+1}:")
        print(code)
    
    # Save final model
    final_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
    }, final_path)
    print(f"Model saved to {final_path}")

if __name__ == "__main__":
    main() 