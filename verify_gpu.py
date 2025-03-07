"""
GPU Verification Script for PyTorch.

This script checks if CUDA is available and properly configured with PyTorch.
It also performs a simple tensor operation on the GPU to ensure it's working.
"""

import torch
import platform
import os

def print_separator():
    """Print a separator line."""
    print("-" * 60)

def print_section(title):
    """Print a section title."""
    print_separator()
    print(f"## {title}")
    print_separator()

def get_nvidia_gpu_info():
    """Try to get NVIDIA GPU info using system commands."""
    try:
        if platform.system() == "Windows":
            import subprocess
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            return result.stdout
        else:
            return "nvidia-smi command check skipped (non-Windows system)"
    except:
        return "nvidia-smi not available or failed to run"

def get_cuda_version():
    """Get CUDA version safely avoiding linter errors."""
    try:
        # Use getattr to avoid linter issues
        version_module = getattr(torch, "version", None)
        if version_module:
            return getattr(version_module, "cuda", "Unknown")
        return "Unknown"
    except:
        return "Unknown"

print_section("SYSTEM INFORMATION")
print(f"Operating System: {platform.system()} {platform.version()}")
print(f"Python Version: {platform.python_version()}")
print(f"PyTorch Version: {torch.__version__}")

print_section("CUDA AVAILABILITY")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {get_cuda_version()}")
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    print(f"Current CUDA Device: {torch.cuda.current_device()}")
    
    # Get device properties
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    print(f"GPU Name: {props.name}")
    print(f"GPU Memory: {props.total_memory / 1e9:.2f} GB")
    
    print_section("GPU TEST")
    try:
        # Create a test tensor and perform operations
        x = torch.rand(1000, 1000).cuda()
        y = torch.rand(1000, 1000).cuda()
        
        # Perform a matrix multiplication (good GPU test)
        # Create CUDA events for timing
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        z = torch.matmul(x, y)
        end_time.record()
        
        # Wait for CUDA to finish
        torch.cuda.synchronize()
        
        # Calculate elapsed time
        elapsed_time = start_time.elapsed_time(end_time)
        
        print(f"Matrix multiplication test PASSED in {elapsed_time:.2f} ms")
        
        # Force cleanup
        del x, y, z
        torch.cuda.empty_cache()
        
        print("GPU test completed successfully!")
    except Exception as e:
        print(f"GPU test failed: {e}")
else:
    print_section("GPU NOT DETECTED")
    print("Checking environment variables:")
    print(f"CUDA_PATH: {os.environ.get('CUDA_PATH', 'Not set')}")
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
    
    print("\nChecking NVIDIA driver:")
    nvidia_info = get_nvidia_gpu_info()
    print(nvidia_info if nvidia_info else "No NVIDIA driver information available")
    
    print("\nPossible solutions:")
    print("1. Install NVIDIA drivers from: https://www.nvidia.com/download/index.aspx")
    print("2. Install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads")
    print("3. Make sure you have the CUDA version of PyTorch:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --user")
    print("4. If using virtual environment, make sure it has the CUDA-enabled PyTorch")

print_section("NEXT STEPS")
if torch.cuda.is_available():
    print("Your GPU is working correctly! You can now run the training script with:")
    print("python examples/train_code_model.py --dataset code_search_net/python --batch-size 24 --use-fp16")
else:
    print("Please fix your GPU installation using the suggestions above,")
    print("then try running this verification script again.")

print_separator() 