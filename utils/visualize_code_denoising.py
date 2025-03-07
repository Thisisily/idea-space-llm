"""
Visualization utilities for code denoising process.

This module provides functions to visualize how the IdeaSpaceLLM model
converts noise into structured code through the denoising process.
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Optional, Tuple, Any, Union

# Optional imports - handled with proper type comments
HAS_IPYTHON = False
try:
    # type: ignore[import] # Ignore import errors for IPython
    from IPython.display import display, HTML  # type: ignore
    from IPython import get_ipython  # type: ignore
    HAS_IPYTHON = True
except ImportError:
    # Define dummy functions/classes if IPython is not available
    def display(obj: Any) -> None:
        """Dummy display function when IPython is not available."""
        pass
        
    class HTML:  # type: ignore
        """Dummy HTML class when IPython is not available."""
        def __init__(self, data: str) -> None:
            self.data = data

def visualize_code_denoising(model, tokenizer, num_steps=10, temperature=0.8, seed=42, 
                            output_dir=None, starting_text=None, max_length=256):
    """
    Visualize the denoising process of the model generating code from noise.
    
    Args:
        model: The IdeaSpaceLLM model
        tokenizer: The tokenizer used with the model
        num_steps: Number of denoising steps to visualize
        temperature: Temperature for generation
        seed: Random seed for reproducibility
        output_dir: Directory to save visualization images (if None, displays inline)
        starting_text: Optional starting text as a condition
        max_length: Maximum sequence length
        
    Returns:
        List of generated texts at each step
    """
    # Set the seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Get device
    device = next(model.parameters()).device
    
    # Generate random noise in the latent space
    z = torch.randn(1, model.latent_dim).to(device)
    
    # Prepare starting hidden state if provided
    if starting_text:
        # Encode the starting text
        encoded = tokenizer.encode_plus(
            starting_text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
        
        # Get the hidden state for this text
        with torch.no_grad():
            hidden_state = model.encoder_model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"]
            )[0]
        
        # Use this as conditioning
        condition = model.encoder_projection(hidden_state)
        condition = condition[:, 0]  # Take CLS token embedding
    else:
        condition = None
    
    # Calculate how many diffusion steps to use for each visualization step
    if num_steps > model.inference_steps:
        num_steps = model.inference_steps
    
    step_size = max(1, model.inference_steps // num_steps)
    visualization_steps = list(range(0, model.inference_steps, step_size))
    if model.inference_steps - 1 not in visualization_steps:
        visualization_steps.append(model.inference_steps - 1)
    
    # Generate text at each step
    generated_texts = []
    step_temperatures = []
    
    # Create figure for visualization
    fig, axes = plt.subplots(len(visualization_steps), 1, figsize=(12, len(visualization_steps) * 3))
    if len(visualization_steps) == 1:
        axes = [axes]
    
    for i, step in enumerate(visualization_steps):
        with torch.no_grad():
            # Override the inference steps with current step
            temp_inference_steps = step + 1
            
            # Adjust temperature based on step (higher noise = higher temp)
            step_temp = temperature * (1.0 - (step / model.inference_steps))
            step_temp = max(0.3, step_temp)  # Clamp to minimum of 0.3
            step_temperatures.append(step_temp)
            
            # Generate text at this step
            generated = model.generate(
                z=z.clone(),
                condition=condition,
                inference_steps=temp_inference_steps,
                temperature=step_temp
            )
            
            generated_texts.append(generated)
            
            # Visualize the code with syntax highlighting
            ax = axes[i]
            ax.axis('off')
            
            # Calculate completion percentage
            completion_pct = step / model.inference_steps * 100
            
            # Create text colored based on generation progress
            color_map = np.linspace(0, 1, len(generated))
            cmap = LinearSegmentedColormap.from_list('RdYlGn', ['#ffcccc', '#66ff66'])
            
            # Simple text display with step information
            ax.text(0, 0.5, 
                   f"Step {step}/{model.inference_steps} ({completion_pct:.1f}%) - Temp: {step_temp:.2f}\n\n{generated}",
                   fontsize=10, fontfamily='monospace', verticalalignment='center',
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=1'))
    
    # Add overall title
    plt.suptitle("Code Generation Denoising Process", fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    
    # Save or display the figure
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"code_denoising_process_{seed}.png"), dpi=150)
    else:
        plt.show()
    
    # If in a notebook, we can display a more interactive view
    if HAS_IPYTHON:
        try:
            ipython_instance = get_ipython()
            if ipython_instance is not None:
                display_html_visualization(generated_texts, visualization_steps, model.inference_steps, step_temperatures)
        except (NameError, RuntimeError):
            pass
    
    return generated_texts

def display_html_visualization(texts, steps, total_steps, temperatures):
    """
    Display an HTML visualization of the denoising process.
    
    Args:
        texts: List of generated texts at each step
        steps: List of step numbers
        total_steps: Total number of steps in the diffusion process
        temperatures: List of temperatures used at each step
    """
    if not HAS_IPYTHON:
        return
    
    html = """
    <style>
    .denoising-container {
        font-family: monospace;
        width: 100%;
        margin: 20px 0;
    }
    .denoising-step {
        margin-bottom: 15px;
        border: 1px solid #ddd;
        border-radius: 5px;
        overflow: hidden;
    }
    .step-header {
        background: #f5f5f5;
        padding: 8px;
        font-weight: bold;
        border-bottom: 1px solid #ddd;
    }
    .progress-bar {
        height: 5px;
        background: linear-gradient(to right, #ff6b6b 0%, #ffdd59 50%, #32ff7e 100%);
    }
    .code-block {
        padding: 10px;
        white-space: pre-wrap;
        word-wrap: break-word;
        background: #272822;
        color: #f8f8f2;
        overflow-x: auto;
    }
    </style>
    <div class="denoising-container">
        <h2>Code Generation Denoising Process</h2>
    """
    
    for i, (text, step, temp) in enumerate(zip(texts, steps, temperatures)):
        progress = (step / total_steps) * 100
        html += f"""
        <div class="denoising-step">
            <div class="step-header">Step {step}/{total_steps} ({progress:.1f}%) - Temperature: {temp:.2f}</div>
            <div class="progress-bar" style="width: {progress}%;"></div>
            <pre class="code-block">{text}</pre>
        </div>
        """
    
    html += "</div>"
    display(HTML(html))

def compare_code_generations(model, tokenizer, num_samples=5, seed=42, output_dir=None):
    """
    Generate multiple code samples and compare them.
    
    Args:
        model: The IdeaSpaceLLM model
        tokenizer: The tokenizer used with the model
        num_samples: Number of samples to generate
        seed: Starting seed for reproducibility
        output_dir: Directory to save visualization images
        
    Returns:
        List of generated code samples
    """
    # Set initial seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = next(model.parameters()).device
    samples = []
    
    # Create figure for visualization
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, num_samples * 3))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        # Use a different seed for each sample
        current_seed = seed + i
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        
        # Generate random noise
        z = torch.randn(1, model.latent_dim).to(device)
        
        # Generate code
        with torch.no_grad():
            generated = model.generate(z=z, temperature=0.8)
            samples.append(generated)
            
            # Visualize
            ax = axes[i]
            ax.axis('off')
            ax.text(0, 0.5, f"Sample {i+1} (seed={current_seed}):\n\n{generated}",
                   fontsize=10, fontfamily='monospace', verticalalignment='center',
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=1'))
    
    # Add title
    plt.suptitle("Multiple Code Generation Samples", fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    
    # Save or display
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"code_generation_samples.png"), dpi=150)
    else:
        plt.show()
    
    return samples 