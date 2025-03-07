#!/usr/bin/env python
"""
Code Generation Demo for IdeaSpaceLLM.

This script demonstrates how to use the IdeaSpaceLLM model to generate code samples
and visualize the denoising process. It loads a trained model and generates
various code examples, showing the step-by-step denoising process.
"""

import sys
import os
import torch
import argparse
from tqdm import tqdm

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import IdeaSpaceLLM
from utils.tokenizer import Tokenizer
from utils.visualize_code_denoising import visualize_code_denoising, compare_code_generations

# Example prompts for code generation
EXAMPLE_PROMPTS = {
    "python_hello": "# A Python hello world program",
    "python_sort": "# A Python function to sort a list using quicksort",
    "python_class": "# A Python class for a simple bank account",
    "java_hello": "// A Java hello world program",
    "javascript_hello": "// A JavaScript hello world function",
    "cpp_hello": "// A C++ hello world program",
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Code Generation Demo for IdeaSpaceLLM")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--pretrained-model", type=str, default="microsoft/codebert-base", 
                        help="Name of the pretrained model to use")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Directory to save outputs")
    parser.add_argument("--num-examples", type=int, default=5, help="Number of examples to generate")
    parser.add_argument("--num-steps", type=int, default=10, help="Number of denoising steps to visualize")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--show-unconditional", action="store_true", help="Show unconditional generation")
    parser.add_argument("--show-conditional", action="store_true", help="Show conditional generation")
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt for conditional generation")
    
    return parser.parse_args()

def load_model(args):
    """Load the trained model."""
    print(f"Loading model from {args.model_path}")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize the model
    model = IdeaSpaceLLM(
        pretrained_model_name=args.pretrained_model,
        latent_dim=768,
        hidden_dim=768,
        max_seq_len=args.max_length,
        diffusion_steps=1000,
        inference_steps=50,
        use_variational=True
    )
    
    # Load the checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    # Initialize the tokenizer
    tokenizer = Tokenizer(
        pretrained_model_name=args.pretrained_model,
        max_length=args.max_length
    )
    
    return model, tokenizer

def unconditional_generation_demo(model, tokenizer, args):
    """Demonstrate unconditional code generation."""
    print("\n" + "="*80)
    print("UNCONDITIONAL CODE GENERATION DEMO")
    print("="*80)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate multiple examples and compare them
    print("\nGenerating and comparing multiple code samples...")
    samples = compare_code_generations(
        model=model,
        tokenizer=tokenizer,
        num_samples=args.num_examples,
        seed=args.seed,
        output_dir=args.output_dir
    )
    
    # Visualize the denoising process
    print("\nVisualizing the denoising process...")
    generated_texts = visualize_code_denoising(
        model=model,
        tokenizer=tokenizer,
        num_steps=args.num_steps,
        temperature=args.temperature,
        seed=args.seed,
        output_dir=args.output_dir
    )
    
    return samples, generated_texts

def conditional_generation_demo(model, tokenizer, args):
    """Demonstrate conditional code generation."""
    print("\n" + "="*80)
    print("CONDITIONAL CODE GENERATION DEMO")
    print("="*80)
    
    prompts = EXAMPLE_PROMPTS.copy()
    
    # Add custom prompt if provided
    if args.prompt:
        prompts["custom"] = args.prompt
    
    # Generate code for each prompt
    for name, prompt in prompts.items():
        print(f"\nGenerating code for prompt: '{prompt}'")
        
        # Visualize denoising process with this prompt
        output_path = os.path.join(args.output_dir, f"denoising_{name}")
        os.makedirs(output_path, exist_ok=True)
        
        generated_texts = visualize_code_denoising(
            model=model,
            tokenizer=tokenizer,
            num_steps=args.num_steps,
            temperature=args.temperature,
            seed=args.seed,
            output_dir=output_path,
            starting_text=prompt
        )
        
        # Print the final result
        print("\nFinal generated code:")
        print("="*40)
        print(generated_texts[-1])
        print("="*40)
    
    return None

def main():
    """Run the code generation demo."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Load model
    model, tokenizer = load_model(args)
    
    # Run demos
    if args.show_unconditional or (not args.show_unconditional and not args.show_conditional):
        unconditional_generation_demo(model, tokenizer, args)
    
    if args.show_conditional or (not args.show_unconditional and not args.show_conditional):
        conditional_generation_demo(model, tokenizer, args)
    
    print("\nDemo completed. Results saved to:", args.output_dir)

if __name__ == "__main__":
    main() 