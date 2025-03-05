#!/usr/bin/env python
"""
IdeaSpace LLM Demonstration Script

This script demonstrates the capabilities of the IdeaSpace LLM model, which combines 
a structured latent space with a diffusion-based generation process to enable concept 
manipulation and high-speed token generation.

Run this script to see examples of:
1. Basic text encoding and generation
2. Latent space arithmetic for concept blending
3. Interpolation between concepts
4. Attribute manipulation
5. Finding similar concepts

To run: python ideaspace_demo.py
"""

import sys
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoTokenizer

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import IdeaSpaceLLM
from utils import EmbeddingUtils, LatentOperations, VisualizationUtils

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="IdeaSpace LLM Demonstration")
    parser.add_argument("--model-path", type=str, default=None, 
                        help="Path to model checkpoint (optional)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda or cpu, default: auto-detect)")
    parser.add_argument("--save-plots", action="store_true",
                        help="Save plots to files")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                        help="Directory to save outputs")
    
    return parser.parse_args()

def initialize_model(args):
    """Initialize the IdeaSpace LLM model."""
    # Select device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = IdeaSpaceLLM(
        pretrained_model_name="bert-base-uncased",
        latent_dim=512,
        hidden_dim=512,
        max_seq_len=64,
        diffusion_steps=1000,
        inference_steps=30,
        use_variational=True
    )

    # Load from checkpoint if available
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        
    model = model.to(device)
    model.eval()
    return model, device

def encode_text(model, text, device):
    """Helper function to tokenize and encode text."""
    # Tokenize the text
    if not hasattr(model, 'tokenizer') or model.tokenizer is None:
        # If model doesn't have a tokenizer, create one
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        encoded = tokenizer(text, return_tensors="pt")
    else:
        # Use the model's tokenizer
        encoded = model.tokenizer(text, return_tensors="pt")
    
    # Move to device
    encoded = {k: v.to(device) for k, v in encoded.items()}
    
    # Encode to latent space
    with torch.no_grad():
        z = model.encode(encoded['input_ids'], encoded['attention_mask'])
        
    return z

def demo_basic_encoding_generation(model, device):
    """Demonstrate basic text encoding and generation."""
    print("\n" + "="*80)
    print("DEMONSTRATION 1: Basic Text Encoding and Generation")
    print("="*80)
    
    examples = [
        "The cat sat on the mat.",
        "Artificial intelligence will transform society in unexpected ways.",
        "Climate change requires global cooperation to address effectively."
    ]
    
    latent_vectors = []
    for example in examples:
        print(f"\nOriginal text: {example}")
        
        # Encode using the helper function
        z = encode_text(model, example, device)
            
        # Generate
        generated_text = model.generate(z=z, temperature=0.7)
        print(f"Generated text: {generated_text}")
        
        latent_vectors.append(z)
    
    return latent_vectors

def demo_vector_arithmetic(model, device, save_plots=False, output_dir=None):
    """Demonstrate vector arithmetic in latent space."""
    print("\n" + "="*80)
    print("DEMONSTRATION 2: Latent Space Arithmetic")
    print("="*80)
    
    # Define some interesting equations
    equations = [
        (
            "king - man + woman = queen",
            ["king", "man", "woman"],
            [1.0, -1.0, 1.0]
        ),
        (
            "paris - france + italy = rome",
            ["paris", "france", "italy"],
            [1.0, -1.0, 1.0]
        ),
        (
            "cat + big = tiger",
            ["cat", "big"],
            [1.0, 0.8]
        ),
        (
            "artificial intelligence + ethics = responsible AI",
            ["artificial intelligence", "ethics"],
            [1.0, 0.5]
        )
    ]
    
    for name, texts, coefficients in equations:
        print(f"\nEquation: {name}")
        print(f"Texts: {texts}")
        print(f"Coefficients: {coefficients}")
        
        # Encode texts to latent vectors
        vectors = []
        for text in texts:
            vectors.append(encode_text(model, text, device))
            
        # Perform arithmetic
        result_vector = LatentOperations.arithmetic(vectors, coefficients)
        
        # Generate from result
        result_text = model.generate(z=result_vector, temperature=0.7)
        print(f"Result: {result_text}")

def demo_interpolation(model, device, save_plots=False, output_dir=None):
    """Demonstrate interpolation between concepts in latent space."""
    print("\n" + "="*80)
    print("DEMONSTRATION 3: Latent Space Interpolation")
    print("="*80)
    
    # Define text pairs to interpolate between
    text_pairs = [
        ("The cat sat on the mat.", "The dog played in the yard."),
        ("Artificial intelligence will transform society.", 
         "Machine learning models require large datasets."),
        ("Climate change is a global challenge.", 
         "Local actions can have worldwide impacts.")
    ]
    
    steps = 5
    for text1, text2 in text_pairs:
        print(f"\nInterpolating between:\n'{text1}'\nand\n'{text2}'\n")
        
        # Encode both texts
        z1 = encode_text(model, text1, device)
        z2 = encode_text(model, text2, device)
        
        # Generate intermediates
        alphas = np.linspace(0, 1, steps)
        results = []
        
        for alpha in alphas:
            # Linear interpolation
            z_interp = LatentOperations.interpolate(z1, z2, alpha=alpha)
            
            # Generate from interpolated vector
            generated = model.generate(z=z_interp, temperature=0.7)
            results.append(generated)
            print(f"alpha={alpha:.2f}: {generated}")
            
        # Try spherical interpolation for the last pair
        if text1 == text_pairs[-1][0] and text2 == text_pairs[-1][1]:
            print("\nSpherical interpolation:")
            for alpha in alphas:
                # Spherical interpolation
                z_interp = LatentOperations.spherical_interpolate(z1, z2, alpha=alpha)
                
                # Generate from interpolated vector
                generated = model.generate(z=z_interp, temperature=0.7)
                print(f"alpha={alpha:.2f}: {generated}")

def compute_attribute_direction(model, positive_examples, negative_examples=None, device=None):
    """Compute an attribute direction by averaging positive examples and subtracting negative examples."""
    # Encode positive examples
    pos_vectors = []
    for text in positive_examples:
        pos_vectors.append(encode_text(model, text, device))
    
    # Average positive vectors
    pos_direction = torch.stack(pos_vectors).mean(dim=0)
    
    if negative_examples:
        # Encode negative examples
        neg_vectors = []
        for text in negative_examples:
            neg_vectors.append(encode_text(model, text, device))
        
        # Average negative vectors
        neg_direction = torch.stack(neg_vectors).mean(dim=0)
        
        # Compute direction
        direction = pos_direction - neg_direction
    else:
        direction = pos_direction
    
    # Normalize
    direction = direction / direction.norm()
    
    return direction

def demo_attribute_manipulation(model, device, save_plots=False, output_dir=None):
    """Demonstrate attribute manipulation in latent space."""
    print("\n" + "="*80)
    print("DEMONSTRATION 4: Attribute Manipulation")
    print("="*80)
    
    # Define attributes with positive and negative examples
    attributes = [
        (
            "positive_sentiment",
            ["This is excellent!", "I love this idea.", "Great work!"],
            ["This is terrible.", "I hate this concept.", "Poor execution."]
        ),
        (
            "technical_language",
            ["The algorithm computes the gradient descent optimization.", 
             "Neural networks leverage backpropagation for training.", 
             "The quantum processor utilizes superposition for parallel computation."],
            ["The computer does math really well.", 
             "The brain-like thing learns from examples.", 
             "The special chip does many things at once."]
        ),
        (
            "formality",
            ["I would like to request your consideration of the following proposal.", 
             "We hereby acknowledge receipt of your correspondence.", 
             "It is with great pleasure that we announce our annual conference."],
            ["Hey, check this out!", 
             "Got your message, thanks!", 
             "Super excited about our meetup!"]
        )
    ]
    
    # Base texts to manipulate
    base_texts = [
        "The project was completed on time.",
        "The team developed a new approach to solve the problem.",
        "We're having a meeting tomorrow to discuss the results."
    ]
    
    # Compute attribute directions
    attribute_directions = {}
    for attr_name, pos_examples, neg_examples in attributes:
        direction = compute_attribute_direction(model, pos_examples, neg_examples, device)
        attribute_directions[attr_name] = direction
        print(f"Computed direction for attribute: {attr_name}")
    
    # Apply attribute manipulations
    for base_text in base_texts:
        print(f"\nBase text: {base_text}")
        
        # Encode base text using the helper function
        base_z = encode_text(model, base_text, device)
        
        # Apply each attribute
        for attr_name, direction in attribute_directions.items():
            print(f"\nApplying attribute: {attr_name}")
            
            # Try different strengths
            strengths = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
            for strength in strengths:
                # Apply attribute
                modified_z = base_z + direction * strength
                
                # Generate from modified vector
                generated = model.generate(z=modified_z, temperature=0.7)
                print(f"Strength {strength:.1f}: {generated}")

def demo_nearest_neighbors(model, device):
    """Demonstrate finding nearest neighbors in latent space."""
    print("\n" + "="*80)
    print("DEMONSTRATION 5: Finding Similar Concepts")
    print("="*80)
    
    # Define query and candidate texts
    query_texts = [
        "Renewable energy is the future of power generation.",
        "The artificial intelligence algorithm learned from the data.",
        "Healthy diet includes fruits and vegetables."
    ]
    
    candidate_texts = [
        "Solar and wind power are sustainable energy sources.",
        "Fossil fuels contribute to climate change.",
        "Machine learning models improve with more training data.",
        "Neural networks can recognize patterns in images.",
        "A balanced nutrition plan includes proteins, carbohydrates, and fats.",
        "Regular exercise is important for cardiovascular health.",
        "Cloud computing enables scalable data processing.",
        "The stock market fluctuated dramatically last week.",
        "Green technologies reduce environmental impact.",
        "Data science combines statistics and programming.",
        "Organic produce is grown without synthetic pesticides.",
        "Virtual reality creates immersive digital experiences.",
        "Sustainable development meets present needs without compromising the future.",
        "Deep learning has revolutionized natural language processing.",
        "Nutritionists recommend eating a variety of foods."
    ]
    
    k = 3  # Number of neighbors to find
    
    # Encode all candidate texts
    candidate_vectors = []
    for text in candidate_texts:
        candidate_vectors.append(encode_text(model, text, device))
    
    # Process each query
    for query_text in query_texts:
        print(f"\nQuery: {query_text}")
        
        # Encode query
        query_vector = encode_text(model, query_text, device)
        
        # Compute cosine similarities
        similarities = []
        for i, candidate_vector in enumerate(candidate_vectors):
            similarity = EmbeddingUtils.cosine_similarity(query_vector, candidate_vector)
            similarities.append((i, similarity.item()))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Print top-k results
        print(f"Top {k} similar concepts:")
        for i in range(min(k, len(similarities))):
            idx, sim = similarities[i]
            print(f"  {candidate_texts[idx]} (similarity: {sim:.4f})")

def demo_visualization(model, device, save_plots=False, output_dir=None):
    """Demonstrate visualization of the latent space."""
    print("\n" + "="*80)
    print("DEMONSTRATION 6: Visualizing Latent Space")
    print("="*80)
    
    # Generate examples for visualization
    categories = [
        "The cat sat on the mat.", 
        "Dogs are friendly pets.",
        "Artificial intelligence is advancing rapidly.",
        "Machine learning models require large datasets.",
        "Climate change is a global challenge.",
        "Ocean pollution affects marine ecosystems."
    ]
    
    variations_per_example = 2
    all_examples = []
    all_latents = []
    all_labels = []
    
    for i, base in enumerate(categories):
        all_examples.append(base)
        z = encode_text(model, base, device)
        all_latents.append(z.cpu().numpy())
        all_labels.append(f"Base {i+1}")
        
        # Generate variations
        for j in range(variations_per_example):
            # Add noise to the latent vector
            z_noisy = LatentOperations.add_noise(z, scale=0.1)
            variation = model.generate(z=z_noisy, temperature=0.8)
            
            all_examples.append(variation)
            # Re-encode to get the proper latent
            z_var = encode_text(model, variation, device)
            all_latents.append(z_var.cpu().numpy())
            all_labels.append(f"Var {i+1}.{j+1}")
    
    # Convert list of arrays to a single numpy array
    latent_array = np.vstack(all_latents)
    
    # Print examples and their variations
    for i in range(0, len(all_examples), variations_per_example + 1):
        base_idx = i
        var_indices = range(i+1, min(i+variations_per_example+1, len(all_examples)))
        
        print(f"\nBase example: {all_examples[base_idx]}")
        for j, var_idx in enumerate(var_indices):
            print(f"  Variation {j+1}: {all_examples[var_idx]}")
    
    print("\nVisualization would normally show a 2D plot of these vectors.")
    print("To see visualizations, run with --save-plots or in a notebook environment.")
    
    # If save_plots is True, generate and save the plots
    if save_plots and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # PCA plot
        plt.figure(figsize=(12, 10))
        fig = VisualizationUtils.plot_embeddings_2d(
            embeddings=latent_array, 
            labels=all_labels, 
            method="pca",
            figsize=(12, 10),
            title="PCA Visualization of Latent Space"
        )
        plt.savefig(os.path.join(output_dir, "pca_visualization.png"))
        plt.close()
        
        # t-SNE plot
        plt.figure(figsize=(12, 10))
        fig = VisualizationUtils.plot_embeddings_2d(
            embeddings=latent_array, 
            labels=all_labels, 
            method="tsne",
            figsize=(12, 10),
            title="t-SNE Visualization of Latent Space"
        )
        plt.savefig(os.path.join(output_dir, "tsne_visualization.png"))
        plt.close()
        
        print(f"Plots saved to {output_dir}")

def patch_model(model):
    """Patch the model to fix the _embeddings_to_tokens method."""
    
    def patched_embeddings_to_tokens(self, embeddings, temperature=1.0, top_k=None, top_p=None):
        """
        Convert embeddings to tokens using nearest neighbor lookup in the embedding table.
        
        Args:
            embeddings (torch.Tensor): Generated embeddings [batch_size, seq_len, embedding_dim]
            temperature (float): Sampling temperature
            top_k (int, optional): Top-k sampling
            top_p (float, optional): Nucleus sampling threshold
            
        Returns:
            torch.Tensor: Token IDs [batch_size, seq_len]
        """
        batch_size, seq_len, embedding_dim = embeddings.shape
        
        # Get the full embedding matrix
        embedding_matrix = self.token_embedding.weight  # [vocab_size, embedding_dim]
        
        # Calculate similarity between generated embeddings and vocabulary embeddings
        # Reshape embeddings for batch similarity calculation
        embeddings_flat = embeddings.reshape(-1, embedding_dim)  # [batch_size * seq_len, embedding_dim]
        
        # Calculate cosine similarity - FIXED VERSION
        # Normalize the embeddings
        normalized_embeddings = embeddings_flat / embeddings_flat.norm(dim=-1, keepdim=True)
        normalized_embedding_matrix = embedding_matrix / embedding_matrix.norm(dim=-1, keepdim=True)
        
        # Compute similarity
        similarity = torch.matmul(normalized_embeddings, normalized_embedding_matrix.T)
        
        # Apply temperature
        logits = similarity / temperature
        
        # Apply top-k sampling if specified
        if top_k is not None:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Apply nucleus (top-p) sampling if specified
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            
            # Shift the indices to the right to keep the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        # Sample from the distribution
        probs = torch.softmax(logits, dim=-1)
        token_ids = torch.multinomial(probs, 1).squeeze(-1)
        
        # Reshape back to [batch_size, seq_len]
        token_ids = token_ids.reshape(batch_size, seq_len)
        
        return token_ids
    
    # Replace the method
    import types
    model._embeddings_to_tokens = types.MethodType(patched_embeddings_to_tokens, model)
    
    return model

def main():
    """Run the demonstration."""
    args = parse_args()
    
    # Create output directory if needed
    if args.save_plots:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model
    model, device = initialize_model(args)
    
    # Patch the model to fix the _embeddings_to_tokens method
    model = patch_model(model)
    
    try:
        # Run demonstrations
        demo_basic_encoding_generation(model, device)
        demo_vector_arithmetic(model, device, args.save_plots, args.output_dir)
        demo_interpolation(model, device, args.save_plots, args.output_dir)
        demo_attribute_manipulation(model, device, args.save_plots, args.output_dir)
        demo_nearest_neighbors(model, device)
        demo_visualization(model, device, args.save_plots, args.output_dir)
        
        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETE")
        print("="*80)
        print("\nThese demonstrations showcase how the model's structured latent space")
        print("enables powerful semantic operations and concept manipulation, which")
        print("can be valuable for creative text generation, semantic search, and")
        print("other natural language processing applications.")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 