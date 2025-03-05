#!/usr/bin/env python
"""
Debug script to understand the model architecture and dimensions.
"""

import sys
import os
import torch
from transformers import AutoTokenizer

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import IdeaSpaceLLM

def main():
    """Initialize the model and print out dimensions."""
    # Select device
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
    
    model = model.to(device)
    model.eval()
    
    # Print model dimensions
    print("\nModel dimensions:")
    print(f"Latent dimension: {model.latent_dim}")
    print(f"Token embedding shape: {model.token_embedding.weight.shape}")
    
    # Tokenize a sample text
    text = "The cat sat on the mat."
    print(f"\nSample text: {text}")
    
    encoded = model.tokenizer(text, return_tensors="pt").to(device)
    print(f"Tokenized shape: {encoded.input_ids.shape}")
    
    # Encode to latent space
    with torch.no_grad():
        z = model.encode(encoded.input_ids, encoded.attention_mask)
    print(f"Latent vector shape: {z.shape}")
    
    # Try to generate
    try:
        # Generate via the diffusion process
        shape = (1, model.max_seq_len, model.token_embedding.embedding_dim)
        print(f"Target shape for diffusion: {shape}")
        
        # Use fewer steps for inference
        timesteps = list(range(0, model.diffusion_steps, model.diffusion_steps // model.inference_steps))
        
        # Generate embeddings
        print("\nGenerating embeddings...")
        generated_embeddings = model.noise_process.p_sample_loop(
            model=model.decoder,
            shape=shape,
            z=z,
            device=device,
            timesteps=timesteps,
            progress=True,
            model_output_type="x0" if model.predict_x0 else "noise"
        )
        print(f"Generated embeddings shape: {generated_embeddings.shape}")
        
        # Print embedding dimensions
        print("\nEmbedding dimensions:")
        print(f"Generated embeddings: {generated_embeddings.shape}")
        print(f"Token embedding weight: {model.token_embedding.weight.shape}")
        
        # Reshape embeddings for batch similarity calculation
        embeddings_flat = generated_embeddings.reshape(-1, shape[2])
        print(f"Flattened embeddings: {embeddings_flat.shape}")
        
        # Get the embedding matrix
        embedding_matrix = model.token_embedding.weight
        print(f"Embedding matrix: {embedding_matrix.shape}")
        
        # Calculate norms
        embeddings_norm = embeddings_flat.norm(dim=-1, keepdim=True)
        embedding_matrix_norm = embedding_matrix.norm(dim=-1, keepdim=True)
        print(f"Embeddings norm: {embeddings_norm.shape}")
        print(f"Embedding matrix norm: {embedding_matrix_norm.shape}")
        
        # Try the calculation that's failing
        print("\nTrying the calculation that's failing...")
        normalized_embeddings = embeddings_flat / embeddings_norm
        normalized_embedding_matrix = embedding_matrix / embedding_matrix_norm
        print(f"Normalized embeddings: {normalized_embeddings.shape}")
        print(f"Normalized embedding matrix: {normalized_embedding_matrix.shape}")
        
        # This is where it fails - we need to transpose the embedding matrix
        similarity = torch.matmul(normalized_embeddings, normalized_embedding_matrix.T)
        print(f"Similarity matrix: {similarity.shape}")
        
        print("\nSuccess! The calculation works when we transpose the embedding matrix correctly.")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 