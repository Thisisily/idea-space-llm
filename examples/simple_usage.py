import torch
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import IdeaSpaceLLM
from utils.visualization import VisualizationUtils
from utils.latent_operations import LatentOperations

def main():
    """
    Simple example of using the Idea Space LLM model.
    """
    print("Initializing Idea Space LLM...")
    
    # Initialize the model
    model = IdeaSpaceLLM(
        pretrained_model_name="bert-base-uncased",
        latent_dim=512,
        max_seq_len=64,
        inference_steps=10,
        use_variational=True
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Define some example texts
    texts = [
        "The cat sat on the mat.",
        "El gato está en la alfombra.",  # Spanish version
        "A feline rests on a floor covering.",  # Paraphrase
        "The dog played in the yard.",  # Different concept
        "A canine romped in the garden."  # Paraphrase of different concept
    ]
    
    print("\nEncoding texts to latent space...")
    # Encode texts to latent vectors
    latent_vectors = []
    for text in texts:
        print(f"\nEncoding: \"{text}\"")
        encoded = model.tokenizer(text, return_tensors="pt").to(device)
        z = model.encode(encoded.input_ids, encoded.attention_mask)
        latent_vectors.append(z)
        print(f"Latent vector shape: {z.shape}")
    
    # Stack latent vectors
    z_stack = torch.cat(latent_vectors, dim=0)
    
    print("\nGenerating from latent vectors...")
    # Generate text from each latent vector
    for i, z in enumerate(latent_vectors):
        generated_text = model.generate(z=z, temperature=0.7, top_k=50)
        print(f"Original: \"{texts[i]}\"")
        print(f"Generated: \"{generated_text}\"")
        print()
    
    print("\nPerforming vector arithmetic in latent space...")
    # Perform latent space arithmetic (cat - dog = feline - canine)
    # Index 0: cat, Index 3: dog
    cat_minus_dog = latent_vectors[0] - latent_vectors[3]
    
    # Apply this difference to canine (Index 4)
    result_vector = latent_vectors[4] + cat_minus_dog
    
    # Generate from the result
    generated_text = model.generate(z=result_vector, temperature=0.7, top_k=50)
    print(f"cat - dog + canine = \"{generated_text}\"")
    
    print("\nInterpolating between concepts...")
    # Interpolate between cat and dog
    interpolation_steps = 5
    for i in range(interpolation_steps + 1):
        alpha = i / interpolation_steps
        z_interp = LatentOperations.interpolate(latent_vectors[0], latent_vectors[3], alpha)
        
        generated_text = model.generate(z=z_interp, temperature=0.7, top_k=50)
        print(f"Alpha = {alpha:.1f}: \"{generated_text}\"")
    
    # In a real application, we would save the visualizations
    # Here we'll just create the plots but not save them to avoid dependencies
    print("\nCreating visualizations (not saving to file)...")
    
    # Plot latent space embeddings
    labels = ["cat", "cat_es", "cat_para", "dog", "dog_para"]
    try:
        vis_fig = VisualizationUtils.plot_embeddings_2d(
            z_stack.cpu(), 
            labels=labels, 
            method="pca",
            title="Latent Space Embeddings"
        )
        print("Created embeddings visualization")
    except Exception as e:
        print(f"Couldn't create embeddings visualization: {e}")
    
    # Plot latent space heatmap
    try:
        heatmap_fig = VisualizationUtils.plot_latent_heatmap(
            model, 
            texts,
            title="Latent Space Distances"
        )
        print("Created latent space heatmap")
    except Exception as e:
        print(f"Couldn't create latent heatmap: {e}")
    
    print("\nExample complete!")

if __name__ == "__main__":
    main() 