import torch
import sys
import os
import argparse
import matplotlib.pyplot as plt

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import IdeaSpaceLLM
from utils.latent_operations import LatentOperations
from utils.visualization import VisualizationUtils

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Concept Manipulation in Latent Space")
    
    parser.add_argument("--model-path", type=str, help="Path to a saved model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run on (cuda or cpu)")
    parser.add_argument("--save-plots", action="store_true", help="Save visualization plots")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Directory to save outputs")
    
    return parser.parse_args()

def initialize_model(args):
    """Initialize the model from scratch or a checkpoint."""
    print(f"Initializing model on {args.device}...")
    
    model = IdeaSpaceLLM(
        pretrained_model_name="bert-base-uncased",
        latent_dim=512,
        max_seq_len=64,
        inference_steps=10,
    )
    
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    
    model = model.to(args.device)
    model.eval()
    
    return model

def demonstrate_vector_arithmetic(model, device, save_plots=False, output_dir=None):
    """Demonstrate vector arithmetic operations in latent space."""
    print("\n--- Vector Arithmetic ---")
    
    # Define pairs of related concepts
    concept_pairs = [
        # Semantic relations
        ["king", "queen", "man", "woman"],  # king - man + woman = queen
        ["paris", "france", "rome", "italy"],  # paris - france + italy = rome
        ["einstein", "scientist", "picasso", "artist"],  # einstein - scientist + artist = picasso
        
        # Attributes
        ["happy", "sad", "smile", "frown"],
        ["big", "small", "large", "tiny"],
        ["hot", "cold", "summer", "winter"],
    ]
    
    results = []
    
    for concepts in concept_pairs:
        a, a_category, b, b_category = concepts
        print(f"\nTesting: {a} - {a_category} + {b_category} = ?")
        
        # Encode concepts to latent vectors
        encoded_a = model.tokenizer(a, return_tensors="pt").to(device)
        encoded_a_category = model.tokenizer(a_category, return_tensors="pt").to(device)
        encoded_b_category = model.tokenizer(b_category, return_tensors="pt").to(device)
        
        with torch.no_grad():
            z_a = model.encode(encoded_a.input_ids, encoded_a.attention_mask)
            z_a_category = model.encode(encoded_a_category.input_ids, encoded_a_category.attention_mask)
            z_b_category = model.encode(encoded_b_category.input_ids, encoded_b_category.attention_mask)
            
            # Vector arithmetic: a - a_category + b_category
            z_result = z_a - z_a_category + z_b_category
            
            # Generate from the result vector
            result_text = model.generate(z=z_result, temperature=0.7, top_k=50)
            
            # Generate from the expected target for comparison
            encoded_b = model.tokenizer(b, return_tensors="pt").to(device)
            z_b = model.encode(encoded_b.input_ids, encoded_b.attention_mask)
            expected_text = model.generate(z=z_b, temperature=0.7, top_k=50)
            
        print(f"{a} - {a_category} + {b_category} = \"{result_text}\"")
        print(f"Expected ({b}): \"{expected_text}\"")
        
        results.append({
            "equation": f"{a} - {a_category} + {b_category}",
            "result_text": result_text,
            "expected": b,
            "expected_text": expected_text,
            "z_result": z_result.cpu(),
            "z_expected": z_b.cpu()
        })
    
    if save_plots and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot the latent space with concepts
        all_vectors = []
        labels = []
        
        for result in results:
            equation = result["equation"].split(" ")
            a, _, a_cat, _, b_cat = equation
            b = result["expected"]
            
            all_vectors.append(result["z_result"].squeeze(0))
            all_vectors.append(result["z_expected"].squeeze(0))
            
            labels.append(f"result_{a}_{b_cat}")
            labels.append(b)
        
        all_vectors = torch.stack(all_vectors)
        
        try:
            fig = VisualizationUtils.plot_embeddings_2d(
                all_vectors, 
                labels=labels, 
                method="pca",
                title="Vector Arithmetic in Latent Space"
            )
            
            fig.savefig(os.path.join(output_dir, "vector_arithmetic.png"))
            plt.close(fig)
            print(f"Saved vector arithmetic plot to {os.path.join(output_dir, 'vector_arithmetic.png')}")
        except Exception as e:
            print(f"Could not create vector arithmetic plot: {e}")
    
    return results

def demonstrate_interpolation(model, device, save_plots=False, output_dir=None):
    """Demonstrate interpolation between concepts."""
    print("\n--- Concept Interpolation ---")
    
    concept_pairs = [
        ["The cat sat on the mat.", "The dog played in the yard."],
        ["I love this movie.", "I hate this film."],
        ["The sun rises in the east.", "The moon glows in the night sky."],
    ]
    
    for pair in concept_pairs:
        text1, text2 = pair
        print(f"\nInterpolating between: \"{text1}\" and \"{text2}\"")
        
        # Encode texts
        encoded1 = model.tokenizer(text1, return_tensors="pt").to(device)
        encoded2 = model.tokenizer(text2, return_tensors="pt").to(device)
        
        with torch.no_grad():
            z1 = model.encode(encoded1.input_ids, encoded1.attention_mask)
            z2 = model.encode(encoded2.input_ids, encoded2.attention_mask)
            
            # Linear interpolation
            steps = 5
            for i in range(steps + 1):
                alpha = i / steps
                z_interp = LatentOperations.interpolate(z1, z2, alpha)
                
                # Generate from interpolated vector
                interp_text = model.generate(z=z_interp, temperature=0.7, top_k=50)
                print(f"α = {alpha:.1f}: \"{interp_text}\"")
        
        # Visualize interpolation
        if save_plots and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            try:
                fig = VisualizationUtils.plot_latent_interpolation(
                    model,
                    text1,
                    text2,
                    steps=10,
                    title=f"Interpolation: \"{text1}\" to \"{text2}\""
                )
                
                # Save the figure
                clean_name = f"interp_{'_'.join(text1.split()[:3])}_to_{'_'.join(text2.split()[:3])}"
                clean_name = clean_name.replace(".", "").replace(",", "")
                fig.savefig(os.path.join(output_dir, f"{clean_name}.png"))
                plt.close(fig)
                print(f"Saved interpolation plot to {os.path.join(output_dir, f'{clean_name}.png')}")
            except Exception as e:
                print(f"Could not create interpolation plot: {e}")

def demonstrate_attribute_manipulation(model, device, save_plots=False, output_dir=None):
    """Demonstrate attribute manipulation in latent space."""
    print("\n--- Attribute Manipulation ---")
    
    # Define some base texts and attributes
    base_texts = [
        "The house is small.",
        "The food tastes good.",
        "The movie was interesting.",
    ]
    
    attribute_pairs = [
        ("small", "large"),
        ("good", "bad"),
        ("interesting", "boring"),
    ]
    
    # Encode attribute directions
    attribute_vectors = {}
    
    for pos_attr, neg_attr in attribute_pairs:
        # Encode attribute words
        encoded_pos = model.tokenizer(pos_attr, return_tensors="pt").to(device)
        encoded_neg = model.tokenizer(neg_attr, return_tensors="pt").to(device)
        
        with torch.no_grad():
            z_pos = model.encode(encoded_pos.input_ids, encoded_pos.attention_mask)
            z_neg = model.encode(encoded_neg.input_ids, encoded_neg.attention_mask)
            
            # Calculate attribute direction
            direction = z_pos - z_neg
            attribute_vectors[f"{pos_attr}_vs_{neg_attr}"] = direction
    
    for text in base_texts:
        print(f"\nBase text: \"{text}\"")
        
        # Encode base text
        encoded_base = model.tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            z_base = model.encode(encoded_base.input_ids, encoded_base.attention_mask)
            
            # Apply different attribute directions with varying strengths
            for attr_name, attr_vector in attribute_vectors.items():
                pos_attr, neg_attr = attr_name.split("_vs_")
                
                print(f"\nApplying {attr_name} attribute:")
                
                # Apply attribute with different strengths
                for strength in [-1.5, -0.75, 0, 0.75, 1.5]:
                    modified_z = z_base + strength * attr_vector
                    
                    # Generate from modified vector
                    modified_text = model.generate(z=modified_z, temperature=0.7, top_k=50)
                    
                    if strength < 0:
                        print(f"More {neg_attr} ({strength:.2f}): \"{modified_text}\"")
                    elif strength > 0:
                        print(f"More {pos_attr} ({strength:.2f}): \"{modified_text}\"")
                    else:
                        print(f"Neutral (0.00): \"{modified_text}\"")

def main():
    """Main function to demonstrate concept manipulation."""
    args = parse_args()
    
    if args.save_plots and not args.output_dir:
        os.makedirs('./outputs', exist_ok=True)
        args.output_dir = './outputs'
    
    # Initialize model
    model = initialize_model(args)
    
    # Demonstrate vector arithmetic
    demonstrate_vector_arithmetic(model, args.device, args.save_plots, args.output_dir)
    
    # Demonstrate interpolation
    demonstrate_interpolation(model, args.device, args.save_plots, args.output_dir)
    
    # Demonstrate attribute manipulation
    demonstrate_attribute_manipulation(model, args.device, args.save_plots, args.output_dir)
    
    print("\nConcept manipulation demonstration complete!")

if __name__ == "__main__":
    main() 