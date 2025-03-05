import torch
import sys
import os
import argparse
from torch.utils.data import Dataset
from datasets import load_dataset

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import IdeaSpaceLLM
from training import Trainer
from utils.tokenizer import Tokenizer

class TextDataset(Dataset):
    """Simple dataset for text sequences."""
    
    def __init__(self, texts, tokenizer, max_length=64):
        """
        Initialize the dataset.
        
        Args:
            texts: List of texts
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
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

def load_data(dataset_name="wikicorpus", split="train", subsample=1000, starting_index=0):
    """
    Load a dataset for training.
    
    Args:
        dataset_name: Name of the dataset
        split: Split to load
        subsample: Number of examples to use (for testing)
        starting_index: Starting index for subsampling
        
    Returns:
        list: List of texts
    """
    print(f"Loading {dataset_name} dataset ({split} split)...")
    
    if dataset_name == "wikicorpus":
        try:
            dataset = load_dataset("wikicorpus", "raw_en", split=split)
            texts = dataset["text"][starting_index:starting_index + subsample]
        except Exception as e:
            print(f"Error loading wikicorpus: {e}")
            print("Using a fallback small dataset...")
            texts = [
                "The cat sat on the mat.",
                "Dogs are often called man's best friend.",
                "Natural language processing is a subfield of artificial intelligence.",
                "Deep learning has revolutionized machine learning in recent years.",
                "The quick brown fox jumps over the lazy dog."
            ] * 200  # Repeat to make it larger
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    print(f"Loaded {len(texts)} text examples")
    return texts

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train the Idea Space LLM model")
    
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=512, help="Latent dimension")
    parser.add_argument("--max-seq-len", type=int, default=64, help="Maximum sequence length")
    parser.add_argument("--diffusion-steps", type=int, default=100, help="Number of diffusion steps")
    parser.add_argument("--inference-steps", type=int, default=10, help="Number of inference steps")
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--output-dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--subsample", type=int, default=1000, help="Number of examples to use")
    
    return parser.parse_args()

def main():
    """Train the Idea Space LLM model."""
    args = parse_args()
    
    print("Initializing training...")
    print(f"Arguments: {args}")
    
    # Load data
    texts = load_data(subsample=args.subsample)
    
    # Split into train and evaluation sets
    train_ratio = 0.9
    train_size = int(len(texts) * train_ratio)
    
    train_texts = texts[:train_size]
    eval_texts = texts[train_size:]
    
    # Create tokenizer
    tokenizer = Tokenizer(
        pretrained_model_name="bert-base-uncased",
        max_length=args.max_seq_len
    )
    
    # Create datasets
    train_dataset = TextDataset(train_texts, tokenizer, max_length=args.max_seq_len)
    eval_dataset = TextDataset(eval_texts, tokenizer, max_length=args.max_seq_len)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    
    # Initialize model
    model = IdeaSpaceLLM(
        pretrained_model_name="bert-base-uncased",
        latent_dim=args.latent_dim,
        max_seq_len=args.max_seq_len,
        diffusion_steps=args.diffusion_steps,
        inference_steps=args.inference_steps,
        use_variational=True
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        project_name="idea-space-llm"
    )
    
    # Train the model
    print("\nStarting training...")
    trainer.train()
    
    print("\nTraining complete!")
    
    # Generate some examples from the trained model
    print("\nGenerating examples from trained model...")
    device = next(model.parameters()).device
    
    test_texts = [
        "The cat sat on the mat.",
        "Machine learning is a field of study.",
        "The sun rises in the east."
    ]
    
    for text in test_texts:
        print(f"\nOriginal: \"{text}\"")
        
        # Encode to latent space
        encoded = tokenizer.encode(text, return_tensors="pt").to(device)
        z = model.encode(encoded.input_ids, encoded.attention_mask)
        
        # Generate from latent
        generated = model.generate(z=z, temperature=0.7, top_k=50)
        print(f"Generated: \"{generated}\"")
    
    print("\nExample complete!")

if __name__ == "__main__":
    main() 