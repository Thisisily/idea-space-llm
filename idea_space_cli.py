#!/usr/bin/env python
import argparse
import os
import sys
import torch
import readline  # For better command-line input handling
import json
import numpy as np
from pathlib import Path

from model import IdeaSpaceLLM
from utils.latent_operations import LatentOperations

class LatentVectorDatabase:
    """Simple database for storing and retrieving named latent vectors."""
    
    def __init__(self, save_path="latent_vectors.json"):
        self.save_path = save_path
        self.vectors = {}
        self.load()
        
    def add(self, name, vector):
        """Add a vector to the database."""
        # Convert tensor to list for JSON serialization
        if isinstance(vector, torch.Tensor):
            vector = vector.cpu().numpy().tolist()
        
        self.vectors[name] = vector
        self.save()
        
    def get(self, name, device=None):
        """Get a vector from the database."""
        if name not in self.vectors:
            return None
        
        vector = self.vectors[name]
        
        # Convert list back to tensor
        if isinstance(vector, list):
            vector = torch.tensor(vector)
            
        if device is not None:
            vector = vector.to(device)
        
        return vector
        
    def remove(self, name):
        """Remove a vector from the database."""
        if name in self.vectors:
            del self.vectors[name]
            self.save()
            return True
        return False
    
    def list(self):
        """List all vectors in the database."""
        return list(self.vectors.keys())
    
    def save(self):
        """Save the database to disk."""
        with open(self.save_path, 'w') as f:
            json.dump(self.vectors, f)
    
    def load(self):
        """Load the database from disk."""
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'r') as f:
                    self.vectors = json.load(f)
            except json.JSONDecodeError:
                print(f"Error: Could not load latent vector database from {self.save_path}")
                self.vectors = {}

class IdeaSpaceCLI:
    """Command-line interface for Idea Space LLM."""
    
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_db = LatentVectorDatabase()
        self.current_z = None
        self.history = []
        self.help_text = """
Idea Space LLM CLI - Available commands:

BASIC COMMANDS:
  help                            Show this help
  exit/quit                       Exit the CLI
  device [cpu/cuda]               Set or show the current device

MODEL COMMANDS:
  load <model_path>               Load a model checkpoint
  load_default                    Load the default model
  info                            Show model information

TEXT OPERATIONS:
  encode <text>                   Encode text to latent vector
  generate                        Generate text from current latent vector
  generate_with <text>            Encode and generate in one step
  
LATENT VECTOR OPERATIONS:
  save <name>                     Save the current latent vector
  load_vector <name>              Load a latent vector
  list_vectors                    List saved latent vectors
  delete_vector <name>            Delete a saved latent vector
  add <name> <scale>              Add a saved vector to current vector
  subtract <name> <scale>         Subtract a saved vector from current vector
  interpolate <name> <alpha>      Interpolate between current vector and saved vector
  
ARITHMETIC:
  calc <equation>                 Calculate vector equation (e.g., "king - man + woman")
  
EXPLORATION:
  nearest <text> <top_k>          Find texts with most similar latent vectors
  
HISTORY:
  history                         Show command history
  clear_history                   Clear command history
"""
        
    def initialize_model(self, model_path=None):
        """Initialize the model from scratch or a checkpoint."""
        print(f"Initializing model on {self.device}...")
        
        self.model = IdeaSpaceLLM(
            pretrained_model_name="bert-base-uncased",
            latent_dim=512,
            max_seq_len=64,
            inference_steps=10,
        )
        
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print("Model initialized successfully.")
        
    def encode_text(self, text):
        """Encode text to a latent vector."""
        if not self.model:
            print("Error: Model not loaded. Please load a model first.")
            return None
        
        try:
            # Tokenize
            encoded = self.model.tokenizer(text, return_tensors="pt").to(self.device)
            
            # Encode
            with torch.no_grad():
                z = self.model.encode(encoded.input_ids, encoded.attention_mask)
                
            print(f"Text encoded successfully. Latent vector shape: {z.shape}")
            
            # Set as current vector
            self.current_z = z
            
            return z
        except Exception as e:
            print(f"Error encoding text: {e}")
            return None
    
    def generate_text(self, z=None, temperature=0.7, top_k=50):
        """Generate text from a latent vector."""
        if not self.model:
            print("Error: Model not loaded. Please load a model first.")
            return None
        
        if z is None:
            if self.current_z is None:
                print("Error: No latent vector available. Please encode text first.")
                return None
            z = self.current_z
        
        try:
            with torch.no_grad():
                generated_text = self.model.generate(
                    z=z,
                    temperature=temperature,
                    top_k=top_k
                )
                
            print(f"Generated: \"{generated_text}\"")
            return generated_text
        except Exception as e:
            print(f"Error generating text: {e}")
            return None
    
    def calculate_vector_equation(self, equation):
        """Calculate a vector equation with saved vectors."""
        if not self.latent_db.list():
            print("Error: No saved vectors available.")
            return None
        
        # Parse the equation, e.g., "king - man + woman"
        tokens = equation.replace("+", " + ").replace("-", " - ").split()
        
        result = None
        current_op = "+"
        
        for token in tokens:
            if token in ["+", "-"]:
                current_op = token
                continue
            
            # Get vector
            vector = self.latent_db.get(token, self.device)
            
            if vector is None:
                print(f"Error: Vector '{token}' not found.")
                return None
            
            # Apply operation
            if result is None:
                result = vector.clone()
            elif current_op == "+":
                result = result + vector
            elif current_op == "-":
                result = result - vector
        
        if result is not None:
            print(f"Calculated vector equation: {equation}")
            self.current_z = result
            
        return result
    
    def run_command(self, command):
        """Parse and execute a command."""
        self.history.append(command)
        
        parts = command.strip().split()
        if not parts:
            return
        
        cmd = parts[0].lower()
        args = parts[1:]
        
        # Basic commands
        if cmd == "help":
            print(self.help_text)
            
        elif cmd in ["exit", "quit"]:
            print("Exiting Idea Space CLI.")
            sys.exit(0)
            
        elif cmd == "device":
            if args:
                device_name = args[0].lower()
                if device_name == "cuda" and torch.cuda.is_available():
                    self.device = torch.device("cuda")
                elif device_name == "cpu":
                    self.device = torch.device("cpu")
                else:
                    print(f"Error: Invalid device '{device_name}' or CUDA not available.")
                    return
                
                # Move model to new device if loaded
                if self.model:
                    self.model = self.model.to(self.device)
                    
                # Move current vector to new device if available
                if self.current_z is not None:
                    self.current_z = self.current_z.to(self.device)
                    
                print(f"Device set to: {self.device}")
            else:
                print(f"Current device: {self.device}")
        
        # Model commands
        elif cmd == "load":
            if not args:
                print("Error: Please specify a model path.")
                return
                
            model_path = args[0]
            self.initialize_model(model_path)
            
        elif cmd == "load_default":
            self.initialize_model()
            
        elif cmd == "info":
            if not self.model:
                print("Error: Model not loaded. Please load a model first.")
                return
                
            print("\nModel Information:")
            print(f"  Device: {self.device}")
            print(f"  Latent dimension: {self.model.latent_dim}")
            print(f"  Max sequence length: {self.model.max_seq_len}")
            print(f"  Diffusion steps: {self.model.diffusion_steps}")
            print(f"  Inference steps: {self.model.inference_steps}")
            print(f"  Variational: {self.model.encoder.use_variational}")
            print(f"  Tokenizer: {self.model.tokenizer.name_or_path}")
            
            if self.current_z is not None:
                print(f"  Current latent vector shape: {self.current_z.shape}")
            else:
                print("  No current latent vector")
        
        # Text operations
        elif cmd == "encode":
            if not args:
                print("Error: Please provide text to encode.")
                return
                
            text = " ".join(args)
            self.encode_text(text)
            
        elif cmd == "generate":
            self.generate_text()
            
        elif cmd == "generate_with":
            if not args:
                print("Error: Please provide text to encode and generate.")
                return
                
            text = " ".join(args)
            z = self.encode_text(text)
            
            if z is not None:
                self.generate_text(z)
        
        # Latent vector operations
        elif cmd == "save":
            if not args:
                print("Error: Please provide a name for the vector.")
                return
                
            if self.current_z is None:
                print("Error: No current latent vector to save.")
                return
                
            name = args[0]
            self.latent_db.add(name, self.current_z)
            print(f"Saved current latent vector as '{name}'.")
            
        elif cmd == "load_vector":
            if not args:
                print("Error: Please provide a vector name.")
                return
                
            name = args[0]
            vector = self.latent_db.get(name, self.device)
            
            if vector is None:
                print(f"Error: Vector '{name}' not found.")
                return
                
            self.current_z = vector
            print(f"Loaded vector '{name}' as current vector.")
            
        elif cmd == "list_vectors":
            vectors = self.latent_db.list()
            
            if not vectors:
                print("No saved vectors found.")
                return
                
            print("Saved vectors:")
            for v in vectors:
                print(f"  {v}")
                
        elif cmd == "delete_vector":
            if not args:
                print("Error: Please provide a vector name.")
                return
                
            name = args[0]
            success = self.latent_db.remove(name)
            
            if success:
                print(f"Deleted vector '{name}'.")
            else:
                print(f"Error: Vector '{name}' not found.")
                
        elif cmd == "add":
            if len(args) < 1:
                print("Error: Please provide a vector name and optional scale.")
                return
                
            name = args[0]
            scale = float(args[1]) if len(args) > 1 else 1.0
            
            if self.current_z is None:
                print("Error: No current latent vector.")
                return
                
            vector = self.latent_db.get(name, self.device)
            
            if vector is None:
                print(f"Error: Vector '{name}' not found.")
                return
                
            self.current_z = self.current_z + scale * vector
            print(f"Added vector '{name}' with scale {scale} to current vector.")
            
        elif cmd == "subtract":
            if len(args) < 1:
                print("Error: Please provide a vector name and optional scale.")
                return
                
            name = args[0]
            scale = float(args[1]) if len(args) > 1 else 1.0
            
            if self.current_z is None:
                print("Error: No current latent vector.")
                return
                
            vector = self.latent_db.get(name, self.device)
            
            if vector is None:
                print(f"Error: Vector '{name}' not found.")
                return
                
            self.current_z = self.current_z - scale * vector
            print(f"Subtracted vector '{name}' with scale {scale} from current vector.")
            
        elif cmd == "interpolate":
            if len(args) < 2:
                print("Error: Please provide a vector name and interpolation factor.")
                return
                
            name = args[0]
            try:
                alpha = float(args[1])
            except ValueError:
                print("Error: Interpolation factor must be a number between 0 and 1.")
                return
                
            if self.current_z is None:
                print("Error: No current latent vector.")
                return
                
            vector = self.latent_db.get(name, self.device)
            
            if vector is None:
                print(f"Error: Vector '{name}' not found.")
                return
                
            self.current_z = LatentOperations.interpolate(self.current_z, vector, alpha)
            print(f"Interpolated current vector with '{name}' using alpha={alpha}.")
        
        # Arithmetic
        elif cmd == "calc":
            if not args:
                print("Error: Please provide a vector equation.")
                return
                
            equation = " ".join(args)
            self.calculate_vector_equation(equation)
            
        # History
        elif cmd == "history":
            print("Command history:")
            for i, cmd in enumerate(self.history):
                print(f"  {i+1}: {cmd}")
                
        elif cmd == "clear_history":
            self.history = []
            print("Command history cleared.")
            
        # Unknown command
        else:
            print(f"Unknown command: {cmd}")
            print("Type 'help' for a list of available commands.")
    
    def run_interactive(self):
        """Run the CLI in interactive mode."""
        print("Idea Space LLM CLI")
        print("Type 'help' for a list of commands, 'exit' to quit.")
        
        while True:
            try:
                command = input("\nidea-space> ")
                if command.strip():
                    self.run_command(command)
            except KeyboardInterrupt:
                print("\nExiting Idea Space CLI.")
                break
            except Exception as e:
                print(f"Error: {e}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Idea Space LLM Command Line Interface")
    
    parser.add_argument("--model", type=str, help="Path to a model checkpoint")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], 
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on (cuda or cpu)")
    parser.add_argument("--command", type=str, help="Run a single command and exit")
    
    return parser.parse_args()

def main():
    """Main function for the CLI."""
    args = parse_args()
    
    cli = IdeaSpaceCLI()
    
    # Set device
    if args.device == "cuda" and torch.cuda.is_available():
        cli.device = torch.device("cuda")
    else:
        cli.device = torch.device("cpu")
    
    # Load model if specified
    if args.model:
        cli.initialize_model(args.model)
    
    # Run single command if specified
    if args.command:
        cli.run_command(args.command)
    else:
        cli.run_interactive()

if __name__ == "__main__":
    main() 