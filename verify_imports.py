#!/usr/bin/env python
"""
Verify that imports are working correctly.
This script attempts to import all the key modules to ensure they can be found.
"""

print("Verifying imports...")

try:
    import torch
    print("✓ torch")
    
    import torch.nn as nn
    print("✓ torch.nn")
    
    import torch.nn.functional as F
    print("✓ torch.nn.functional")
    
    from transformers import AutoTokenizer
    print("✓ transformers")
    
    from model import IdeaSpaceLLM, Encoder, DiffusionDecoder, NoiseProcess
    print("✓ model components")
    
    from utils import EmbeddingUtils, LatentOperations, Tokenizer, VisualizationUtils
    print("✓ utils components")
    
    from training import Trainer, ReconstructionLoss, LatentRegularizationLoss
    print("✓ training components")
    
    print("\nAll imports verified successfully!")
    
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    print("Please check your virtual environment and package installations.") 