import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class EmbeddingUtils:
    """
    Utility functions for working with embeddings and latent vectors.
    """
    
    @staticmethod
    def cosine_similarity(a, b):
        """
        Compute cosine similarity between two vectors or batches of vectors.
        
        Args:
            a: First vector or batch of vectors
            b: Second vector or batch of vectors
            
        Returns:
            torch.Tensor: Cosine similarity
        """
        a_norm = a / a.norm(dim=-1, keepdim=True)
        b_norm = b / b.norm(dim=-1, keepdim=True)
        return torch.matmul(a_norm, b_norm.transpose(-2, -1))
    
    @staticmethod
    def find_nearest_neighbors(query, embeddings, k=5):
        """
        Find the k nearest neighbors of a query vector in a set of embeddings.
        
        Args:
            query: Query vector [dim] or batch of queries [batch_size, dim]
            embeddings: Database of embeddings [num_embeddings, dim]
            k: Number of neighbors to return
            
        Returns:
            tuple: (indices, distances) of the nearest neighbors
        """
        if query.dim() == 1:
            query = query.unsqueeze(0)
            
        # Compute cosine similarity
        similarities = EmbeddingUtils.cosine_similarity(query, embeddings)
        
        # Get top-k values and indices
        distances, indices = torch.topk(similarities, k=k, dim=-1)
        
        return indices, distances
    
    @staticmethod
    def interpolate(a, b, alpha=0.5):
        """
        Interpolate between two latent vectors.
        
        Args:
            a: First latent vector
            b: Second latent vector
            alpha: Interpolation coefficient (0.0 = a, 1.0 = b)
            
        Returns:
            torch.Tensor: Interpolated vector
        """
        return (1 - alpha) * a + alpha * b
    
    @staticmethod
    def spherical_interpolate(a, b, alpha=0.5):
        """
        Spherical linear interpolation (slerp) between two latent vectors.
        
        Args:
            a: First latent vector
            b: Second latent vector
            alpha: Interpolation coefficient (0.0 = a, 1.0 = b)
            
        Returns:
            torch.Tensor: Interpolated vector
        """
        a_norm = a / a.norm(dim=-1, keepdim=True)
        b_norm = b / b.norm(dim=-1, keepdim=True)
        
        omega = torch.acos((a_norm * b_norm).sum(-1))
        so = torch.sin(omega)
        
        if so.abs() < 1e-6:
            # If vectors are nearly parallel, use linear interpolation
            return EmbeddingUtils.interpolate(a, b, alpha)
        
        return (torch.sin((1.0 - alpha) * omega) / so).unsqueeze(-1) * a + \
               (torch.sin(alpha * omega) / so).unsqueeze(-1) * b
    
    @staticmethod
    def reduce_dimensions(embeddings, method="pca", n_components=2):
        """
        Reduce dimensionality of embeddings for visualization.
        
        Args:
            embeddings: High-dimensional embeddings [num_embeddings, dim]
            method: Dimensionality reduction method ('pca' or 'tsne')
            n_components: Number of components in the reduced space
            
        Returns:
            np.ndarray: Reduced embeddings [num_embeddings, n_components]
        """
        # Convert to numpy if needed
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
            
        if method == "pca":
            reducer = PCA(n_components=n_components)
        elif method == "tsne":
            reducer = TSNE(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method}")
            
        return reducer.fit_transform(embeddings)
    
    @staticmethod
    def vector_arithmetic(word_embeddings, equation, embedding_model=None, tokenizer=None):
        """
        Perform vector arithmetic in the latent space (e.g., "king - man + woman").
        
        Args:
            word_embeddings: Dictionary of word to embedding or callable that returns embeddings
            equation: String equation (e.g., "king - man + woman")
            embedding_model: Optional model to encode words not in word_embeddings
            tokenizer: Tokenizer for the embedding model
            
        Returns:
            torch.Tensor: Resulting vector from the arithmetic operation
        """
        # Parse the equation
        tokens = equation.replace("+", " + ").replace("-", " - ").split()
        result = None
        current_op = "+"
        
        for token in tokens:
            if token in ["+", "-"]:
                current_op = token
                continue
                
            # Get embedding for the token
            if callable(word_embeddings):
                embedding = word_embeddings(token)
            elif token in word_embeddings:
                embedding = word_embeddings[token]
            elif embedding_model is not None and tokenizer is not None:
                # Encode the token using the model
                inputs = tokenizer(token, return_tensors="pt").to(embedding_model.device)
                with torch.no_grad():
                    embedding = embedding_model.encode(**inputs)
            else:
                raise ValueError(f"Word '{token}' not found in embeddings")
                
            # Apply the operation
            if result is None:
                result = embedding
            elif current_op == "+":
                result = result + embedding
            elif current_op == "-":
                result = result - embedding
                
        # Normalize the result
        if result is not None:
            return result / result.norm()
        return None 