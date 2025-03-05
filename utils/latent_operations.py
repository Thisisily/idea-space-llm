import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN

class LatentOperations:
    """
    Utility class for operations in the latent space.
    Provides functionality for manipulating latent vectors, exploring
    latent space structure, and performing operations like interpolation,
    arithmetic, and clustering.
    """
    
    @staticmethod
    def interpolate(a, b, alpha=0.5):
        """
        Linear interpolation between two latent vectors.
        
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
            return LatentOperations.interpolate(a, b, alpha)
        
        return (torch.sin((1.0 - alpha) * omega) / so).unsqueeze(-1) * a + \
               (torch.sin(alpha * omega) / so).unsqueeze(-1) * b
    
    @staticmethod
    def interpolate_batch(a, b, steps=10):
        """
        Create a batch of interpolated vectors between two latent vectors.
        
        Args:
            a: First latent vector
            b: Second latent vector
            steps: Number of interpolation steps
            
        Returns:
            torch.Tensor: Batch of interpolated vectors [steps, dim]
        """
        alphas = torch.linspace(0, 1, steps, device=a.device)
        interp_vectors = []
        
        for alpha in alphas:
            interp_vectors.append(LatentOperations.interpolate(a, b, alpha=alpha.item()))
            
        return torch.stack(interp_vectors)
    
    @staticmethod
    def arithmetic(vectors, coefficients):
        """
        Perform vector arithmetic in latent space.
        
        Args:
            vectors: List of latent vectors
            coefficients: List of coefficients
            
        Returns:
            torch.Tensor: Result of vector arithmetic
        """
        if len(vectors) != len(coefficients):
            raise ValueError("Number of vectors and coefficients must match")
            
        result = torch.zeros_like(vectors[0])
        for vector, coef in zip(vectors, coefficients):
            result += coef * vector
            
        return result
    
    @staticmethod
    def add_noise(z, scale=0.1):
        """
        Add Gaussian noise to a latent vector.
        
        Args:
            z: Latent vector
            scale: Standard deviation of noise
            
        Returns:
            torch.Tensor: Noisy latent vector
        """
        return z + scale * torch.randn_like(z)
    
    @staticmethod
    def cluster_latent_vectors(vectors, method="kmeans", n_clusters=5, **kwargs):
        """
        Cluster latent vectors.
        
        Args:
            vectors: Latent vectors [num_vectors, dim]
            method: Clustering method ('kmeans' or 'dbscan')
            n_clusters: Number of clusters (for KMeans)
            **kwargs: Additional arguments for clustering algorithms
            
        Returns:
            tuple: (labels, centroids) - Cluster assignments and centroids
        """
        # Convert to numpy if needed
        if isinstance(vectors, torch.Tensor):
            vectors_np = vectors.detach().cpu().numpy()
        else:
            vectors_np = vectors
            
        if method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, **kwargs)
            labels = clusterer.fit_predict(vectors_np)
            centroids = clusterer.cluster_centers_
        elif method == "dbscan":
            clusterer = DBSCAN(**kwargs)
            labels = clusterer.fit_predict(vectors_np)
            
            # Compute centroids for each cluster
            unique_labels = np.unique(labels)
            centroids = []
            for label in unique_labels:
                if label == -1:  # Noise points in DBSCAN
                    centroid = np.zeros(vectors_np.shape[1])
                else:
                    centroid = vectors_np[labels == label].mean(axis=0)
                centroids.append(centroid)
            centroids = np.array(centroids)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
            
        # Convert centroids back to torch if needed
        if isinstance(vectors, torch.Tensor):
            centroids = torch.tensor(centroids, device=vectors.device, dtype=vectors.dtype)
            
        return labels, centroids
    
    @staticmethod
    def find_principal_directions(vectors, n_components=10):
        """
        Find principal directions in the latent space using PCA.
        
        Args:
            vectors: Latent vectors [num_vectors, dim]
            n_components: Number of principal components to extract
            
        Returns:
            tuple: (principal_components, explained_variance_ratio)
        """
        # Convert to numpy if needed
        if isinstance(vectors, torch.Tensor):
            vectors_np = vectors.detach().cpu().numpy()
        else:
            vectors_np = vectors
            
        pca = PCA(n_components=n_components)
        pca.fit(vectors_np)
        
        principal_components = pca.components_
        explained_variance_ratio = pca.explained_variance_ratio_
        
        # Convert back to torch if needed
        if isinstance(vectors, torch.Tensor):
            principal_components = torch.tensor(
                principal_components, device=vectors.device, dtype=vectors.dtype
            )
            explained_variance_ratio = torch.tensor(
                explained_variance_ratio, device=vectors.device, dtype=vectors.dtype
            )
            
        return principal_components, explained_variance_ratio
    
    @staticmethod
    def attribute_manipulation(base_z, attribute_vectors, strengths):
        """
        Manipulate attributes in the latent space by adding attribute vectors.
        
        Args:
            base_z: Base latent vector
            attribute_vectors: Dictionary of attribute name to attribute vector
            strengths: Dictionary of attribute name to strength (coefficient)
            
        Returns:
            torch.Tensor: Manipulated latent vector
        """
        result = base_z.clone()
        
        for attr_name, strength in strengths.items():
            if attr_name in attribute_vectors:
                result = result + strength * attribute_vectors[attr_name]
                
        return result
    
    @staticmethod
    def compute_attribute_direction(positive_examples, negative_examples=None):
        """
        Compute an attribute direction in latent space by averaging the 
        difference between positive and negative examples.
        
        Args:
            positive_examples: Latent vectors with the attribute [num_pos, dim]
            negative_examples: Latent vectors without the attribute [num_neg, dim]
            
        Returns:
            torch.Tensor: Attribute direction vector
        """
        # Average positive examples
        pos_centroid = torch.mean(positive_examples, dim=0)
        
        if negative_examples is not None:
            # Average negative examples
            neg_centroid = torch.mean(negative_examples, dim=0)
            
            # Compute direction
            direction = pos_centroid - neg_centroid
        else:
            direction = pos_centroid
            
        # Normalize
        direction = direction / direction.norm()
        
        return direction 