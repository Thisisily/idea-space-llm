import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

class VisualizationUtils:
    """
    Utilities for visualizing embeddings, latent space, and model outputs.
    """
    
    @staticmethod
    def plot_embeddings_2d(embeddings, labels=None, method="pca", figsize=(10, 8),
                           colormap="tab10", marker="o", alpha=0.8, title=None):
        """
        Plot embeddings in 2D space after dimensionality reduction.
        
        Args:
            embeddings: High-dimensional embeddings [num_embeddings, dim]
            labels: Optional labels or text for the points
            method: Dimensionality reduction method ('pca' or 'tsne')
            figsize: Figure size
            colormap: Colormap for different classes
            marker: Marker style
            alpha: Alpha (transparency) value
            title: Plot title
            
        Returns:
            matplotlib.figure.Figure: The figure
        """
        # Convert tensors to numpy
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
            
        # Reduce dimensions
        if method == "pca":
            reducer = PCA(n_components=2)
            reduced_embeddings = reducer.fit_transform(embeddings)
            method_name = "PCA"
        elif method == "tsne":
            # Compute an appropriate perplexity value - must be less than the number of samples
            n_samples = embeddings.shape[0]
            perplexity = min(30, max(5, n_samples // 3))  # Between 5 and 30, or n_samples/3
            print(f"Using t-SNE with perplexity={perplexity} for {n_samples} samples")
            
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            reduced_embeddings = reducer.fit_transform(embeddings)
            method_name = "t-SNE"
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method}")
            
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Color points by label if provided
        if labels is not None:
            unique_labels = np.unique(labels)
            cmap = plt.get_cmap(colormap, len(unique_labels))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(
                    reduced_embeddings[mask, 0],
                    reduced_embeddings[mask, 1],
                    c=[cmap(i)],
                    label=label,
                    marker=marker,
                    alpha=alpha
                )
            ax.legend()
        else:
            ax.scatter(
                reduced_embeddings[:, 0],
                reduced_embeddings[:, 1],
                marker=marker,
                alpha=alpha
            )
            
        # Add text annotations if labels are strings
        if labels is not None and isinstance(labels[0], str):
            for i, txt in enumerate(labels):
                ax.annotate(txt, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))
                
        # Set title and labels
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Embeddings visualized with {method_name}")
            
        ax.set_xlabel(f"Dimension 1")
        ax.set_ylabel(f"Dimension 2")
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_latent_interpolation(model, text1, text2, steps=10, temperature=0.7,
                                  figsize=(14, 6), title=None):
        """
        Visualize interpolation between two points in latent space.
        
        Args:
            model: The Idea Space LLM model
            text1: First text
            text2: Second text
            steps: Number of interpolation steps
            temperature: Sampling temperature
            figsize: Figure size
            title: Plot title
            
        Returns:
            matplotlib.figure.Figure: The figure
        """
        device = next(model.parameters()).device
        
        # Encode texts
        with torch.no_grad():
            encoded1 = model.tokenizer(text1, return_tensors="pt").to(device)
            encoded2 = model.tokenizer(text2, return_tensors="pt").to(device)
            
            z1 = model.encode(encoded1.input_ids, encoded1.attention_mask)
            z2 = model.encode(encoded2.input_ids, encoded2.attention_mask)
            
        # Generate interpolated texts
        interpolated_texts = []
        alphas = np.linspace(0, 1, steps)
        
        for alpha in alphas:
            # Linear interpolation
            z_interp = (1 - alpha) * z1 + alpha * z2
            
            # Generate from interpolated z
            with torch.no_grad():
                generated_text = model.generate(
                    z=z_interp,
                    temperature=temperature
                )
                
            interpolated_texts.append(generated_text)
            
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, steps)
        
        # Plot texts
        for i, (text, alpha) in enumerate(zip(interpolated_texts, alphas)):
            ax.text(0.5, steps - i - 0.5, text, fontsize=10, 
                    horizontalalignment='center', verticalalignment='center')
            
            # Draw lines at ends
            if i == 0 or i == steps - 1:
                ax.axhline(y=steps - i, color='black', linestyle='-', alpha=0.3)
                
        # Draw interpolation line
        ax.plot([0, 1], [steps, 0], 'r--', alpha=0.5)
        
        # Add text labels
        ax.text(0, steps + 0.5, text1, fontsize=12, fontweight='bold', 
                horizontalalignment='center', verticalalignment='center')
        ax.text(1, -0.5, text2, fontsize=12, fontweight='bold', 
                horizontalalignment='center', verticalalignment='center')
        
        # Set title
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Latent Space Interpolation")
            
        ax.set_xlabel("Interpolation (α)")
        ax.set_yticks([])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(['0.0', '0.25', '0.5', '0.75', '1.0'])
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_loss_history(loss_history, figsize=(10, 6), smoothing=0.9):
        """
        Plot training loss history.
        
        Args:
            loss_history: Dictionary of loss metrics per step
            figsize: Figure size
            smoothing: Exponential moving average smoothing factor
            
        Returns:
            matplotlib.figure.Figure: The figure
        """
        fig, axes = plt.subplots(len(loss_history), 1, figsize=figsize, sharex=True)
        
        # Handle case with only one loss type
        if len(loss_history) == 1:
            axes = [axes]
            
        for i, (loss_name, values) in enumerate(loss_history.items()):
            steps = np.arange(1, len(values) + 1)
            
            # Apply exponential moving average smoothing
            if smoothing > 0:
                smoothed_values = []
                last = values[0]
                for value in values:
                    smoothed_val = last * smoothing + (1 - smoothing) * value
                    smoothed_values.append(smoothed_val)
                    last = smoothed_val
            else:
                smoothed_values = values
                
            # Plot raw and smoothed values
            axes[i].plot(steps, values, 'b-', alpha=0.3, label='Raw')
            if smoothing > 0:
                axes[i].plot(steps, smoothed_values, 'r-', label='Smoothed')
                
            axes[i].set_ylabel(loss_name)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
        axes[-1].set_xlabel('Steps')
        fig.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_latent_heatmap(model, texts, figsize=(12, 10), cmap="viridis",
                          normalize=True, title=None):
        """
        Plot a heatmap of pairwise distances in latent space.
        
        Args:
            model: The Idea Space LLM model
            texts: List of texts to encode
            figsize: Figure size
            cmap: Colormap
            normalize: Whether to normalize distances to [0, 1]
            title: Plot title
            
        Returns:
            matplotlib.figure.Figure: The figure
        """
        device = next(model.parameters()).device
        
        # Encode texts
        z_vectors = []
        for text in texts:
            with torch.no_grad():
                encoded = model.tokenizer(text, return_tensors="pt").to(device)
                z = model.encode(encoded.input_ids, encoded.attention_mask)
                z_vectors.append(z.cpu())
                
        # Stack vectors
        z_matrix = torch.cat(z_vectors, dim=0)
        
        # Compute pairwise cosine similarities
        z_norm = z_matrix / z_matrix.norm(dim=1, keepdim=True)
        sim_matrix = torch.mm(z_norm, z_norm.t())
        
        # Convert to distance (1 - similarity)
        dist_matrix = 1 - sim_matrix
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(dist_matrix.numpy(), ax=ax, cmap=cmap, 
                   annot=True, fmt=".2f", xticklabels=texts, 
                   yticklabels=texts)
        
        # Set title
        if title:
            ax.set_title(title)
        else:
            ax.set_title("Pairwise Distances in Latent Space")
            
        plt.tight_layout()
        return fig 