import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for the timestep in the diffusion process.
    Similar to the position embeddings used in Transformer models.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TimestepEmbedding(nn.Module):
    """
    Timestep embedding layer that projects the timestep to a higher-dimensional
    representation for conditioning the diffusion model.
    """
    def __init__(self, dim, output_dim):
        super().__init__()
        self.sinusoidal_embedding = SinusoidalPositionEmbeddings(dim)
        self.proj = nn.Sequential(
            nn.Linear(dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, timestep):
        x = self.sinusoidal_embedding(timestep)
        return self.proj(x)


class TransformerBlock(nn.Module):
    """
    Transformer block used in the diffusion decoder.
    Combines self-attention and feed-forward layers with conditioning from z and timestep.
    """
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None
    ):
        super().__init__()
        
        # Self-attention
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        
        # Cross-attention for conditioning on z
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True) if context_dim is not None else None
        self.norm2 = nn.LayerNorm(dim) if context_dim is not None else None
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None):
        # Self-attention block
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = residual + x
        
        # Cross-attention block (conditioning on z)
        if self.cross_attn is not None and context is not None:
            residual = x
            if self.norm2 is not None:
                x = self.norm2(x)
            context = context.unsqueeze(1) if context.dim() == 2 else context
            x, _ = self.cross_attn(x, context, context)
            x = residual + x
            
        # Feed-forward block
        return x + self.ff(x)


class DiffusionDecoder(nn.Module):
    """
    Diffusion-based decoder that generates a sequence by iteratively denoising,
    conditioned on a latent vector z.
    """
    def __init__(
        self,
        input_dim,
        hidden_dim=512,
        output_dim=None,
        timestep_dim=128,
        latent_dim=512,
        num_layers=8,
        num_heads=8,
        dropout=0.1,
        max_seq_len=64,
        predict_x0=False,
    ):
        """
        Initialize the diffusion decoder.
        
        Args:
            input_dim: Dimension of input embeddings (e.g., token embeddings)
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output (if different from input_dim)
            timestep_dim: Dimension of timestep embeddings
            latent_dim: Dimension of latent vector z
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            predict_x0: Whether to predict x0 directly (if False, predict noise)
        """
        super().__init__()
        
        if output_dim is None:
            output_dim = input_dim
            
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.predict_x0 = predict_x0
        
        # Timestep embedding
        self.time_embed = TimestepEmbedding(timestep_dim, hidden_dim)
        
        # Project latent vector z to conditioning context
        self.z_embed = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Input projection
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding for sequence positions
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=hidden_dim,
                n_heads=num_heads,
                d_head=hidden_dim // num_heads,
                dropout=dropout,
                context_dim=hidden_dim
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        
        # Initialize position embeddings
        nn.init.normal_(self.pos_embed, std=0.02)
        
    def forward(self, x, timestep, z):
        """
        Forward pass of the diffusion decoder.
        
        Args:
            x: Noisy embeddings [batch_size, seq_len, input_dim]
            timestep: Diffusion timesteps [batch_size]
            z: Latent vectors [batch_size, latent_dim]
            
        Returns:
            Predicted noise or x0, depending on configuration
        """
        batch_size, seq_len, _ = x.shape
        
        # Timestep embedding
        t_emb = self.time_embed(timestep)  # [batch_size, hidden_dim]
        
        # Latent conditioning
        z_emb = self.z_embed(z)  # [batch_size, hidden_dim]
        
        # Input projection
        h = self.in_proj(x)  # [batch_size, seq_len, hidden_dim]
        
        # Add positional embeddings
        pos_emb = self.pos_embed[:, :seq_len, :]
        h = h + pos_emb
        
        # Add timestep embeddings to each position
        h = h + t_emb.unsqueeze(1)
        
        # Prepare z context for cross-attention
        z_context = z_emb.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Process through transformer blocks
        for block in self.blocks:
            h = block(h, context=z_context)
            
        # Output projection
        h = self.out_norm(h)
        output = self.out_proj(h)
        
        return output 