import torch
import torch.nn as nn
from transformers import AutoModel

class Encoder(nn.Module):
    """
    Encoder component of the Idea Space LLM.
    
    Maps input sequences to latent vectors (z) that represent concepts in a 
    continuous, high-dimensional space where semantically similar inputs map to 
    nearby points.
    """
    
    def __init__(
        self,
        pretrained_model_name="bert-base-uncased",
        latent_dim=512,
        pooling_strategy="cls",
        use_variational=True,
    ):
        """
        Initialize the encoder.
        
        Args:
            pretrained_model_name (str): Name of the pretrained transformer model to use
            latent_dim (int): Dimension of the latent vector z
            pooling_strategy (str): Strategy for pooling transformer outputs 
                                    ('cls', 'mean', or 'max')
            use_variational (bool): Whether to use a variational approach (like VAE)
        """
        super().__init__()
        self.transformer = AutoModel.from_pretrained(pretrained_model_name)
        self.hidden_dim = self.transformer.config.hidden_size
        self.latent_dim = latent_dim
        self.pooling_strategy = pooling_strategy
        self.use_variational = use_variational
        
        # Projection from transformer hidden size to latent dimension
        if use_variational:
            # For variational approach, predict mean and log variance
            self.mu_projection = nn.Linear(self.hidden_dim, latent_dim)
            self.logvar_projection = nn.Linear(self.hidden_dim, latent_dim)
        else:
            # For deterministic approach, directly project to latent space
            self.projection = nn.Linear(self.hidden_dim, latent_dim)
            
    def pool_hidden_states(self, hidden_states, attention_mask=None):
        """
        Pool the transformer's hidden states according to the pooling strategy.
        
        Args:
            hidden_states: The output hidden states from the transformer
            attention_mask: Attention mask for ignoring padding tokens
            
        Returns:
            torch.Tensor: Pooled representation
        """
        if self.pooling_strategy == "cls":
            # Use the [CLS] token representation
            pooled = hidden_states[:, 0]
        
        elif self.pooling_strategy == "mean":
            # Mean pooling
            if attention_mask is not None:
                # Create proper mask (batch_size, seq_length, 1) and apply it
                mask = attention_mask.unsqueeze(-1).float()
                pooled = torch.sum(hidden_states * mask, dim=1) / torch.sum(mask, dim=1)
            else:
                pooled = torch.mean(hidden_states, dim=1)
        
        elif self.pooling_strategy == "max":
            # Max pooling
            if attention_mask is not None:
                # Set padding tokens to large negative value
                mask = attention_mask.unsqueeze(-1).float()
                hidden_states = hidden_states * mask - 1e10 * (1 - mask)
            
            pooled = torch.max(hidden_states, dim=1)[0]
        
        else:
            raise ValueError(f"Unsupported pooling strategy: {self.pooling_strategy}")
            
        return pooled
            
    def forward(self, input_ids, attention_mask=None):
        """
        Encode input tokens to latent vector z.
        
        Args:
            input_ids: Token IDs of the input sequence
            attention_mask: Attention mask for transformer
            
        Returns:
            z: Latent vector representation
            kl_loss: KL divergence loss (if variational, otherwise None)
        """
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Pool hidden states
        pooled = self.pool_hidden_states(outputs.last_hidden_state, attention_mask)
        
        if self.use_variational:
            # Variational approach (like VAE)
            mu = self.mu_projection(pooled)
            logvar = self.logvar_projection(pooled)
            
            # Reparameterization trick
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            
            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            
            return z, kl_loss
        else:
            # Deterministic approach
            z = self.projection(pooled)
            return z, None
            
    def encode(self, input_ids, attention_mask=None):
        """
        Encode input to latent representation (for inference).
        
        Args:
            input_ids: Token IDs of the input sequence
            attention_mask: Attention mask for transformer
            
        Returns:
            z: Latent vector representation
        """
        with torch.no_grad():
            if self.use_variational:
                # For variational approach, return mean directly (no sampling)
                outputs = self.transformer(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                pooled = self.pool_hidden_states(outputs.last_hidden_state, attention_mask)
                z = self.mu_projection(pooled)
            else:
                # For deterministic approach
                z, _ = self.forward(input_ids, attention_mask)
                
        return z 