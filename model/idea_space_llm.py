import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from .encoder import Encoder
from .decoder import DiffusionDecoder
from .noise_process import NoiseProcess

class IdeaSpaceLLM(nn.Module):
    """
    Idea Space LLM: A diffusion-based language model where similar concepts 
    occupy the same region in latent space.
    
    This model combines:
    1. An encoder that maps text to latent vectors
    2. A diffusion-based decoder that generates text from latent vectors
    3. A noise process for training and inference
    """
    
    def __init__(
        self,
        pretrained_model_name="bert-base-uncased",
        latent_dim=512,
        hidden_dim=512,
        embedding_dim=768,
        max_seq_len=64,
        diffusion_steps=1000,
        inference_steps=50,
        beta_schedule="linear",
        use_variational=True,
        predict_x0=False,
        tokenizer_name=None,
    ):
        """
        Initialize the Idea Space LLM.
        
        Args:
            pretrained_model_name (str): Name of the pretrained model for the encoder
            latent_dim (int): Dimension of the latent vector z
            hidden_dim (int): Dimension of hidden layers in the decoder
            embedding_dim (int): Dimension of token embeddings
            max_seq_len (int): Maximum sequence length
            diffusion_steps (int): Number of diffusion steps during training
            inference_steps (int): Number of diffusion steps during inference
            beta_schedule (str): Schedule for noise variance ('linear', 'cosine', 'quadratic')
            use_variational (bool): Whether to use a variational approach
            predict_x0 (bool): Whether the decoder should predict x0 directly
            tokenizer_name (str): Name of the tokenizer (defaults to pretrained_model_name)
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len
        self.diffusion_steps = diffusion_steps
        self.inference_steps = inference_steps
        self.predict_x0 = predict_x0
        
        # Set up tokenizer
        if tokenizer_name is None:
            tokenizer_name = pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Set up embedding table
        self.token_embedding = nn.Embedding(
            self.tokenizer.vocab_size, embedding_dim)
        
        # Set up encoder
        self.encoder = Encoder(
            pretrained_model_name=pretrained_model_name,
            latent_dim=latent_dim,
            pooling_strategy="mean",
            use_variational=use_variational,
        )
        
        # Set up decoder
        self.decoder = DiffusionDecoder(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            timestep_dim=128,
            latent_dim=latent_dim,
            num_layers=8,
            num_heads=8,
            dropout=0.1,
            max_seq_len=max_seq_len,
            predict_x0=predict_x0,
        )
        
        # Set up noise process
        self.noise_process = NoiseProcess(
            max_timesteps=diffusion_steps,
            beta_schedule=beta_schedule,
        )
        
    def get_embeddings(self, input_ids):
        """
        Convert token IDs to embeddings.
        
        Args:
            input_ids (torch.Tensor): Token IDs [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Token embeddings [batch_size, seq_len, embedding_dim]
        """
        return self.token_embedding(input_ids)
        
    def encode(self, input_ids, attention_mask=None):
        """
        Encode text to latent vector z.
        
        Args:
            input_ids (torch.Tensor): Token IDs [batch_size, seq_len]
            attention_mask (torch.Tensor): Attention mask [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Latent vector z [batch_size, latent_dim]
        """
        return self.encoder.encode(input_ids, attention_mask)
        
    def sample_timesteps(self, batch_size, device):
        """
        Sample random timesteps for training.
        
        Args:
            batch_size (int): Batch size
            device: Device to create tensor on
            
        Returns:
            torch.Tensor: Random timesteps [batch_size]
        """
        return torch.randint(0, self.diffusion_steps, (batch_size,), device=device, dtype=torch.long)
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass for training.
        
        Args:
            input_ids (torch.Tensor): Token IDs [batch_size, seq_len]
            attention_mask (torch.Tensor): Attention mask [batch_size, seq_len]
            
        Returns:
            dict: Dictionary containing loss terms and predictions
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Get clean embeddings (x_0)
        x_0 = self.get_embeddings(input_ids)
        
        # Encode to latent vector z
        z, kl_loss = self.encoder(input_ids, attention_mask)
        
        # Sample random timesteps
        t = self.sample_timesteps(batch_size, device)
        
        # Add noise to embeddings
        noise = torch.randn_like(x_0)
        x_t = self.noise_process.q_sample(x_0, t, noise)
        
        # Predict noise or x_0
        pred = self.decoder(x_t, t, z)
        
        if self.predict_x0:
            # Decoder predicts x_0 directly
            target = x_0
        else:
            # Decoder predicts noise
            target = noise
            
        # Calculate reconstruction loss
        rec_loss = F.mse_loss(pred, target)
        
        # Total loss
        loss = rec_loss
        if kl_loss is not None:
            loss = loss + 0.1 * kl_loss  # Beta weight for KL term
            
        return {
            "loss": loss,
            "rec_loss": rec_loss,
            "kl_loss": kl_loss,
            "pred": pred,
            "target": target,
            "z": z,
        }
    
    def generate(self, z=None, text=None, num_tokens=None, temperature=1.0, top_k=None, top_p=None):
        """
        Generate text from a latent vector or encode from input text.
        
        Args:
            z (torch.Tensor, optional): Latent vector [batch_size, latent_dim]
            text (str or list, optional): Text to encode to latent space
            num_tokens (int, optional): Number of tokens to generate (defaults to max_seq_len)
            temperature (float): Sampling temperature
            top_k (int, optional): Top-k sampling
            top_p (float, optional): Nucleus sampling threshold
            
        Returns:
            str or list: Generated text
        """
        device = next(self.parameters()).device
        
        # Set batch size and sequence length
        batch_size = z.shape[0] if z is not None else 1
        if num_tokens is None:
            num_tokens = self.max_seq_len
        
        # Get latent vector z either from input or by encoding text
        if z is None and text is not None:
            # Tokenize input text
            if isinstance(text, str):
                text = [text]
            
            encoded = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_seq_len,
                return_tensors="pt"
            ).to(device)
            
            # Encode to latent vector
            z = self.encode(encoded.input_ids, encoded.attention_mask)
            batch_size = z.shape[0]
            
        elif z is None:
            # Sample random latent vector
            z = torch.randn(batch_size, self.latent_dim, device=device)
            
        # Ensure z is on the correct device
        z = z.to(device)
        
        # Generate via the diffusion process
        shape = (batch_size, num_tokens, self.token_embedding.embedding_dim)
        
        # Use fewer steps for inference
        timesteps = list(range(0, self.diffusion_steps, self.diffusion_steps // self.inference_steps))
        
        # Generate embeddings
        generated_embeddings = self.noise_process.p_sample_loop(
            model=self.decoder,
            shape=shape,
            z=z,
            device=device,
            timesteps=timesteps,
            progress=True,
            model_output_type="x0" if self.predict_x0 else "noise"
        )
        
        # Convert embeddings to tokens
        generated_tokens = self._embeddings_to_tokens(
            generated_embeddings, 
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        
        # Convert tokens to text
        generated_texts = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )
        
        return generated_texts[0] if batch_size == 1 else generated_texts
    
    def _embeddings_to_tokens(self, embeddings, temperature=1.0, top_k=None, top_p=None):
        """
        Convert embeddings to tokens using nearest neighbor lookup in the embedding table.
        
        Args:
            embeddings (torch.Tensor): Generated embeddings [batch_size, seq_len, embedding_dim]
            temperature (float): Sampling temperature
            top_k (int, optional): Top-k sampling
            top_p (float, optional): Nucleus sampling threshold
            
        Returns:
            torch.Tensor: Token IDs [batch_size, seq_len]
        """
        batch_size, seq_len, embedding_dim = embeddings.shape
        
        # Get the full embedding matrix
        embedding_matrix = self.token_embedding.weight  # [vocab_size, embedding_dim]
        
        # Calculate similarity between generated embeddings and vocabulary embeddings
        # Reshape embeddings for batch similarity calculation
        embeddings_flat = embeddings.reshape(-1, embedding_dim)  # [batch_size * seq_len, embedding_dim]
        
        # Calculate cosine similarity
        similarity = torch.matmul(
            embeddings_flat / embeddings_flat.norm(dim=-1, keepdim=True),
            embedding_matrix.T / embedding_matrix.norm(dim=-1, keepdim=True)
        )  # [batch_size * seq_len, vocab_size]
        
        # Apply temperature
        logits = similarity / temperature
        
        # Apply top-k sampling if specified
        if top_k is not None:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Apply nucleus (top-p) sampling if specified
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            
            # Shift the indices to the right to keep the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Create a scatter map
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        # Sample from the distribution
        probs = torch.softmax(logits, dim=-1)
        tokens = torch.multinomial(probs, 1).squeeze(-1)
        
        # Reshape back to [batch_size, seq_len]
        tokens = tokens.reshape(batch_size, seq_len)
        
        return tokens 