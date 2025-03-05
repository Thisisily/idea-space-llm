import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

class NoiseProcess:
    """
    Implements the forward (noising) and reverse (denoising) processes for the diffusion model.
    
    The forward process gradually adds noise to embeddings over multiple timesteps.
    The reverse process learns to predict the noise or the previous step's embeddings,
    conditioned on the latent vector z.
    """
    
    def __init__(
        self,
        max_timesteps=1000,
        beta_schedule="linear",
        beta_start=1e-4,
        beta_end=0.02,
    ):
        """
        Initialize the noise process.
        
        Args:
            max_timesteps (int): Maximum number of noise steps
            beta_schedule (str): Schedule for noise variance ('linear', 'cosine', 'quadratic')
            beta_start (float): Initial noise scale
            beta_end (float): Final noise scale
        """
        self.max_timesteps = max_timesteps
        self.beta_schedule = beta_schedule
        
        # Set up variance schedule
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, max_timesteps)
        elif beta_schedule == "cosine":
            # Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
            steps = max_timesteps + 1
            x = torch.linspace(0, max_timesteps, steps)
            alphas_cumprod = torch.cos(((x / max_timesteps) + 0.008) / 1.008 * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        elif beta_schedule == "quadratic":
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, max_timesteps) ** 2
        else:
            raise ValueError(f"Unsupported beta schedule: {beta_schedule}")
            
        # Precompute values for diffusion process
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
    def q_sample(self, x_0, t, noise=None):
        """
        Sample from q(x_t | x_0) - the forward diffusion process.
        
        Args:
            x_0: Initial clean embeddings (batch_size, seq_len, embedding_dim)
            t: Timesteps (batch_size,)
            noise: Optional pre-generated noise
            
        Returns:
            Noisy embeddings x_t
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        # Extract the appropriate alphas for the current timesteps
        sqrt_alphas_cumprod_t = self._extract_at_t(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract_at_t(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        
        # Apply noise according to the diffusion SDE
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def _extract_at_t(self, values, t, broadcast_shape):
        """
        Extract values at specific timesteps and reshape for broadcasting.
        
        Args:
            values: Tensor of shape (max_timesteps,) or (max_timesteps, 1, ..., 1)
            t: Timesteps tensor of shape (batch_size,)
            broadcast_shape: Shape to broadcast to
            
        Returns:
            Tensor with values extracted at timesteps t, broadcasted to broadcast_shape
        """
        # Extract values at indices t
        out = values.to(t.device).gather(0, t)
        
        # Reshape for broadcasting
        return out.reshape(t.shape[0], *((1,) * (len(broadcast_shape) - 1)))
    
    def predict_start_from_noise(self, x_t, t, noise):
        """
        Predict x_0 from noise and x_t.
        
        Args:
            x_t: Noisy embeddings at timestep t
            t: Timesteps
            noise: Predicted noise
            
        Returns:
            Predicted x_0
        """
        sqrt_alphas_cumprod_t = self._extract_at_t(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract_at_t(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        
        # Solve for x_0 given the noise
        return (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t
    
    def q_posterior_mean(self, x_0, x_t, t):
        """
        Compute mean of q(x_{t-1} | x_t, x_0).
        
        Args:
            x_0: Predicted original embeddings
            x_t: Current noisy embeddings
            t: Current timesteps
            
        Returns:
            Posterior mean
        """
        posterior_mean_coef1 = self._extract_at_t(
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod),
            t, x_t.shape
        )
        posterior_mean_coef2 = self._extract_at_t(
            torch.sqrt(self.alphas) * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod),
            t, x_t.shape
        )
        
        posterior_mean = posterior_mean_coef1 * x_0 + posterior_mean_coef2 * x_t
        return posterior_mean
    
    def p_sample(self, model_output, x_t, t, z, model_output_type="noise"):
        """
        Sample from p(x_{t-1} | x_t) - the reverse diffusion process.
        
        Args:
            model_output: Output from the diffusion model
            x_t: Current noisy embeddings
            t: Current timesteps
            z: Latent vector conditioning the generation
            model_output_type: What the model predicts ('noise' or 'x_0')
            
        Returns:
            Sample of x_{t-1}
        """
        if model_output_type == "noise":
            # Model predicts the noise
            predicted_noise = model_output
            x_0_pred = self.predict_start_from_noise(x_t, t, predicted_noise)
        elif model_output_type == "x_0":
            # Model directly predicts x_0
            x_0_pred = model_output
        else:
            raise ValueError(f"Unsupported model_output_type: {model_output_type}")
            
        # Get the mean for q(x_{t-1} | x_t, x_0)
        posterior_mean = self.q_posterior_mean(x_0_pred, x_t, t)
        
        # Get the variance
        posterior_variance_t = self._extract_at_t(self.posterior_variance, t, x_t.shape)
        
        # No noise when t == 0
        noise = torch.randn_like(x_t)
        mask = (t > 0).float().reshape(t.shape[0], *((1,) * (len(x_t.shape) - 1)))
        
        return posterior_mean + torch.sqrt(posterior_variance_t) * noise * mask
    
    def p_sample_loop(self, model, shape, z, device, timesteps=None, progress=False, model_output_type="noise"):
        """
        Generate samples by iterating through the reverse diffusion process.
        
        Args:
            model: Diffusion decoder model
            shape: Shape of embeddings to generate (batch_size, seq_len, embedding_dim)
            z: Latent vector conditioning the generation
            device: Device to use
            timesteps: Specific timesteps to use (default: range(0, max_timesteps))
            progress: Whether to show progress
            model_output_type: What the model predicts ('noise' or 'x_0')
            
        Returns:
            Generated embeddings
        """
        # Start with pure noise
        x_t = torch.randn(shape, device=device)
        
        if timesteps is None:
            timesteps = list(range(0, self.max_timesteps))[::-1]
        else:
            timesteps = timesteps[::-1]
            
        if progress:
            from tqdm import tqdm
            timesteps = tqdm(timesteps)
            
        for t in timesteps:
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            with torch.no_grad():
                model_output = model(x_t, t_batch, z)
                x_t = self.p_sample(model_output, x_t, t_batch, z, model_output_type)
                
        return x_t 