import torch
import torch.nn as nn
import torch.nn.functional as F

class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss for the diffusion model.
    Computes the mean squared error between predictions and targets.
    """
    
    def __init__(self, reduction="mean"):
        """
        Initialize the reconstruction loss.
        
        Args:
            reduction (str): Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.reduction = reduction
        
    def forward(self, pred, target, mask=None):
        """
        Compute the reconstruction loss.
        
        Args:
            pred: Model predictions
            target: Ground truth targets
            mask: Optional mask for masked loss computation
            
        Returns:
            torch.Tensor: Loss value
        """
        if mask is None:
            return F.mse_loss(pred, target, reduction=self.reduction)
        else:
            # Apply mask
            loss = ((pred - target) ** 2) * mask.unsqueeze(-1)
            
            if self.reduction == "mean":
                return loss.sum() / (mask.sum() * pred.size(-1) + 1e-8)
            elif self.reduction == "sum":
                return loss.sum()
            else:
                return loss


class LatentRegularizationLoss(nn.Module):
    """
    KL divergence loss for regularizing the latent space.
    Used in variational approaches to encourage the latent distribution
    to be close to a prior (typically standard normal).
    """
    
    def __init__(self, beta=1.0, reduction="mean"):
        """
        Initialize the latent regularization loss.
        
        Args:
            beta (float): Weight for the KL term
            reduction (str): Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.beta = beta
        self.reduction = reduction
        
    def forward(self, mu, logvar):
        """
        Compute the KL divergence between the encoded distribution and the prior.
        
        Args:
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution
            
        Returns:
            torch.Tensor: KL divergence loss
        """
        # KL divergence between N(mu, sigma) and N(0, 1)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        
        if self.reduction == "mean":
            kl_div = kl_div.mean()
        elif self.reduction == "sum":
            kl_div = kl_div.sum()
            
        return self.beta * kl_div
        

class CosineEmbeddingLoss(nn.Module):
    """
    Cosine embedding loss for aligning embeddings in the latent space.
    Used to ensure that similar concepts map to similar latent vectors.
    """
    
    def __init__(self, margin=0.0, reduction="mean"):
        """
        Initialize the cosine embedding loss.
        
        Args:
            margin (float): Margin for the loss
            reduction (str): Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.loss_fn = nn.CosineEmbeddingLoss(margin=margin, reduction=reduction)
        
    def forward(self, z1, z2, y):
        """
        Compute the cosine embedding loss.
        
        Args:
            z1: First set of embeddings
            z2: Second set of embeddings
            y: Labels (1 for similar pairs, -1 for dissimilar pairs)
            
        Returns:
            torch.Tensor: Loss value
        """
        return self.loss_fn(z1, z2, y)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for pulling similar concepts together in latent space
    and pushing dissimilar concepts apart.
    """
    
    def __init__(self, margin=1.0, reduction="mean"):
        """
        Initialize the contrastive loss.
        
        Args:
            margin (float): Margin for negative pairs
            reduction (str): Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        
    def forward(self, z1, z2, y):
        """
        Compute the contrastive loss.
        
        Args:
            z1: First set of embeddings
            z2: Second set of embeddings
            y: Labels (1 for similar pairs, 0 for dissimilar pairs)
            
        Returns:
            torch.Tensor: Loss value
        """
        # Euclidean distance
        dist = F.pairwise_distance(z1, z2)
        
        # Contrastive loss
        loss = y * dist.pow(2) + (1 - y) * F.relu(self.margin - dist).pow(2)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss 